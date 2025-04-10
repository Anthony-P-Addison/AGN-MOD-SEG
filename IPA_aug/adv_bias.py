# code adapted from Chen Chen's implementation
# modified for 3D domain generalization
import torch
import torch.nn.functional as F
import numpy as np
from .adv_transformation_base import AdvTransformBase3D
from pdb import set_trace



# TODO sort into 3d 

def rescale_intensity(data,new_min=0,new_max=1,eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c , h, w,d = data.size(0),data.size(1),data.size(2), data.size(3),data.size(4)
    data = data.view(bs*c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    # co1818: in adjust to pytorch 0.4 for wtbenv
    #old_max = torch.max(data, dim=1, keepdim=True)[0]
    #old_min = torch.min(data, dim=1, keepdim=True)[0]

    new_data = (data - old_min+eps) / (old_max - old_min + eps)*(new_max-new_min)+new_min
    new_data = new_data.view(bs, c, h, w, d)
    return new_data


def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=torch.float32, device='cuda'):
    '''
    generate bspline 3D kernel matrix.
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation
    :param asTensor:
    :param dtype: data type
    :param use_gpu: bool
    :return:
    '''
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv3d(kernel, kernel_ones, padding=(i * padding).tolist()) / (sigma[0] * sigma[1] * sigma[2])

    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()
    


class AdvBias3D(AdvTransformBase3D):
    """
     Adv Bias field for 3D data
    """
    def __init__(self,
                 config_dict={
                  'epsilon':0.3,
                 'xi':1e-6,
                 'control_point_spacing':[32,32,32],
                 'downscale':2,
                 'data_size':[2,1,64,128,128],
                 'interpolation_order':3,
                 'init_mode':'gaussian',
                 'space':'log'},
                 use_gpu:bool = True, 
                 debug: bool = False,
                 device_id = '0'
                 ):
        super(AdvBias3D, self).__init__(config_dict=config_dict,use_gpu=use_gpu,debug=debug)
        self.param=None

        self.cuda_id = "cuda:" + str(device_id)
        self.device = torch.device(self.cuda_id)


    def init_config(self,config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.epsilon = config_dict['epsilon']
        self.xi = config_dict['xi']
        self.data_size = config_dict['data_size']
        self.control_point_spacing = config_dict['control_point_spacing']
        self.downscale = config_dict['downscale']
        self.interpolation_order = config_dict['interpolation_order']
        self.space=config_dict['space']
        self.init_mode=config_dict['init_mode']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        
        #self._device = 'cuda' if self.use_gpu else 'cpu'

        self._dim = len(self.control_point_spacing)
        self.spacing = self.control_point_spacing
        self._dtype = torch.float32
        self.batch_size = self.data_size[0]
        self._image_size = np.array([self.data_size[2], self.data_size[3], self.data_size[4]])
        self.magnitude = self.epsilon
        self.order = self.interpolation_order
        self.downscale = self.downscale  # reduce image size to save memory

        self.use_log = False  # if self.space == 'log' else False

        # construct and initialize control points grid with random values
        self.param, self.interp_kernel, self.bias_field = self.init_bias_field()
        return self.param

    def make_small_parameters(self):
        raise Exception('by co1818: should not call a detaching op')
        self.param = self.unit_normalize(self.param, p_type='l2')*self.xi
        self.param = self.param.detach()
        self.param.requires_grad=True
        return self.param

    def rescale_parameters(self, power_iteration=False):
        # restrict control points values in the 1-ball space
        self.param = self.unit_normalize(self.param, p_type='l2')

    def optimize_parameters(self, power_iteration=False, step_size=1, upd_direction='GA'):
        '''
        Args:
            upd_direction: direction of update. GA for gradient ascend in original implementation. GD for descent.
        '''
        if self.debug: 
            print('optimize bias')
        grad = self.unit_normalize(self.param.grad, p_type='l2')
        if power_iteration:
            raise Exception('co1818: behavior unknown')
            self.param = grad.clone().detach()
        else:
            if upd_direction == 'GA':
                # Gradient ascent
                self.param = self.param + step_size * grad.detach()
                self.param = self.param.clone().detach()
            elif upd_direction == 'GD':
                self.param = self.param - step_size * grad.detach()
                self.param = self.param.clone().detach()
            else:
                raise NotImplementedError(f'Unknown updating direction {upd_direction}')

        return self.param

    def set_parameters(self, param):
        raise Exception('by co1818: should not call a detached op')
        self.param = param.detach()

    def forward(self, data):
        '''
        forward the data to get transformed data
        :param data: input 3D images x, N5DHW
        :return:
        tensor: transformed images
        '''
        assert self.param is not None, 'init param before transform data'

        bias_field = self.compute_smoothed_bias(self.param)
        bias_field = self.rescale_bias(bias_field, magnitude=self.epsilon)
        self.bias_field = bias_field
        self.diff = bias_field

        # in case the input image is a multi-channel input
        if bias_field.size(1) < data.size(1):
            bias_field = bias_field.expand(data.size())

        transformed_input = bias_field * data
        if self.debug:
            print('bias transformed', transformed_input.size())
        return transformed_input

    def backward(self, data):
        if self.debug:
            print('max magnitude', torch.max(torch.abs(self.bias_field-1)))
        return data

    def predict_forward(self, data):
        return data

    def predict_backward(self, data):
        return data

    def init_bias_field(self, init_mode=None):
        '''
        init cp points, interpolation kernel, and resulted bias field.
        :param batch_size:
        :param spacing: tuple of ints
        :param order:
        :return:bias field
        '''
        if init_mode is None:
            mode = self.init_mode

        # set up cpoints grid
        self._stride = np.array(self.spacing)  # 32 or 24
        cp_grid = np.ceil(np.divide(self._image_size/(1.0*self.downscale), self._stride)).astype(dtype=int)
        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # add one control point outside each side
        cp_grid = cp_grid + 2
        # image size with additional control points
        new_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)
        # center image between control points
        image_size_diff = inner_image_size - self._image_size/(1.0*self.downscale)
        image_size_diff_floor = np.floor((np.abs(image_size_diff)/2))*np.sign(image_size_diff)
        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor
        self.cp_grid = [self.batch_size, 1] + cp_grid.tolist()

        # initialize control points parameters for optimization
        if mode == 'gaussian':
            self.param = torch.ones(*self.cp_grid).normal_(mean=0, std=1)
        elif mode == 'identity':
            # static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise NotImplementedError

        self.param = self.unit_normalize(self.param, p_type='l2')
        self.param = self.param.to(dtype=self._dtype, device=self.device)

        # convert to integer
        self._stride = self._stride.astype(dtype=int).tolist()
        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [self.batch_size, 1] + new_image_size.astype(dtype=int).tolist()
        # initialize interpolation kernel
        self.interp_kernel = self.get_bspline_kernel(order=self.order, spacing=self.spacing)      ####TODO is breaking here
        self.interp_kernel = self.interp_kernel.to(self.param.device)               
        self.bias_field = self.compute_smoothed_bias(self.param, padding=self._padding, stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)
        if self.debug:
            print('initialize {} control points'.format(str(self.param.size())))

        return self.param, self.interp_kernel, self.bias_field

    def reset_bias_value(self):
        '''
        Reset param (control point values)
        '''
        del self.param
        del self.bias_field

        mode = self.init_mode
        if mode == 'gaussian':
            self.param = torch.ones(*self.cp_grid).normal_(mean=0, std=1)
        elif mode == 'identity':
            # static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise NotImplementedError

        self.param = self.unit_normalize(self.param, p_type='l2')
        self.param = self.param.to(dtype=self._dtype, device=self._device)

        self.bias_field = self.compute_smoothed_bias(self.param, padding=self._padding, stride=self._stride)
        self.bias_field = self.rescale_bias(self.bias_field)

        return self.param, self.bias_field

    def compute_smoothed_bias(self, cpoint=None, interpolation_kernel=None, padding=None, stride=None):
        '''
        generate bias field given the cpoints N*1*k*l*m
        :return: bias field bs*1*D*H*W
        '''
        if interpolation_kernel is None:
            interpolation_kernel = self.interp_kernel
        if padding is None:
            padding = self._padding
        if stride is None:
            stride = self._stride
        if cpoint is None:
            cpoint = self.param

        bias_field = F.conv_transpose3d(cpoint, interpolation_kernel,
                                        padding=padding, stride=stride, groups=1)
        # crop bias
        bias_field_tmp = bias_field[:, :,
                         stride[0] + self._crop_start[0]:-stride[0] - self._crop_end[0],
                         stride[1] + self._crop_start[1]:-stride[1] - self._crop_end[1],
                         stride[2] + self._crop_start[2]:-stride[2] - self._crop_end[2]]

        # recover bias field to original image resolution for efficiency
        if self.debug: 
            print('after interp, size:', bias_field_tmp.size())
        scale_factor = self._image_size[0] / bias_field_tmp.size(2)
        if scale_factor > 1:
            upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode='trilinear',
                                          align_corners=True)
            diff_bias = upsampler(bias_field_tmp)
        else:
            diff_bias = bias_field_tmp

        bias_field = torch.exp(diff_bias)

        return bias_field

    def rescale_bias(self, bias_field, magnitude=None):
        """
        Rescale the bias field so that values fall in [1-magnitude, 1+magnitude]
        Args:
            bias_field ([torch 5d tensor]): Bias field tensor
            magnitude ([scalar], optional): Scaling factor. Defaults to use predefined value.

        Returns:
            Rescaled bias field
        """
        if magnitude is None:
            magnitude = self.magnitude
        assert magnitude > 0

        bias_field = rescale_intensity(bias_field, 1-magnitude, 1+magnitude)

        if self.debug:
            print('L_infinity: max|bias-id|', torch.max(torch.abs(bias_field-1)))
        return bias_field

    def get_bspline_kernel(self, spacing, order=3):
        '''
        :param order: bspline order, default to 3
        :param spacing: tuple of int: spacing between control points along d, h and w.
        :return: kernel matrix
        '''
        self._kernel = bspline_kernel_3d(spacing, order=order, asTensor=True, dtype=self._dtype, device=self.device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.to(dtype=self._dtype, device=self.device)
        return self._kernel

    def get_name(self):
        return 'bias3d'
    
    def is_geometric(self):
        return 0

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import nibabel as nib 
    import numpy as np

    # Load a NIfTI image and create a numpy array
    nifti_image_path = '/home/magd6292/Documents/wentian_clone/MultiUnet/data/ATLAS/Images/sub-r001s001_normed.nii.gz'
    nifti_image = nib.load(nifti_image_path)
    images= nifti_image.get_fdata()
    # Crop a random 128x128x128 section from the image
    x, y, z = images.shape[0], images.shape[1], images.shape[2]
    crop_size = 128
    x_start = np.random.randint(0, x - crop_size)
    y_start = np.random.randint(0, y - crop_size)
    z_start = np.random.randint(0, z - crop_size)
    images = images[x_start:x_start + crop_size, y_start:y_start + crop_size, z_start:z_start + crop_size]
    

    cuda_id = "cuda:" + str(1)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)
    
   
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # # Create 3D test images
    # images = torch.ones(2, 1, 128, 128, 128).to(device)
    # images[:, :, ::2, ::2, ::2] = 2.0
    # images[:, :, ::3, ::3, ::3] = 3.0
    # images[:, :, ::1, ::1, ::1] = 1.0
    # images = images.float()
    # images.requires_grad = False
    print('input:', images)


    config_dict = {
        'epsilon': 0.3,
        'xi': 1e-6,
        'control_point_spacing': [8, 8, 8],
        'downscale': 2,  # Increase downscale factor to reduce memory usage
        'data_size': [1, 1, 128, 128, 128],
        'interpolation_order': 3,
        'init_mode': 'gaussian',
        'space': 'log'
    }
    augmentor = AdvBias3D(config_dict=config_dict, use_gpu=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    error = transformed - images
    print('sum error', torch.sum(error))

    # Visualize middle slices in greyscale
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(images.detach().cpu().numpy()[0, 0, 32, :, :], cmap='gray')
    plt.title('Original (middle slice)')

    plt.subplot(232)
    plt.imshow(transformed.detach().cpu().numpy()[0, 0, 32, :, :], cmap='gray')
    plt.title('Transformed (middle slice)')

    plt.subplot(233)
    plt.imshow((transformed/images).detach().cpu().numpy()[0, 0, 32, :, :], cmap='gray')
    plt.title('Ratio (middle slice)')

    plt.subplot(234)
    plt.imshow(images.detach().cpu().numpy()[0, 0, 64, :, :], cmap='gray')
    plt.title('Original (another slice)')

    plt.subplot(235)
    plt.imshow(transformed.detach().cpu().numpy()[0, 0, 64, :, :], cmap='gray')
    plt.title('Transformed (another slice)')

    plt.subplot(236)
    plt.imshow((transformed/images).detach().cpu().numpy()[0, 0, 64, :, :], cmap='gray')
    plt.title('Ratio (another slice)')

    plt.savefig('test_bias3d.png')
