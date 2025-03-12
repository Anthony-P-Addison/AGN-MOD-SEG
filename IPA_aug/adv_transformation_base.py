# code adapted from Chen Chen
# updated to explicitly support 3D transformations

import torch


class AdvTransformBase3D(torch.nn.Module):
    """
     Adv Transformer base for 3D data
    """

    def __init__(self,
                 config_dict={'size': 1,
                             'mean': 0,
                             'std': 0.1,
                             'xi': 1e-6
                             },
                 use_gpu: bool = True, debug: bool = False):
        '''
        Base class for 3D adversarial transformations
        '''
        super(AdvTransformBase3D, self).__init__()
        self.config_dict = config_dict
        self.param = None
        self.is_training = False
        self.use_gpu = use_gpu
        self.debug = debug
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        if self.debug: print('init base class')
        self.size = config_dict['size']
        self.mean = config_dict['mean']
        self.std = config_dict['std']
        self.xi = config_dict['xi']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformation parameters
        '''
        self.init_config(self.config_dict)
        noise = torch.randn(self.size, device=self.device, dtype=torch.float32) * self.std + self.mean
        self.param = noise
        return noise

    def set_parameters(self, param):
        self.param = param.detach()

    def make_small_parameters(self):
        self.param = self.xi * self.unit_normalize(self.param)

    def get_parameters(self):
        return self.param

    def train(self):
        self.is_training = True
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def eval(self):
        self.param.requires_grad = False
        self.is_training = False

    def rescale_parameters(self, power_iteration=False):
        self.param = self.xi * self.unit_normalize(self.param)
        return self.param

    def optimize_parameters(self, power_iteration=False, step_size=1, upd_direction='GA'):
        '''
        Args:
            power_iteration: Whether to use power iteration
            step_size: Step size for gradient updates
            upd_direction: Direction of update (GA for gradient ascent, GD for gradient descent)
        '''
        grad = self.param.grad.sign()
        if self.debug: 
            print('grad', grad.size())
        
        if power_iteration:
            self.param = grad.detach()
        else:
            if upd_direction == 'GA':
                # Gradient ascent
                self.param = self.param + step_size * grad.detach()
            elif upd_direction == 'GD':
                # Gradient descent
                self.param = self.param - step_size * grad.detach()
            else:
                raise NotImplementedError(f'Unknown updating direction {upd_direction}')
                
            self.param = self.param.detach()
            
        return self.param

    def forward(self, data):
        '''
        forward the data to get transformed data
        :param data: input 3D images x, NCDHW
        :return:
        tensor: transformed images
        '''
        assert self.param is not None, 'init param before transform data'
        transformed_input = data + self.param
        if self.debug:
            print('transformed', transformed_input.size())
        return transformed_input

    def backward(self, data):
        assert self.param is not None, 'init param before transform data'
        warped_back_output = data - self.param
        if self.debug:
            print('back:', warped_back_output.size())
        return warped_back_output

    def predict_forward(self, data):
        '''
        Forward pass during prediction phase
        '''
        return self.forward(data)

    def predict_backward(self, data):
        '''
        Backward pass during prediction phase
        '''
        return self.backward(data)

    def unit_normalize(self, d, p_type='l2'):
        '''
        Normalize the parameters based on different norms
        
        Args:
            d: Input tensor
            p_type: Type of normalization ('l1', 'l2', 'infinity')
            
        Returns:
            Normalized tensor
        '''
        # Handle different tensor dimensions for 3D data
        if len(d.shape) == 5:  # For 5D tensors (B,C,D,H,W)
            view_size = d.size(0)
            reshape_dim = (-1,)
            restore_view = d.size()
            final_view = (d.size(0), 1, 1, 1, 1)
        else:
            view_size = d.size(0)
            reshape_dim = (-1,)
            restore_view = d.size()
            final_view = (d.size(0),) + (1,) * (len(d.size()) - 1)
        
        if p_type == 'l1':
            d_flatten = d.view(view_size, *reshape_dim)
            norm = d_flatten.norm(p=1, dim=1, keepdim=True)
            d_normalized = d_flatten.div(norm.expand_as(d_flatten))
            return d_normalized.view(restore_view)
        
        elif p_type == 'infinity':
            d_abs_max = torch.max(
                torch.abs(d.view(view_size, -1)), 1, keepdim=True)[0].view(final_view)
            d = d / (1e-20 + d_abs_max)  # d' = d/d_max
            
        if p_type == 'l2':
            d_abs_max = torch.max(
                torch.abs(d.view(view_size, -1)), 1, keepdim=True)[0].view(final_view)
            d = d / (1e-20 + d_abs_max)  # d' = d/d_max
            
            # Compute L2 norm considering all dimensions except batch
            d = d / torch.sqrt(1e-6 + torch.sum(
                torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))  # d'/sqrt(d'^2)
            
        return d

    def is_geometric(self):
        '''
        Whether this is a geometric transformation
        '''
        return 0

    def get_name(self):
        '''
        Get the name of the transformation
        '''
        return 'base3d'


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example with 3D data
    images = torch.zeros((2, 1, 8, 8, 8)).cuda()
    print('input:', images.shape)
    
    augmentor = AdvTransformBase3D(config_dict={
        'size': (2, 1, 8, 8, 8),  # Match the input shape
        'mean': 0,
        'std': 0.1,
        'xi': 1e-6
    }, debug=True)
    
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    recovered = augmentor.backward(transformed)
    error = recovered - images
    print('Sum error:', torch.sum(error))
    
    # Display a slice for visualization
    if transformed.size(2) > 4:  # If we have depth dimension
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(images[0, 0, 4].cpu().numpy())
        plt.title('Original (middle slice)')
        
        plt.subplot(132)
        plt.imshow(transformed[0, 0, 4].cpu().numpy())
        plt.title('Transformed (middle slice)')
        
        plt.subplot(133)
        plt.imshow(recovered[0, 0, 4].cpu().numpy())
        plt.title('Recovered (middle slice)')
        
        plt.savefig('test_base3d.png')