import torch
import numpy as np
import torchvision
from IPA_aug.imagefilter3d import GINGroupConv3D


##### any defintions for augmentations go here ######


def mixup1_augmentation(x: torch.tensor, all_mod_dropped: bool, one_mod_dropped:bool, two_not_dropped:bool, mod_3: bool):
    """Returns mixed inputs based on the MIxUp method.
    Takes into consideration probability of modality being dropped and which modalities to mix up.
    """
    if all_mod_dropped:
        # mixup data both dropped.
        lam = np.random.uniform(0, 1) 
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))
    elif one_mod_dropped:
        # one mixup data dropped and other not dropped.
        lam = np.random.uniform(0, 0.8)
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))
    elif two_not_dropped:
        lam =np.random.beta(2,2)
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))
    # mixing of three modalities only if they are all dropped. 
    elif mod_3:
        lam1, lam2, lam3 = np.random.dirichlet([2,2,2], size=1)[0]
        lam1, lam2, lam3 = float(lam1), float(lam2), float(lam3)
        mixed_x = torch.mul(mix_operations(x[0]),lam1) + torch.mul(mix_operations(x[1]),lam2) + torch.mul(mix_operations(x[2]),lam3)   
    ###################################################################################
    # import matplotlib.pyplot as plt

    # # Assuming x is a batch of images with shape (batch_size, channels, height, width)
    # # Convert the tensor to a numpy array and transpose to (height, width, channels) for plotting
    # mixed_x_np = mixed_x.cpu().numpy()

    # mixed_x_slice = mixed_x_np[:, :, 64]

    # # Plot original images
    # x_np_0 = x[0].cpu().numpy()
    # x_np_1 = x[1].cpu().numpy()

    # if mod_3:
    #     x_np_2 = x[2].cpu().numpy()
    #     x_slice_2 = x_np_2[:, :, 64]

    # x_slice_0 = x_np_0[:, :, 64]
    # x_slice_1 = x_np_1[:, :, 64]

    # plt.figure(figsize=(12, 5))



    # plt.subplot(1, 4, 1)
    # plt.imshow(x_slice_0, cmap="gray")
    # plt.title("Original Image 1")
    # plt.axis("off")

    # plt.subplot(1, 4, 2)
    # plt.imshow(x_slice_1, cmap="gray")
    # plt.title("Original Image 2")
    # plt.axis("off")

    # plt.subplot(1, 4, 3)
    # if mod_3 == True:
    #     plt.imshow(x_slice_2, cmap="gray")
    # plt.title("Original Image 3")
    # plt.axis("off")


    # plt.subplot(1, 4, 4)
    # plt.imshow(mixed_x_slice, cmap="gray")
    # plt.title("Mixed Image")
    # plt.axis("off")
    # plt.show()
    # # plt.savefig('mixed_image.png')
    # # print("Image saved as 'mixed_image.png'")
    # plt.subplot(1, 4, 1)
    # plt.imshow(x_slice_0, cmap="gray")
    # plt.title("Original Image 1")
    # plt.axis("off")

    # plt.subplot(1, 4, 2)
    # plt.imshow(x_slice_1, cmap="gray")
    # plt.title("Original Image 2")
    # plt.axis("off")

    # plt.subplot(1, 4, 3)
    # if mod_3 == True:
    #     plt.imshow(x_slice_2, cmap="gray")
    # plt.title("Original Image 3")
    # plt.axis("off")


    # plt.subplot(1, 4, 4)
    # plt.imshow(mixed_x_slice, cmap="gray")
    # plt.title("Mixed Image")
    # plt.axis("off")
    # plt.show()
    # # plt.savefig('mixed_image.png')
    # print("Image saved as 'mixed_image.png'")
    return mixed_x

# Example usage in mix_operations
def mix_operations(
    x,
    translation=True,
    inversion=True,
    multiply = True,):
    """Apply mixup operations with optional translation, inversion, and contrast stretching.
    - Multiply
    - shift
    - invert"""
    if multiply:
        # Apply multiplication
        threshold = np.min(x)+0.01
        multiplicate = (np.random.uniform(0.9, 1.1))
        x = torch.where(x > threshold, x * multiplicate, x)
    if translation:
        translation = np.random.uniform(-0.3, 0.3)
        x = x + translation
    if inversion:
        # Apply inversion
        # invert_prob = torch.tensor(
        #     np.random.choice(
        #         [np.random.uniform(0.9, 1.1), np.random.uniform(-0.9, -1.1)]
        #     )
        # )
        threshold1 = translation + threshold
        invert_prob = torch.tensor(np.random.choice([1, -1]))
        x = torch.where(x > threshold1, x * invert_prob, x)
    return x



####### GIN module from cheng et al. #######

def GIN_module(x,device_id ):
    """Apply GIN transformation to the input tensor x.
    """
    x= x.unsqueeze(0)
    size = x.shape[1]
    #ensure correct dimension    assert len(x.shape) ==5
    augmenter = GINGroupConv3D(out_channel = size, in_channel = size, interim_channel= 4,scale_pool= [1,3], n_layer = 2, device_id= device_id)
    out = augmenter(x)
    return out


# def mixup_data(x: torch.tensor, all_mod_dropped: bool, one_mod_dropped:bool, two_not_dropped:bool, mod_3: bool):
#     """Returns mixed inputs, pairs of targets, and lambda
#     """
#     if all_mod_dropped or one_mod_dropped or two_not_dropped or mod_3:
#         gin_x=GIN_module(x)
#     return gin_x


####  IPA from CHENG et al. ####



def IPA_module(x:torch.tensor,device_id = '1'):
    """Apply IPA transformation to the input tensor x.
    """ 
    # TODO: hard coded for now sort this later and only want to apply these augments to the brain and not the background.
    ipa_config_dict = {
    'epsilon': 0.3,
    'xi': 1e-6,
    'control_point_spacing': [8, 8, 8],
    'downscale': 2,  # Increase downscale factor to reduce memory usage
    'data_size': [1, 1, 128, 128, 128],
    'interpolation_order': 3,
    'init_mode': 'gaussian',
    'space': 'log'
    }
    from IPA_aug.adv_bias import AdvBias3D
    ##GIN##
    gin_module = GIN_module(x,device_id)
    ##IPA##
    augmentor = AdvBias3D(config_dict=ipa_config_dict, use_gpu=True,device_id = device_id)
    augmentor.init_parameters()
    transformed = augmentor.forward(gin_module)
    # error =  transformed - gin_module 
    #print('sum error', torch.sum(error))

    return transformed


def mixup_data_causality(x: torch.tensor, device_id, aug_type:str, all_mod_dropped: bool, one_mod_dropped:bool, two_not_dropped:bool, mod_3: bool):
    """Returns mixed inputs, pairs of targets, and lambda
    """

    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    x = x.to(device)
  
    threshold = torch.min(x)+0.01

    ###FIXME: REMOVE 
    x_ed = (torch.where(x > threshold, x, torch.tensor(0.0, dtype=x.dtype)))
   
    if all_mod_dropped or one_mod_dropped or two_not_dropped or mod_3:
        if aug_type == 'GIN':
            aug_x=GIN_module(x_ed,device_id)
        elif aug_type == 'GIN_IPA':
             aug_x=IPA_module(x_ed,device_id = device_id)
        else:
            raise NotImplementedError(f'Unknown aug type: {aug_type}')
        # replace the background with the original background
        x = torch.where(x < threshold, x, aug_x)
    return x






















if __name__ == "__main__":

    #### unit test for gin/ipa #### 

    # random array 
    #mixup_data(torch.rand( 1, 128, 128, 128),device_id = '1', aug_type = 'GIN_IPA', all_mod_dropped=True, one_mod_dropped=False, two_not_dropped=False, mod_3=False)

    # load up nifty and do it with that. 

    # Load NIfTI file
    import nibabel as nib
    nifti_file = 'data/BRATS/Images/BRATS_056_normed_on_mask.nii.gz'
    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()

    # Crop to 128^3
    crop_size = 128
    nifti_data = np.transpose(nifti_data, (3, 0, 1, 2))[0]
    x, y, z = nifti_data.shape    # take the fo
    start_x = (x - crop_size) // 2
    start_y = (y - crop_size) // 2
    start_z = (z - crop_size) // 2
    cropped_data = nifti_data[start_x:start_x+crop_size, start_y:start_y+crop_size, start_z:start_z+crop_size]

    # Convert to torch tensor
    cropped_tensor = torch.tensor(cropped_data, dtype=torch.float32).unsqueeze(0)

    device_id = '1'
    cuda_id = "cuda:" + str(device_id)
    device = torch.device(cuda_id)
    cropped_tensor = cropped_tensor.to(device)

    # Apply mixup_data function

    result = mixup_data(cropped_tensor, device_id='1', aug_type='GIN_IPA', all_mod_dropped=True, one_mod_dropped=False, two_not_dropped=False, mod_3=False)

    import matplotlib.pyplot as plt

    # Assuming result is a tensor with shape (1, 128, 128, 128)
    result_np = result.cpu().numpy()

    # Select a slice to plot
    slice_index = 50
    result_slice = result_np[0,0, :, :, slice_index]

    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.imshow(result_slice, cmap="gray")
    plt.title("Transformed Image Slice")
    plt.axis("off")
    plt.show()

    # I only want to apply to the brains itself and not the bakcground 
    


