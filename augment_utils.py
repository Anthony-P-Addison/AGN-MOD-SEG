import torch
import numpy as np
from IPA_aug.imagefilter3d import GINGroupConv3D
import random
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from config import Augmentation_config

##### any defintions for augmentations go here ######


def spatial_contrast_aug(
    aug_config: Augmentation_config,
    channel_add:list,
    pathology_label: torch.Tensor,
    brain_mask: torch.Tensor,
    batch_img: torch.Tensor,
    blur_boundary_prob: float = 1,   #   1 
    blur_sigma_range: tuple = (0.5, 0.5),   # 0.5 0.5 
    dilation_iterations: int = 1,
    plot_image: bool = False


):
    """
    Apply spatial contrast augmentation by modifying intensity in brain and tumor regions,
    followed by optional spatial blurring of image features directly across the tumor boundary.
    Background (non-brain) regions are preserved.
    """
   
    # Track if significant augmentations (beyond scale/shift) have been applied
    significant_aug_applied = False

    # dropped modality
    img = batch_img[channel_add,:,:,:]

    if pathology_label.dim() == 3: pathology_label = pathology_label.unsqueeze(0)
    if brain_mask.dim() == 3: brain_mask = brain_mask.unsqueeze(0)
    if img.dim() != 4: raise ValueError(f"Expected 4D image tensor (C,H,W,D), got {img.shape}")
    if pathology_label.dim() != 4 or pathology_label.shape[0] != 1: raise ValueError(f"Expected 4D pathology label (1,H,W,D), got {pathology_label.shape}")
    if brain_mask.dim() != 4 or brain_mask.shape[0] != 1: raise ValueError(f"Expected 4D brain mask (1,H,W,D), got {brain_mask.shape}")

    tumor_mask_bool = (pathology_label > 0)  # Shape (1, H, W, D), boolean
    brain_mask_bool = (brain_mask > 0)    # Shape (1, H, W, D), boolean

    brain_inversion = None
    tumor_inversion = None
    tumor_channel = None
    brain_factor_multiply = None
    brain_factor_intensity = None
    tumor_factor_multiply = None
    tumor_factor_intensity = None

    # Create a working copy that only contains the brain region initially
    #working_img = torch.where(brain_mask_bool, img, img) # Start with original brain + background
    working_img = img.clone()


    # --- 0. Pathology Modality Switch (Applied first) ---
    if random.random() < aug_config.prob_pathology_switch:
        possible_tumor_channels = [i for i in range(batch_img.shape[0]-1) if i not in channel_add ]
        tumor_channel = random.choice(possible_tumor_channels)
        working_img = torch.where(tumor_mask_bool, batch_img[tumor_channel], img)
        significant_aug_applied = True  # Modality switch is significant

    #### Healthy brain tissue Augmentations ####

    # --- 1. INVERSION | BRAIN TISSUE ---
    if random.random() < aug_config.prob_brain_invert:    
        brain_inversion = True
        working_img = torch.where(brain_mask_bool, working_img * -1.0, working_img)
        significant_aug_applied = True  # Inversion is significant

    # --- 2. MixUP | BRAIN TISSUE ---
    
    if random.random() < aug_config.prob_brain_mixup:
        working_img = MixUp(batch_img,working_img,channel_add,brain_mask_bool)
        significant_aug_applied = True  # Mixup is significant

    # --- 3. INTENSITY/CONTRAST | BRAIN TISSUE  ---
   
    # Scale factors relative to the image's statistics
    # brain_factor_intensity = torch.tensor(random.choice([random.uniform(0.1, 0.5), random.uniform(-0.5, -0.1)]), device=img.device, dtype=img.dtype)
    if aug_config.prob_brain_scale_shift:
        if aug_config.uniform_scale_shift:
            brain_factor_multiply = torch.tensor(random.uniform(*aug_config.brain_factor_multiply), device=img.device, dtype=img.dtype)
            brain_factor_intensity = torch.tensor(random.uniform(*aug_config.brain_factor_intensity), device=img.device, dtype=img.dtype)
        else:
            brain_factor_multiply = torch.tensor(random.uniform(*aug_config.brain_factor_multiply), device=img.device, dtype=img.dtype)
            brain_factor_intensity = torch.tensor(random.uniform(*aug_config.brain_factor_intensity), device=img.device, dtype=img.dtype)
        
        # Apply brain factors to the whole brain region first
        working_img = torch.where(
            brain_mask_bool,
            working_img * brain_factor_multiply + brain_factor_intensity,
            working_img # Keep background unchanged
        )
        # Scale/shift is NOT considered significant augmentation

    ################### Pathology Brain Tissue Augmentations ###################

    if torch.any(tumor_mask_bool):
        # --- 1. INVERSION | PATHOLOGY TISSUE --
        if random.random() < aug_config.prob_pathology_invert:
            tumor_inversion = True
            working_img = torch.where(tumor_mask_bool, working_img * -1.0, working_img)
            significant_aug_applied = True  # Pathology inversion is significant

        # --- 2. MixUP | PATHOLOGY TISSUE ---
        if random.random() < aug_config.prob_pathology_mixup:
            working_img = MixUp(batch_img,working_img,channel_add,tumor_mask_bool)
            significant_aug_applied = True  # Pathology mixup is significant

        # --- 3. INTENSITY/CONTRAST | PATHOLOGY TISSUE ---
        # Scale factors relative to the image's statistics
        if aug_config.prob_tumor_scale_shift:
            if aug_config.uniform_scale_shift:
                tumor_factor_multiply = brain_factor_multiply
                tumor_factor_intensity = brain_factor_intensity
            else:
                tumor_factor_multiply = torch.tensor(random.uniform(*aug_config.tumor_factor_multiply), device=img.device, dtype=img.dtype)
                tumor_factor_intensity = torch.tensor(random.uniform(*aug_config.tumor_factor_intensity), device=img.device, dtype=img.dtype)
                # tumor_factor_intensity = torch.tensor(random.choice([random.uniform(0.1, 0.5), random.uniform(-0.5, -0.1)]), device=img.device, dtype=img.dtype)

            # Apply tumor factors specifically where tumor_mask_bool is True
            working_img = torch.where(
                tumor_mask_bool,
                working_img * tumor_factor_multiply + tumor_factor_intensity,
                working_img
            )
            # Scale/shift is NOT considered significant augmentation

        # --- Spatial Blurring at Tumor Boundary ---
        # Only apply if significant augmentations have been applied
        if significant_aug_applied and random.random() < blur_boundary_prob:
            tumor_mask_np = tumor_mask_bool[0].cpu().numpy()
            brain_mask_np = brain_mask_bool[0].cpu().numpy()

            dilated_tumor = ndimage.binary_dilation(
                tumor_mask_np,
                iterations=dilation_iterations,
                border_value=0
            )

            eroded_tumor_np = ndimage.binary_erosion(
                tumor_mask_np,
                iterations=dilation_iterations,
                border_value=0
            )

            outer_boundary_component_np = dilated_tumor & ~tumor_mask_np
            inner_boundary_component_np = tumor_mask_np & ~eroded_tumor_np

            boundary_np = (outer_boundary_component_np | inner_boundary_component_np) & brain_mask_np

            if np.any(boundary_np):
                boundary_tensor = torch.from_numpy(boundary_np).to(device=img.device)
                sigma = np.random.uniform(*blur_sigma_range)

                for c in range(working_img.shape[0]):
                    channel_data_np = working_img[c].cpu().numpy()
                    blurred_channel_np = ndimage.gaussian_filter(channel_data_np, sigma=sigma)
                    blurred_channel_tensor = torch.from_numpy(blurred_channel_np).to(device=img.device)

                    working_img[c] = torch.where(
                        boundary_tensor,
                        blurred_channel_tensor,
                        working_img[c]
                    )

    # --- Final Image Assignment and Normalization ---
    augmented_img = working_img

    plot_image = plot_image
    if plot_image:
        # --- Visualization ---
        plot_augmentation(img, tumor_mask_bool, brain_mask_bool, augmented_img,brain_factor_multiply,brain_factor_intensity,tumor_factor_multiply,tumor_factor_intensity,brain_inversion,tumor_inversion,tumor_channel)

    return augmented_img




def plot_augmentation(img, tumor_mask_bool, brain_mask_bool, augmented_img,brain_factor_multiply,brain_factor_intensity,tumor_factor_multiply,tumor_factor_intensity,brain_inversion,tumor_inversion,tumor_channel):
        # --- Visualization ---
    if torch.any(tumor_mask_bool):
        # Convert tensors to numpy for plotting
        img_np = img.cpu().numpy()
        augmented_np = augmented_img.cpu().numpy()
        tumor_mask_np = tumor_mask_bool[0].cpu().numpy()
        brain_mask_np = brain_mask_bool[0].cpu().numpy()

        # Select middle slice for visualization
        slice_idx = img_np.shape[-1] // 2

        # Create figure
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_np[0, :, :, slice_idx], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Tumor mask overlay
        plt.subplot(1, 4, 2)
        plt.imshow(img_np[0, :, :, slice_idx],cmap = 'gray')
        plt.imshow(tumor_mask_np[:, :, slice_idx], alpha=0.3, cmap='Reds')
        plt.title('Tumor Mask Overlay')
        plt.axis('off')

        # Brain mask overlay
        plt.subplot(1, 4, 3)
        plt.imshow(img_np[0, :, :, slice_idx],cmap = 'gray')
        plt.imshow(brain_mask_np[:, :, slice_idx], alpha=0.3, cmap='Blues')
        plt.title('Brain Mask Overlay')
        plt.axis('off')

        # Augmented image
        plt.subplot(1, 4, 4)
        plt.imshow(augmented_np[0, :, :, slice_idx], cmap='gray')
        plt.title('Augmented Image')
        plt.axis('off')

        # Add overall title with augmentation parameters
        # plt.suptitle(
        #     f"Augmentation Parameters:\nBrain scale: {brain_factor_multiply:.2f}, shift: {brain_factor_intensity:.2f}\nTumor scale: {tumor_factor_multiply:.2f},shift: {tumor_factor_intensity:.2f}\nBrain inversion: {brain_inversion},Tumor inversion: {tumor_inversion}\nTumour_swap: {tumor_channel}",
        #     y=1.05,
        # )

        # Save only, don't show
        plt.tight_layout()
        plt.savefig('spatial_contrast_aug.png', bbox_inches='tight', dpi=300)
        plt.close()


#def blur_boundary():





def MixUp(x:torch.tensor, working_image:torch.tensor,channel_add:int,mask:torch.tensor,lam_params:tuple = (0.65, 0.65)):
    """
    Apply mixup contrast augmentation to the input tensor x.
    Args:
        x: torch.tensor, shape (batch_size, 2, height, width)
    Returns:
        mixed_x: torch.tensor, shape (batch_size, 1,height, width)
    """
    possible_channels = [i for i in range(x.shape[0]-1) if i not in channel_add ]
    mix = random.choice(possible_channels)
    mix_image = x[mix]
    lam = np.random.uniform(lam_params[0],lam_params[1])

    mixed_x = torch.mul(working_image[0],lam)+ torch.mul(mix_image,(1-lam))
    mixed_x =torch.where(mask,mixed_x,working_image) 
    return mixed_x








########################  EXTRA AUGMENTATIONS #########################################

def mixup1_augmentation(x: torch.tensor, all_mod_dropped: bool, one_mod_dropped:bool, two_not_dropped:bool, mod_3: bool):
    """Returns mixed inputs based on the MIxUp method.
    Takes into consideration probability of modality being dropped and which modalities to mix up.
    """
    if all_mod_dropped:
        # mixup data both dropped.
        lam = np.random.uniform(0, 1) 
        mixed_x = torch.mul(x[0],lam) + torch.mul((x[1]),(1-lam))
    elif one_mod_dropped:
        # one mixup data dropped and other not dropped.
        lam = np.random.uniform(0, 0.8)
        mixed_x = torch.mul(x[0],lam) + torch.mul(x[1],(1-lam))
    elif two_not_dropped:
        lam =np.random.beta(2,2)
        mixed_x = torch.mul(x[0],lam) + torch.mul(x[1],(1-lam))
    # mixing of three modalities only if they are all dropped. 
    elif mod_3:
        lam1, lam2, lam3 = np.random.dirichlet([2,2,2], size=1)[0]
        lam1, lam2, lam3 = float(lam1), float(lam2), float(lam3)
        mixed_x = torch.mul(x[0],lam1) + torch.mul(x[1],lam2) + torch.mul(x[2],lam3)   
    ###################################################################################
    import matplotlib.pyplot as plt

    # # Assuming x is a batch of images with shape (batch_size, channels, height, width)
    # # Convert the tensor to a numpy array and transpose to (height, width, channels) for plotting
    mixed_x_np = mixed_x.cpu().numpy()

    mixed_x_slice = mixed_x_np[:, :, 64]

    # Plot original images
    x_np_0 = x[0].cpu().numpy()
    x_np_1 = x[1].cpu().numpy()

    if mod_3:
        x_np_2 = x[2].cpu().numpy()
        x_slice_2 = x_np_2[:, :, 64]

    x_slice_0 = x_np_0[:, :, 64]
    x_slice_1 = x_np_1[:, :, 64]

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
    plt.subplot(1, 4, 1)
    plt.imshow(x_slice_0, cmap="gray")
    plt.title("Original Image 1")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(x_slice_1, cmap="gray")
    plt.title("Original Image 2")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    if mod_3 == True:
        plt.imshow(x_slice_2, cmap="gray")
    plt.title("Original Image 3")
    plt.axis("off")


    plt.subplot(1, 4, 4)
    plt.imshow(mixed_x_slice, cmap="gray")
    plt.title("Mixed Image")
    plt.axis("off")
    plt.show()
    plt.savefig('mixed_image.png')
    print("Image saved as 'mixed_image.png'")
    print(f"Before mixing - Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
    print(f"After mixing - Min: {mixed_x.min().item():.4f}, Max: {mixed_x.max().item():.4f}")
    # Calculate and print pixel distribution statistics
    print("\nPixel Distribution Analysis:")
    print("Before augmentation:")
    print(f"Mean: {x.mean().item():.4f}")
    print(f"Median: {torch.median(x).item():.4f}")
    print(f"Standard Deviation: {x.std().item():.4f}")
    print(f"25th percentile: {torch.quantile(x, 0.25).item():.4f}")
    print(f"75th percentile: {torch.quantile(x, 0.75).item():.4f}")
    
    print("\nAfter augmentation:")
    print(f"Mean: {mixed_x.mean().item():.4f}")
    print(f"Median: {torch.median(mixed_x).item():.4f}")
    print(f"Standard Deviation: {mixed_x.std().item():.4f}")
    print(f"25th percentile: {torch.quantile(mixed_x, 0.25).item():.4f}")
    print(f"75th percentile: {torch.quantile(mixed_x, 0.75).item():.4f}")
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


####  IPA from CHENG et al. ####

def IPA_module(x:torch.tensor,device_id ):
    """Apply IPA transformation to the input tensor x.
    """ 
    # TODO: hard coded for now sort this later and only want to apply these augments to the brain and not the background.
    ipa_config_dict = {
    'epsilon': 0.05,
    'xi': 1e-6,
    'control_point_spacing': [8,8,8],
    'downscale': 2,   #ase downscale factor to reduce memory usage
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

    # FIXME: trying to do augmentation to foregorund- need to make this more precise
    threshold = torch.min(x)
    x_ed = (torch.where(x > threshold + 0.01, x, torch.tensor(0.0, dtype=x.dtype)))
   
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
    nifti_file = 'data/TBI/Images/CENTER-TBI-2020-6_Sub-021-4MbF392_Site-06-a72b20.nii.gz'
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

    result = mixup_data_causality(cropped_tensor, device_id='1', aug_type='GIN_IPA', all_mod_dropped=True, one_mod_dropped=False, two_not_dropped=False, mod_3=False)

    import matplotlib.pyplot as plt

    # Assuming result is a tensor with shape (1, 128, 128, 128)
    result_np = result.cpu().numpy()

    # Select a slice to plot
    slice_index = 70
    result_slice = result_np[0,0, :, :, slice_index]
    
    cropped_tensor_np = cropped_tensor.cpu().numpy()
    original_slice = cropped_tensor_np[0, :, :, slice_index]

    
    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap="gray")
    plt.title("Original Image Slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result_slice, cmap="gray")
    plt.title("Transformed Image Slice")
    plt.axis("off")
    plt.show()

    # I only want to apply to the brains itself and not the bakcground 
