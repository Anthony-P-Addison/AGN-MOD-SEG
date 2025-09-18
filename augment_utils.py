import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from config import Augmentation_config



##### any defintions for augmentations to inputs to agnotic channel  go here, ADD AS NEEDED ######


def spatial_contrast_aug(
    aug_config: Augmentation_config,
    channel_add: list,
    pathology_label: torch.Tensor,
    brain_mask: torch.Tensor,
    batch_img: torch.Tensor,
    blur_boundary_prob: float = 1,  #   1
    blur_sigma_range: tuple = (0.5, 0.5),  # 0.5 0.5
    dilation_iterations: int = 1,
    plot_image: bool = False,
    lam_params: tuple = (0.7, 1),
):
    """
    Apply spatial contrast augmentation by modifying intensity in brain and tumor regions,
    followed by optional spatial blurring of image features directly across the tumor boundary.
    Background (non-brain) regions are preserved.
    """

    # # Validate scale shift configuration
    # if aug_config.uniform_scale_shift and (not aug_config.prob_brain_scale_shift or not aug_config.prob_tumor_scale_shift):
    #     raise ValueError("When uniform_scale_shift is True, both prob_brain_scale_shift and prob_tumor_scale_shift must be True")

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
    # Start with original brain + background
    working_img = img.clone()
    working_pathology_img = img.clone()

    if aug_config.uniform_augs:
        # do evething to working image
        if random.random() < aug_config.uniform_mix_up:
            working_img = MixUp(batch_img,working_img,channel_add,brain_mask_bool)
            significant_aug_applied = True  # Mixup is significant

        if random.random() < aug_config.uniform_invert:
            working_img = torch.where(brain_mask_bool, working_img * -1.0, working_img)
            significant_aug_applied = True  # Inversion is significant

        if random.random() < aug_config.uniform_scale_shift:
            brain_factor_multiply = torch.tensor(random.uniform(*aug_config.brain_factor_multiply), device=img.device, dtype=img.dtype)
            brain_factor_intensity = torch.tensor(random.uniform(*aug_config.brain_factor_intensity), device=img.device, dtype=img.dtype)
            working_img = torch.where(brain_mask_bool, working_img * brain_factor_multiply + brain_factor_intensity, working_img)
            significant_aug_applied = True  # Scale/shift is significant

    else:

        #### Healthy brain tissue Augmentations ####

        # --- 1. INVERSION | BRAIN TISSUE ---
        if random.random() < aug_config.prob_brain_invert:    
            brain_inversion = True
            working_img = torch.where(brain_mask_bool, working_img * -1.0, working_img)
            significant_aug_applied = True  # Inversion is significant

        # --- 2. MixUP | BRAIN TISSUE ---
        if random.random() < aug_config.prob_brain_mixup:
            working_img = MixUp(batch_img,working_img,channel_add,brain_mask_bool,lam_params = lam_params)
            significant_aug_applied = True  # Mixup is significant

        # --- 3. INTENSITY/CONTRAST | BRAIN TISSUE  ---

        # Scale factors relative to the image's statistics
        # brain_factor_intensity = torch.tensor(random.choice([random.uniform(0.1, 0.5), random.uniform(-0.5, -0.1)]), device=img.device, dtype=img.dtype)
        if aug_config.prob_brain_scale_shift:

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

        # --- 0. Pathology Modality Switch (Applied first) ---
        if random.random() < aug_config.prob_lesion_switch:
            possible_tumor_channels = [i for i in range(batch_img.shape[0]-1) if i not in channel_add ]
            tumor_channel = random.choice(possible_tumor_channels)
            working_pathology_img = torch.where(tumor_mask_bool, batch_img[tumor_channel], img)
            significant_aug_applied = True  # Modality switch is significant

        if torch.any(tumor_mask_bool):
            # --- 1. INVERSION | PATHOLOGY TISSUE --
            if random.random() < aug_config.prob_pathology_invert:
                tumor_inversion = True
                working_pathology_img = torch.where(tumor_mask_bool, working_pathology_img * -1.0, working_pathology_img)
                significant_aug_applied = True  # Pathology inversion is significant

            # --- 2. MixUP | PATHOLOGY TISSUE ---
            if random.random() < aug_config.prob_pathology_mixup:
                working_pathology_img = MixUp(batch_img,working_pathology_img,channel_add,tumor_mask_bool,lam_params = lam_params)
                significant_aug_applied = True  # Pathology mixup is significant

            # --- 3. INTENSITY/CONTRAST | PATHOLOGY TISSUE ---
            # Scale factors relative to the image's statistics

            if aug_config.prob_tumor_scale_shift:

                tumor_factor_multiply = torch.tensor(random.uniform(*aug_config.tumor_factor_multiply), device=img.device, dtype=img.dtype)
                tumor_factor_intensity = torch.tensor(random.uniform(*aug_config.tumor_factor_intensity), device=img.device, dtype=img.dtype)
                # tumor_factor_intensity = torch.tensor(random.choice([random.uniform(0.1, 0.5), random.uniform(-0.5, -0.1)]), device=img.device, dtype=img.dtype)

                # Apply tumor factors specifically where tumor_mask_bool is True
                working_pathology_img = torch.where(
                    tumor_mask_bool,
                    working_pathology_img * tumor_factor_multiply + tumor_factor_intensity,
                    working_pathology_img
                )
                # Scale/shift is NOT considered significant augmentation

            working_img = torch.where(tumor_mask_bool, working_pathology_img, working_img)

        # --- Spatial Blurring at Tumor Boundary ---
        # Only apply if significant augmentations have been applied
        if significant_aug_applied and random.random() < blur_boundary_prob:
            working_img = lesion_blur(
                tumor_mask_bool,
                brain_mask_bool,
                working_img,
                blur_sigma_range,
                dilation_iterations,
                img,
            )
    # --- Final Image Assignment and Normalization ---
    augmented_img = working_img

    if plot_image:
        # --- Visualization ---
        plot_augmentation(
            img,
            tumor_mask_bool,
            brain_mask_bool,
            augmented_img,
            brain_factor_multiply,
            brain_factor_intensity,
            tumor_factor_multiply,
            tumor_factor_intensity,
            brain_inversion,
            tumor_inversion,
            tumor_channel,
        )

    return augmented_img


def plot_augmentation(
    img,
    tumor_mask_bool,
    brain_mask_bool,
    augmented_img,
):
    """
    Plot the augmentation parameters and the augmented image.
    """
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
        plt.imshow(img_np[0, :, :, slice_idx], cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Tumor mask overlay
        plt.subplot(1, 4, 2)
        plt.imshow(img_np[0, :, :, slice_idx], cmap="gray")
        plt.imshow(tumor_mask_np[:, :, slice_idx], alpha=0.3, cmap="Reds")
        plt.title("Tumor Mask Overlay")
        plt.axis("off")

        # Brain mask overlay
        plt.subplot(1, 4, 3)
        plt.imshow(img_np[0, :, :, slice_idx], cmap="gray")
        plt.imshow(brain_mask_np[:, :, slice_idx], alpha=0.3, cmap="Blues")
        plt.title("Brain Mask Overlay")
        plt.axis("off")

        # Augmented image
        plt.subplot(1, 4, 4)
        plt.imshow(augmented_np[0, :, :, slice_idx], cmap="gray")
        plt.title("Augmented Image")
        plt.axis("off")

        # Save only, don't show
        plt.tight_layout()
        plt.savefig("spatial_contrast_aug.png", bbox_inches="tight", dpi=300)
        plt.close()


def MixUp(
    x: torch.tensor,
    working_image: torch.tensor,
    channel_add: int,
    mask: torch.tensor,
    lam_params: tuple = (0.7, 1),
):
    """
    Apply mixup contrast augmentation to the input tensor x.
    Args:
        x: torch.tensor, shape (batch_size, 2, height, width)
    Returns:
        mixed_x: torch.tensor, shape (batch_size, 1,height, width)
    """
    possible_channels = [i for i in range(x.shape[0] - 1) if i not in channel_add]
    mix = random.choice(possible_channels)
    mix_image = x[mix]
    lam = np.random.uniform(lam_params[0], lam_params[1])
    mixed_x = torch.mul(working_image[0], lam) + torch.mul(mix_image, (1 - lam))
    mixed_x = torch.where(mask, mixed_x, working_image)
    return mixed_x


def lesion_blur(
    tumor_mask_bool: bool,
    brain_mask_bool: bool,
    working_img: torch.tensor,
    blur_sigma_range: tuple = (0.5, 0.5),
    dilation_iterations: int = 1,
    img: torch.tensor = None,
):
    """
    Apply lesion blur augmentation to the input tensor img.
    """

    tumor_mask_np = tumor_mask_bool[0].cpu().numpy()
    brain_mask_np = brain_mask_bool[0].cpu().numpy()

    dilated_tumor = ndimage.binary_dilation(
        tumor_mask_np, iterations=dilation_iterations, border_value=0
    )

    eroded_tumor_np = ndimage.binary_erosion(
        tumor_mask_np, iterations=dilation_iterations, border_value=0
    )

    outer_boundary_component_np = dilated_tumor & ~tumor_mask_np
    inner_boundary_component_np = tumor_mask_np & ~eroded_tumor_np

    boundary_np = (
        outer_boundary_component_np | inner_boundary_component_np
    ) & brain_mask_np

    if np.any(boundary_np):
        boundary_tensor = torch.from_numpy(boundary_np).to(device=img.device)
        sigma = np.random.uniform(*blur_sigma_range)

        for c in range(working_img.shape[0]):
            channel_data_np = working_img[c].cpu().numpy()
            blurred_channel_np = ndimage.gaussian_filter(channel_data_np, sigma=sigma)
            blurred_channel_tensor = torch.from_numpy(blurred_channel_np).to(
                device=img.device
            )

            working_img[c] = torch.where(
                boundary_tensor, blurred_channel_tensor, working_img[c]
            )
    return working_img
