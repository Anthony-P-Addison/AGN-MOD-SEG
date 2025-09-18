import torch
import nibabel as nib
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import random
from config import Database_config, Augmentation_config
from augment_utils import spatial_contrast_aug

# Set publication aesthetics for high quality output - prevent any interpolation
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'image.interpolation': 'nearest',  # Prevent blurring
    'image.resample': False  # No resampling
})

# Load configs
database_config = Database_config()
aug_config = Augmentation_config()

# Choose BRATS dataset which has clear, large tumors
dataset = "BRATS"

# Get file paths
img_path = database_config.img_path[dataset]
seg_path = database_config.seg_path[dataset]
mask_path = database_config.mask_path[dataset]

# Find all files
images = sorted(glob(os.path.join(img_path, "*.*")))
segs = sorted(glob(os.path.join(seg_path, "*.*")))
masks = sorted(glob(os.path.join(mask_path, "*.*")))

# Create output directory
output_dir = "augmentation_results"
os.makedirs(output_dir, exist_ok=True)

# Process all subject indices
subject_indices = [1, 2]

def process_subject(subject_idx, images, segs, masks):
    """Process a single subject and return the data and best slice"""
    try:
        # Load data
        img_nifti = nib.load(images[subject_idx])
        seg_nifti = nib.load(segs[subject_idx])
        mask_nifti = nib.load(masks[subject_idx])
        
        # Convert to tensors
        img_data = torch.from_numpy(img_nifti.get_fdata()).float()
        seg_data = torch.from_numpy(seg_nifti.get_fdata()).float()
        mask_data = torch.from_numpy(mask_nifti.get_fdata()).float()
        
        # Reshape if needed
        if img_data.dim() == 3:
            img_data = img_data.unsqueeze(0)
        elif img_data.dim() == 4 and img_data.shape[0] > 10:
            img_data = img_data.permute(3, 0, 1, 2)
        
        if seg_data.dim() == 3:
            seg_data = seg_data.unsqueeze(0)
            
        if mask_data.dim() == 3:
            mask_data = mask_data.unsqueeze(0)
        
        # Find a slice near the middle with a large tumor
        middle_range = range(seg_data.shape[2]//3, 2*seg_data.shape[2]//3)
        largest_tumor_size = 0
        best_slice = None
        
        for i in middle_range:
            tumor_size = seg_data[0, :, i, :].sum().item()
            if tumor_size > largest_tumor_size:
                largest_tumor_size = tumor_size
                best_slice = i
        
        # If we didn't find a good slice, use the middle one
        if best_slice is None:
            best_slice = seg_data.shape[2] // 2
            largest_tumor_size = seg_data[0, :, best_slice, :].sum().item()
        
        print(f"Subject {subject_idx}: tumor size {largest_tumor_size}, slice {best_slice}")
        return img_data, seg_data, mask_data, best_slice, largest_tumor_size
        
    except Exception as e:
        print(f"Error processing subject {subject_idx}: {e}")
        return None, None, None, None, 0

# Define augmentation settings
aug_types = [
    {"name": "Original", "brain_invert": 0, "pathology_invert": 0, "pathology_switch": 0, "pathology_mixup": 0, "brain_mixup": 0, "shift_scale": 0, "Contrast": 0, 'prob_brain_scale_shift': False, 'prob_pathology_scale_shift': False},
    {"name": "Block Masks", "brain_invert": 0, "pathology_invert": 0, "pathology_switch": 0, "pathology_mixup": 0, "brain_mixup": 0, "shift_scale": 0, "Contrast": 0, 'prob_brain_scale_shift': False, 'prob_pathology_scale_shift': False},
    {"name": "Inversion", "brain_invert": 1.0, "pathology_invert": 1.0, "pathology_switch": 0, "pathology_mixup": 0, "brain_mixup": 0, "shift_scale": 0, "Contrast": 0, 'prob_brain_scale_shift': False, 'prob_pathology_scale_shift': False},
    {"name": "Modality Switch", "brain_invert": 0, "pathology_invert": 0, "pathology_switch": 1.0, "pathology_mixup": 0, "brain_mixup": 0, "shift_scale": 0, "Contrast": 0, 'prob_brain_scale_shift': False, 'prob_pathology_scale_shift': False},
    {"name": "Mixup", "brain_invert": 0, "pathology_invert": 0, "pathology_switch": 0, "pathology_mixup": 1.0, "brain_mixup": 1.0, "shift_scale": 0, "Contrast": 0, 'prob_brain_scale_shift': False, 'prob_pathology_scale_shift': False},
    {"name": "Contrast Changes", "brain_invert": 0, "pathology_invert": 0, "pathology_switch": 0, "pathology_mixup": 0, "brain_mixup": 0, "shift_scale": 1.0, "Contrast": 1.0, 'prob_brain_scale_shift': True, 'prob_pathology_scale_shift': True},
]

# Process each subject and create individual figures
for subject_idx in subject_indices:
    print(f"\nProcessing subject {subject_idx}...")
    
    # Process the subject
    img_data, seg_data, mask_data, best_slice, tumor_size = process_subject(subject_idx, images, segs, masks)
    
    if img_data is None:
        continue
    
    # Create a 2×3 grid for this subject
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.1, hspace=0.2)
    
    # Get masks
    tumor_mask = seg_data[0, :, best_slice, :].numpy() > 0
    brain_mask = mask_data[0, :, best_slice, :].numpy() > 0
    healthy_brain_mask = brain_mask & (~tumor_mask)
    
    # Get original image slice - ABSOLUTELY NO PROCESSING
    orig_slice = img_data[0, :, best_slice, :].numpy()
    
    # Find display range from original image brain region ONCE - use exact values
    if np.any(brain_mask):
        brain_values = orig_slice[brain_mask]
        vmin = np.min(brain_values)
        vmax = np.max(brain_values)
    else:
        vmin = np.min(orig_slice)
        vmax = np.max(orig_slice)
    
    # First, original image - display exactly as is
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig_slice, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title("Original", fontweight='bold')
    ax.axis('off')
    
    # Second, image with block masks
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(orig_slice, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Create solid block masks
    ax.imshow(np.ma.masked_where(~tumor_mask, np.ones_like(tumor_mask)), 
              cmap='Reds', alpha=0.7, vmin=0, vmax=1, interpolation='nearest')
    ax.imshow(np.ma.masked_where(~healthy_brain_mask, np.ones_like(healthy_brain_mask)), 
              cmap='Blues', alpha=0.4, vmin=0, vmax=1, interpolation='nearest')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Pathology'),
        Patch(facecolor='blue', alpha=0.4, label='Brain Tissue')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)
    
    ax.set_title("Block Masks", fontweight='bold')
    ax.axis('off')
    
    # Process and display each augmentation
    for idx, aug in enumerate(aug_types[2:]):  # Skip original and masks
        # Reset augmentation config to defaults first
        aug_config.prob_brain_invert = 0
        aug_config.prob_pathology_invert = 0
        aug_config.prob_pathology_switch = 0
        aug_config.prob_pathology_mixup = 0
        aug_config.prob_brain_mixup = 0
        aug_config.prob_brain_scale_shift = False
        aug_config.prob_tumor_scale_shift = False
        
        # Configure specific augmentation
        aug_config.prob_brain_invert = aug["brain_invert"]
        aug_config.prob_pathology_invert = aug["pathology_invert"]
        aug_config.prob_pathology_switch = aug["pathology_switch"]
        aug_config.prob_pathology_mixup = aug["pathology_mixup"]
        aug_config.prob_brain_mixup = aug["brain_mixup"]
        aug_config.prob_brain_scale_shift = aug["prob_brain_scale_shift"]
        aug_config.prob_tumor_scale_shift = aug["prob_pathology_scale_shift"]
    
        # Set random seed for reproducible results
        torch.manual_seed(42 + idx)
        random.seed(42 + idx)
        
        # For Contrast Changes, set intensity parameters
        if "Contrast" in aug["name"]:
            if hasattr(aug_config, 'tumor_factor_multiply'):
                aug_config.tumor_factor_multiply = (1.5, 1.5)
            if hasattr(aug_config, 'tumor_factor_intensity'):
                aug_config.tumor_factor_intensity = (-0.5, -0.5)
            if hasattr(aug_config, 'brain_factor_multiply'):
                aug_config.brain_factor_multiply = (1.5, 1.5)
            if hasattr(aug_config, 'brain_factor_intensity'):
                aug_config.brain_factor_intensity = (0.5, 0.5)
        
        # Apply augmentation
        augmented = spatial_contrast_aug(
            aug_config=aug_config,
            channel_add=[1],
            pathology_label=seg_data,
            brain_mask=mask_data,
            batch_img=img_data,
            plot_image=False
        )
        
        # Calculate position in grid
        row = (idx + 2) // 3
        col = (idx + 2) % 3
        
        ax = fig.add_subplot(gs[row, col])
        
        # Get augmented slice - display exactly as is, no processing whatsoever
        img_slice = augmented[0, :, best_slice, :].numpy()
        
        # Display with same scaling and no interpolation
        ax.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Clean up axes
        ax.axis('off')
        
        # Add title
        ax.set_title(aug["name"], fontweight='bold')
    
    # Add subfigure labels
    for i, ax in enumerate(fig.axes):
        ax.text(-0.05, 1.05, chr(97+i)+')', transform=ax.transAxes, 
                size=11, weight='bold')
    
    # Add figure title
    plt.suptitle(f"MRI Augmentation Techniques - Subject {subject_idx} (Slice {best_slice})", 
                 fontsize=14, y=0.98)
    
    # Save in high quality with subject index in filename
    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    png_filename = os.path.join(dataset_dir, f'augmentation_subject_{subject_idx}_slice_{best_slice}.png')
    pdf_filename = os.path.join(dataset_dir, f'augmentation_subject_{subject_idx}_slice_{best_slice}.pdf')
    
    plt.savefig(png_filename, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    
    print(f"Saved: {png_filename}")
    print(f"Saved: {pdf_filename}")
    
    plt.close()

print(f"\nAll augmentation figures saved in '{output_dir}' folder!") 