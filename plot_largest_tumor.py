import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_modalities(file_path):
    """Load all modalities from a NIfTI file"""
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data


def find_largest_tumor_slice(data):
    """Find the slice index with the largest tumor area"""
    # Assuming the tumor is labeled in the last channel
    tumor_channel = data[..., -1]
    slice_areas = [np.sum(slice > 0) for slice in tumor_channel]
    largest_slice_idx = np.argmax(slice_areas)
    
    # Debugging output
    print("Slice areas:", slice_areas)
    print("Largest slice index:", largest_slice_idx)
    
    return largest_slice_idx


def plot_slice_with_contrast(data, slice_idx, modality_idx=0, figsize=(8, 8), rotate=True, crop_margin=20, save_path=None):
    """Plot a slice with optional rotation and original contrast for a specific modality, cropping a fixed margin, and save if a path is provided"""
    fig, ax = plt.subplots(figsize=figsize)

    if rotate:
        # Rotate the image 90 degrees counter-clockwise
        img_data = np.rot90(data[..., slice_idx, modality_idx], k=1)
    else:
        # Use the image as it comes
        img_data = data[..., slice_idx, modality_idx]

    # Ensure crop margin does not exceed image dimensions
    height, width = img_data.shape
    crop_margin = min(crop_margin, height // 2, width // 2)

    # Crop a specific margin from each side
    cropped_img = img_data[crop_margin:height-crop_margin, crop_margin:width-crop_margin]

    # Plot the cropped image with original contrast
    img = ax.imshow(cropped_img, cmap='gray', aspect='auto')
    ax.set_title(f'Modality {modality_idx} - Slice {slice_idx}')
    ax.axis('off')

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def main(slice_idx=None, modality_idx=0):
    # Example usage
    file_path = "data/ISLES2022/Images/label_patient_0002.nii.gz"
    data = load_modalities(file_path)
    
    # Use the provided slice index or find the largest tumor slice
    if slice_idx is None:
        slice_idx = find_largest_tumor_slice(data)
    
    # Specify image dimensions, rotation, and modality
    plot_slice_with_contrast(data, slice_idx, modality_idx=modality_idx, figsize=(10, 10), rotate=True)


if __name__ == "__main__":
    # Call main with specific slice and modality indices
    main(slice_idx=None, modality_idx=1) 