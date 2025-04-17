import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def isolate_brain_from_normalized(input_path: str, plot_histogram: bool = True):
    """
    Isolate brain from background in an already z-score normalized NIfTI file.
    
    Args:
        input_path (str): Path to normalized NIfTI file
        plot_histogram (bool): Whether to plot intensity histogram to help identify threshold
    """
    # Load normalized NIfTI file
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    
    # Plot histogram to visualize intensity distribution
    if plot_histogram:
        plt.figure(figsize=(10, 5))
        plt.hist(data.flatten(), bins=100)
        plt.title('Intensity Histogram')
        plt.xlabel('Intensity (z-score)')
        plt.ylabel('Frequency')
        plt.savefig('histogram.png')
        plt.show()

    # Find the valley between background and brain peaks
    # You might need to adjust these parameters based on your data
    hist, bin_edges = np.histogram(data.flatten(), bins=100)
    valley_idx = np.argmin(hist[1:20]) + 1  # Look for valley in first few bins
    threshold = bin_edges[valley_idx]
    
    print(f"Detected threshold: {threshold:.4f}")
    
    # Create brain mask
    brain_mask = data > threshold
    
    # Apply mask to data
    masked_data = np.copy(data)
    masked_data[~brain_mask] = 0
    
    # Visualize middle slices
    z_mid = data.shape[2] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(data[:, :, z_mid], cmap='gray')
    plt.title('Original Normalized Data')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(brain_mask[:, :, z_mid], cmap='gray')
    plt.title('Brain Mask')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(masked_data[:, :, z_mid], cmap='gray')
    plt.title('Masked Data')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('brain_slices.png')
    plt.show()
    
    # Print some statistics
    print("\nData statistics:")
    print(f"Original data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Brain voxels: {np.sum(brain_mask)}")
    print(f"Brain volume percentage: {100 * np.sum(brain_mask) / brain_mask.size:.2f}%")

    return masked_data, brain_mask, threshold

if __name__ == "__main__":
    # Example usage
    input_file = "data/BRATS/Images/BRATS_001_normed_on_mask.nii.gz"
    
    masked_data, brain_mask, threshold = isolate_brain_from_normalized(input_file)
    
    # Optionally save the results
    output_file = "cool/brain_only.nii.gz"
    mask_file = "cool/brain_mask.nii.gz"
    
    # Load original image to get affine transformation
    original_img = nib.load(input_file)
    
    # Save masked data
    masked_img = nib.Nifti1Image(masked_data, original_img.affine)
    nib.save(masked_img, output_file)
    
    # Save brain mask
    mask_img = nib.Nifti1Image(brain_mask.astype(np.float32), original_img.affine)
    nib.save(mask_img, mask_file) 