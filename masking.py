

# Example preprocessing script (simplified)
import nibabel as nib
import numpy as np
from glob import glob
import os
from pathlib import Path
from scipy import ndimage
from typing import Optional




def get_background_value(image_channel_3d, corner_size=5):
    """
    Calculate background value from the corners of a 3D numpy array.
    Assumes background is likely the median value in the corners.
    """
    if image_channel_3d.ndim != 3:
        raise ValueError(f"Input must be a 3D array, but got shape {image_channel_3d.shape}")

    corners = [
        image_channel_3d[:corner_size, :corner_size, :corner_size],
        image_channel_3d[:corner_size, :corner_size, -corner_size:],
        image_channel_3d[:corner_size, -corner_size:, :corner_size],
        image_channel_3d[:corner_size, -corner_size:, -corner_size:],
        image_channel_3d[-corner_size:, :corner_size, :corner_size],
        image_channel_3d[-corner_size:, :corner_size, -corner_size:],
        image_channel_3d[-corner_size:, -corner_size:, :corner_size],
        image_channel_3d[-corner_size:, -corner_size:, -corner_size:],
    ]
    # Filter out corners that might be empty if corner_size > dimension size
    valid_corners = [c.flatten() for c in corners if c.size > 0]
    if not valid_corners:
        print("Warning: Could not sample corners (image might be too small). Using global minimum as background estimate.")
        return np.min(image_channel_3d)

    background_samples = np.concatenate(valid_corners)
    if background_samples.size == 0:
         print("Warning: No valid background samples found in corners. Using global minimum.")
         return np.min(image_channel_3d)

    return np.median(background_samples)

def create_and_save_mask(img_file_path, output_mask_path, threshold_factor, corner_size):
    """
    Loads a NIfTI image, creates a mask based on the first modality,
    and saves the mask.
    """
    try:
        img_obj = nib.load(img_file_path)
        img_data = img_obj.get_fdata()
        img_affine = img_obj.affine
        img_header = img_obj.header

        # --- Select the first modality ---
        if img_data.ndim == 4:
            img_data = np.transpose(img_data, (3, 0, 1, 2))
            first_modality_3d = img_data[0]
            print(f"  Image is 4D (shape {img_data.shape}), using first channel.")
        elif img_data.ndim == 3:
            first_modality_3d = img_data
            print(f"  Image is 3D (shape {img_data.shape}), using as is.")
        else:
            print(f"  Skipping {img_file_path}: Unsupported image dimension {img_data.ndim}.")
            return False
        # ---

        # Calculate background value from the first modality
        background_val = get_background_value(first_modality_3d, corner_size)

        # Calculate threshold
        threshold = background_val * threshold_factor

        # Create boolean mask
        mask_data_bool = first_modality_3d > threshold

        # Label connected components in the thresholded mask (foreground)
        labeled_foreground, num_components = ndimage.label(mask_data_bool)
        
        final_mask_bool = np.zeros_like(mask_data_bool) # Initialize empty mask
        
        if num_components > 0:
            # Find the size of each foreground component (component 0 is background)
            component_sizes = np.bincount(labeled_foreground.ravel())
            # Find the label of the largest component (ignoring background label 0)
            # Add check for component_sizes length in case only background exists (label 0)
            if len(component_sizes) > 1:
                largest_component_label = component_sizes[1:].argmax() + 1
                
                # Create a mask containing only the largest component
                largest_component_mask = (labeled_foreground == largest_component_label)
                
                # Fill all holes within the largest component using SciPy's function
                final_mask_bool = ndimage.binary_fill_holes(largest_component_mask)
            else:
                # Only background was found after thresholding
                print("  Warning: No foreground components found after thresholding.")
                # final_mask_bool remains all zeros
        else:
            print("  Warning: No components found by ndimage.label (check threshold?).")
            # final_mask_bool remains all zeros


        # Convert final boolean mask to uint8 for saving
        mask_data_uint8 = final_mask_bool.astype(np.uint8)

        # Create a new NIfTI image for the mask, preserving spatial info
        mask_obj = nib.Nifti1Image(mask_data_uint8, img_affine, img_header)

        # Ensure the output directory exists
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the mask
        nib.save(mask_obj, output_mask_path)
        print(f"  Saved mask to: {output_mask_path}")
        return True

    except Exception as e:
        print(f"  Error processing {img_file_path}: {e}")
        return False

def main():
    # --- Configuration Section (EDIT THESE VALUES) ---

    input_base_dir = "data"  # Base directory containing dataset subfolders
    output_base_dir = "data" # Base directory where mask subfolders will be created
    datasets = ["TUMOUR2"]     # List of dataset subfolder names to process
    threshold_factor = 1                          # Factor to multiply median corner value by
    corner_size = 5                                   # Size of the cube edge to sample from corners
    img_pattern = "*.nii.gz"                          # Glob pattern for image files (e.g., '*.nii.gz', '*.nii*')
    # --- End Configuration Section ---

    print(f"Starting mask generation...")
    print(f"Input Base Directory: {input_base_dir}")
    print(f"Output Base Directory: {output_base_dir}")
    print(f"Datasets: {datasets}")
    print(f"Threshold Factor: {threshold_factor}")
    print(f"Corner Size: {corner_size}")
    print("-" * 30)

    total_processed = 0
    total_succeeded = 0

    for dataset_name in datasets:
        input_dataset_dir = os.path.join(input_base_dir, dataset_name,'Images')
        output_dataset_dir = os.path.join(output_base_dir, dataset_name, 'Masks')

        print(f"\nProcessing Dataset: {dataset_name}")
        print(f"Input folder: {input_dataset_dir}")
        print(f"Output folder: {output_dataset_dir}")

        if not os.path.isdir(input_dataset_dir):
            print(f"Warning: Input directory not found: {input_dataset_dir}. Skipping.")
            continue

        # Find image files
        image_files = sorted(glob(os.path.join(input_dataset_dir, img_pattern)))

        if not image_files:
            print(f"Warning: No images found matching pattern '{img_pattern}' in {input_dataset_dir}. Skipping.")
            continue

        print(f"Found {len(image_files)} images.")

        dataset_processed = 0
        dataset_succeeded = 0
        for img_path in image_files:
            print(f"Processing image: {Path(img_path).name}")
            dataset_processed += 1

            # Construct output filename
            img_filename_stem = Path(img_path).stem.split('.')[0] # Get name before first dot (handles .nii and .nii.gz)
            mask_filename = f"{img_filename_stem}_mask.nii.gz" # Always save as .nii.gz
            output_path = os.path.join(output_dataset_dir, mask_filename)

            if create_and_save_mask(img_path, output_path, threshold_factor, corner_size):
                dataset_succeeded += 1

        print(f"Finished dataset {dataset_name}: {dataset_succeeded}/{dataset_processed} masks generated successfully.")
        total_processed += dataset_processed
        total_succeeded += dataset_succeeded

    print("-" * 30)
    print(f"Overall Summary: {total_succeeded}/{total_processed} masks generated successfully across all datasets.")
    print("Mask generation complete.")

if __name__ == "__main__":
    main()
