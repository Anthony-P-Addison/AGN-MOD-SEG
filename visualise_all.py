import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from pathlib import Path
import random

# Configuration (Adjust these paths and names)
BASE_DIR = "data"  # Base directory containing dataset folders
DATASETS_TO_PLOT = ["MSSEG", "ISLES", "ISLES2022", "ATLAS", "WMH", "TBI", "BRATS", "TUMOUR2",] # Cleaned list
IMG_SUBDIR = "Images"
SEG_SUBDIR = "Labels"  # Subdirectory containing segmentation masks
IMG_PATTERN = "*.nii.gz"
SEG_SUFFIX = "_label.nii.gz" # Assumed suffix or pattern for finding matching segs
SAVE_DIR = "visualization_results"
OUTPUT_FILENAME = "dataset_modality_comparison_sample.png" # Changed name slightly
SLICE_IDX_TO_PLOT = None # Set to an integer to force a specific slice, None for middle slice
MASK_COLORMAP = 'Reds' # Colormap for the tumor mask overlay
MASK_ALPHA = 0.5       # Transparency of the mask overlay (0=transparent, 1=opaque)

# --- Database Modality Configuration ---
# **CRITICAL**: Ensure this matches the channel order *in the NIfTI files* for each dataset.
DATABASE_MODALITIES = {
    "MSSEG":  ["FLAIR", "T1", "T1c", "T2", "PD"],       
    "ISLES":["FLAIR", "T1", "T2","DWI"] ,      
    "ISLES2022": ['ADC','DWI','FLAIR'] ,                   
    "WMH": ["FLAIR", "T1"],         
    "TBI": ["FLAIR", "T1", "T2", "SWI"], 
    "BRATS": ["FLAIR", "T1", "T1c", "T2"],
    "ATLAS": ["T1"],
    "TUMOUR2": ["T1"],}


# --- Define Fixed ROW Order (Modalities) ---
# List all modalities you want rows for, IN THE ORDER YOU WANT THEM.
DESIRED_ROW_ORDER = ["T1", "T1c", "T2", "FLAIR", "DWI", "ADC", "SWI", "PD"] # Renamed for clarity

def load_image_and_seg(img_path, seg_path):
    """Loads NIfTI image and segmentation, handling potential errors."""
    img_data, seg_data, num_slices = None, None, 0 # Initialize all

    # --- Load Image ---
    try:
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata().astype(np.float32) # Use float32 for consistency

        # Determine num_slices based on initial shape
        initial_ndim = img_data.ndim
        if initial_ndim >= 3:
            # Try guessing Z dim based on standard NIfTI (X, Y, Z, [C])
            z_dim_index = 2 if initial_ndim == 3 else 2 # Assume Z is 3rd dim
            num_slices = img_data.shape[z_dim_index]
        else:
             num_slices = 0 # Cannot determine slices for 1D/2D

        # Handle 3D vs 4D images and Transpose if needed
        if initial_ndim == 4:
            # Assume channels are last dimension (X, Y, Z, C), transpose to (C, X, Y, Z)
            img_data = np.transpose(img_data, (3, 0, 1, 2))
            # Update num_slices based on transposed shape (Z is now dim 3)
            num_slices = img_data.shape[3]
            print(f"  Transposed 4D image {Path(img_path).name} to shape: {img_data.shape}") # Add for debugging
        elif initial_ndim == 3:
            img_data = img_data[np.newaxis, ...] # Add channel dim -> (1, X, Y, Z)
            # Update num_slices based on transposed shape (Z is now dim 3)
            num_slices = img_data.shape[3]
        else:
            print(f"Warning: Unsupported image dimensions {initial_ndim} for {img_path}")
            img_data = None # Set img_data to None if unsupported
            num_slices = 0

    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        img_data = None # Ensure img_data is None on error
        num_slices = 0 # Reset num_slices on error

    # <<< --- CORRECTED INDENTATION FOR SEGMENTATION LOADING --- >>>
    # --- Load Segmentation ---
    seg_data = None # Ensure seg_data starts as None
    # **CRITICAL FIX**: Check if seg_path is a non-empty string AND exists
    if isinstance(seg_path, str) and seg_path and os.path.exists(seg_path):
        try:
            print(f"  Attempting to load seg: '{seg_path}'") # Keep for debug
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata().astype(np.uint8)

            # --- Shape Handling for Seg ---
            # Assuming seg should be 3D (X, Y, Z)
            if seg_data.ndim == 4 and seg_data.shape[-1] == 1: # Handle (X,Y,Z,1)
                seg_data = seg_data.squeeze(-1)
            elif seg_data.ndim == 4 and seg_data.shape[0] == 1: # Handle (1,X,Y,Z)
                 seg_data = seg_data.squeeze(0)

            # Check if it's now 3D
            if seg_data.ndim != 3:
                 print(f"Warning: Loaded seg has non-3D shape {seg_data.shape} for {seg_path}")
                 seg_data = None # Invalidate seg if dimensions wrong after squeeze

        except Exception as e:
            print(f"Error loading segmentation '{seg_path}': {repr(e)}")
            seg_data = None # Ensure seg_data is None on error

    # --- Basic shape check ---
    # Compare spatial dims (X, Y, Z) only if both loaded successfully AND seg_data is 3D
    if img_data is not None and seg_data is not None and seg_data.ndim == 3:
        # Assuming img_data is (C, X, Y, Z) after potential transpose
        img_spatial_shape = img_data.shape[1:4]
        seg_spatial_shape = seg_data.shape # Should be (X, Y, Z)
        if img_spatial_shape != seg_spatial_shape:
            print(f"Warning: Img spatial shape {img_spatial_shape} & Seg shape {seg_spatial_shape} mismatch for {img_path}/{seg_path}")
            seg_data = None # Invalidate seg if shapes mismatch

    # Ensure num_slices is 0 if image loading failed
    if img_data is None:
        num_slices = 0

    return img_data, seg_data, num_slices


def find_matching_seg(img_path, seg_dir, seg_suffix):
    """Tries to find a corresponding segmentation file based on image filename."""
    img_stem = Path(img_path).stem.split('.')[0] # Handle .nii.gz etc.
    
    # Attempt 1: Simple suffix replacement (common pattern)
    potential_seg_name = f"{img_stem}{seg_suffix}"
    potential_seg_path = os.path.join(seg_dir, potential_seg_name)
    if os.path.exists(potential_seg_path):
        return potential_seg_path

    # Attempt 2: Look for any seg file starting with the image stem
    seg_files = glob(os.path.join(seg_dir, f"{img_stem}*.nii*"))
    if seg_files:
        return seg_files[0] # Return the first match

    print(f"Warning: Could not find matching segmentation for {img_path} in {seg_dir}")
    return None


def plot_modality_comparison(datasets_data, dataset_order, modality_order, slice_idx_to_plot, save_path):
    """
    Plots a comparison grid with modalities as rows and datasets as columns.
    """
    num_rows = len(modality_order)
    num_cols = len(dataset_order)

    if num_rows == 0 or num_cols == 0:
        print("No modalities or datasets specified for plotting.")
        return

    # Create figure with much more space for labels
    fig = plt.figure(figsize=(5 * num_cols + 6, 5 * num_rows + 1))  # Even more extra width
    
    # Create GridSpec with extra space on left for labels
    gs = plt.GridSpec(num_rows, num_cols + 1, figure=fig,  # Added extra column for labels
                     width_ratios=[0.3] + [1]*num_cols,  # First column (labels) is narrower
                     left=0.01, right=0.99,  # Use full width
                     top=0.95, bottom=0.05, 
                     wspace=0.1, hspace=0.2)
    
    # Create axes array including label column
    axes = []
    for i in range(num_rows):
        row = []
        # Add label axis
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.axis('off')
        row.append(label_ax)
        # Add plot axes
        for j in range(num_cols):
            row.append(fig.add_subplot(gs[i, j + 1]))
        axes.append(row)
    axes = np.array(axes)
    
    fig.suptitle("Modality vs. Dataset Comparison with Mask Overlay", fontsize=35, y=0.98)

    # --- Set Column Headers (Dataset Names) ---
    for col_idx, dataset_name in enumerate(dataset_order):
        axes[0, col_idx + 1].set_title(dataset_name, fontsize=25, pad=25)  # +1 for label column

    # --- Iterate through Rows (Modalities) and Columns (Datasets) ---
    for row_idx, target_modality in enumerate(modality_order):
        # Add modality label as text in the label column
        label_ax = axes[row_idx, 0]
        label_ax.text(0.95, 0.5, target_modality,
                     fontsize=30,
                     ha='right',
                     va='center',
                     transform=label_ax.transAxes)

        for col_idx, dataset_name in enumerate(dataset_order):
            ax = axes[row_idx, col_idx + 1]  # +1 for label column
            ax.axis('off')

            # Check if we have data for this dataset
            if dataset_name not in datasets_data:
                ax.text(0.5, 0.5, "Dataset\nNot Processed", ha='center', va='center', fontsize=14, color='red')
                continue

            data = datasets_data[dataset_name]
            img_data = data.get('img_data')
            seg_data = data.get('seg_data')
            num_slices = data.get('num_slices', 0)
            dataset_modalities = data.get('modalities', [])
            modality_to_channel_idx = {mod: i for i, mod in enumerate(dataset_modalities)}

            # Handle case where image loading failed for this dataset
            if img_data is None:
                if row_idx == 0:
                    ax.text(0.5, 0.5, f"Load Failed", ha='center', va='center', fontsize=14)
                continue

            # Check if the current target modality exists for this dataset
            if target_modality in modality_to_channel_idx:
                channel_idx = modality_to_channel_idx[target_modality]

                # Validate channel index against actual loaded data shape
                if channel_idx < img_data.shape[0]:
                    # Determine the slice index
                    current_slice_idx = slice_idx_to_plot
                    if current_slice_idx is None:
                        current_slice_idx = num_slices // 2
                    elif current_slice_idx >= num_slices:
                        print(f"Warning: Req slice {slice_idx_to_plot} invalid for {dataset_name}/{target_modality} (max {num_slices-1}). Using its middle {num_slices // 2}.")
                        current_slice_idx = num_slices // 2

                    # --- Plot Image and Mask ---
                    img_slice = img_data[channel_idx, :, :, current_slice_idx]
                    ax.imshow(img_slice, cmap='gray', interpolation='nearest')

                    if seg_data is not None:
                        if seg_data.ndim == 3:
                            seg_slice = seg_data[:, :, current_slice_idx]
                            masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                            ax.imshow(masked_seg, cmap=MASK_COLORMAP, alpha=MASK_ALPHA, interpolation='nearest')
                        else:
                            print(f"Warning: Seg data for {dataset_name} has wrong dims: {seg_data.shape}")

                else:
                    ax.text(0.5, 0.5, f"Channel Idx\n{channel_idx} Invalid\n(Max {img_data.shape[0]-1})", ha='center', va='center', fontsize=14)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=14, color='grey')

    # Save with tight bbox to ensure all labels are included
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.5)
    print(f"Saved comparison plot to: {save_path}")
    plt.close()


def main():
    """Main function to select samples and generate the plot."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    datasets_data = {}

    # Use the DESIRED_ROW_ORDER for rows
    all_modalities_list = DESIRED_ROW_ORDER # This defines the rows
    # The datasets define the columns (passed directly)
    print(f"Plotting rows for modalities: {all_modalities_list}")
    print(f"Plotting columns for datasets: {DATASETS_TO_PLOT}")


    print("\nSelecting sample images and loading data...")
    for dataset_name in DATASETS_TO_PLOT:
        # Check if dataset config exists before processing
        if dataset_name not in DATABASE_MODALITIES:
            print(f"\nWarning: Modality configuration missing for dataset '{dataset_name}'. Skipping.")
            continue # Skip datasets not in the config

        print(f"\nProcessing Dataset: {dataset_name}")
        img_dir = os.path.join(BASE_DIR, dataset_name, IMG_SUBDIR)
        seg_dir = os.path.join(BASE_DIR, dataset_name, SEG_SUBDIR)

        if not os.path.isdir(img_dir):
            print(f"Warning: Image directory not found: {img_dir}. Skipping.")
            # Store placeholder so plot function knows it was attempted
            datasets_data[dataset_name] = {'img_data': None, 'seg_data': None, 'num_slices': 0, 'modalities': []}
            continue

        # Check for seg dir existence here, handle it gracefully later
        seg_dir_exists = os.path.isdir(seg_dir)
        if not seg_dir_exists:
            print(f"Warning: Segmentation directory not found: {seg_dir}. Proceeding without segmentation.")

        image_files = sorted(glob(os.path.join(img_dir, IMG_PATTERN)))
        if not image_files:
            print(f"Warning: No images found matching '{IMG_PATTERN}' in {img_dir}. Skipping.")
            datasets_data[dataset_name] = {'img_data': None, 'seg_data': None, 'num_slices': 0, 'modalities': []}
            continue

        # --- Select a random sample image ---
        sample_img_path = random.choice(image_files)
        print(f"  Selected sample image: {Path(sample_img_path).name}")

        # --- Find the corresponding segmentation ---
        sample_seg_path = None
        if seg_dir_exists:
            sample_seg_path = find_matching_seg(sample_img_path, seg_dir, SEG_SUFFIX)
            # Warning about not finding it is printed inside find_matching_seg

        # --- Load image and potentially segmentation ---
        img_data, seg_data, num_slices = load_image_and_seg(sample_img_path, sample_seg_path) # sample_seg_path is None if not found/dir missing

        # --- Store data AND the modality list for this dataset ---
        datasets_data[dataset_name] = {
            'img_data': img_data,
            'seg_data': seg_data,
            'img_path': sample_img_path,
            'num_slices': num_slices, # Already handled if img_data is None
            'modalities': DATABASE_MODALITIES.get(dataset_name, []) # *** CRITICAL FIX ***
        }

    print("\nGenerating comparison plot...")
    save_path = os.path.join(SAVE_DIR, OUTPUT_FILENAME)
    # Ensure the correct arguments are passed
    plot_modality_comparison(
        datasets_data,         # 1
        DATASETS_TO_PLOT,      # 2 - List of datasets for columns
        DESIRED_ROW_ORDER,     # 3 - List of modalities for rows
        SLICE_IDX_TO_PLOT,     # 4
        save_path              # 5
    )

if __name__ == "__main__":
    main()