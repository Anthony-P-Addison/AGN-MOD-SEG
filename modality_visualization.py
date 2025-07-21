import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from pathlib import Path

def normalize_img(img):
    """Normalize image to [0,1] range"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def load_modalities(file_path):
    """Load all modalities from a NIfTI file"""
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data

def plot_modality_grid(data, modalities, slice_idx=None, save_path=None):
    """
    Create a grid visualization of different modalities with tumor labels
    """
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    
    print(f"Data shape: {data.shape}")
    print(f"Number of modalities: {len(modalities)}")
    print(f"Slice index: {slice_idx}")
    
    n_modalities = len(modalities)
    
    # Create a figure with 2 rows
    fig, axes = plt.subplots(2, n_modalities, figsize=(5*n_modalities, 10))
    
    # First row: modalities
    for i, modality in enumerate(modalities):
        try:
            img_data = data[..., slice_idx, i]
            print(f"Modality {modality} shape: {img_data.shape}")
            axes[0, i].imshow(normalize_img(img_data), cmap='gray')
            axes[0, i].set_title(modality)
            axes[0, i].axis('off')
        except Exception as e:
            print(f"Error plotting modality {modality}: {e}")
    
    # Second row: tumor labels
    try:
        tumor_data = data[..., slice_idx, -1]  # Get tumor data from last channel
        print(f"Tumor data shape: {tumor_data.shape}")
        for i in range(n_modalities):
            axes[1, i].imshow(tumor_data, cmap='Reds')
            axes[1, i].set_title('Tumor')
            axes[1, i].axis('off')
    except Exception as e:
        print(f"Error plotting tumor data: {e}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_intensity_distributions(data, modalities, save_path=None):
    """
    Plot intensity distributions for different modalities
    
    Args:
        data: 4D numpy array (x, y, z, channels)
        modalities: List of modality names
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(modalities)))
    
    for i, (modality, color) in enumerate(zip(modalities, colors)):
        # Get all non-zero values for this modality
        values = data[..., i][data[..., i] > 0]
        plt.hist(values, bins=100, alpha=0.5, label=modality, color=color)
    
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Modality Intensity Distributions')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_interactive_blend(data, modalities, slice_idx=None, save_path=None):
    """
    Create an interactive visualization that allows blending between modalities
    
    Args:
        data: 4D numpy array (x, y, z, channels)
        modalities: List of modality names
        slice_idx: Slice index to visualize (default: middle slice)
        save_path: Path to save the visualization
    """
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Normalize all modalities
    normalized_data = np.array([normalize_img(data[..., i, slice_idx]) for i in range(len(modalities))])
    
    # Initial blend (equal weights)
    weights = np.ones(len(modalities)) / len(modalities)
    blended = np.sum(normalized_data * weights[:, None, None], axis=0)
    
    # Create initial image
    img = ax.imshow(blended, cmap='gray')
    ax.set_title('Interactive Modality Blend')
    ax.axis('off')
    
    # Create sliders for each modality
    sliders = []
    slider_axes = []
    for i, modality in enumerate(modalities):
        slider_ax = plt.axes([0.2, 0.1 - 0.05*i, 0.6, 0.03])
        slider = Slider(slider_ax, modality, 0, 1, valinit=1/len(modalities))
        sliders.append(slider)
        slider_axes.append(slider_ax)
    
    def update(val):
        # Get current weights from sliders
        weights = np.array([s.val for s in sliders])
        weights = weights / weights.sum()  # Normalize weights
        
        # Update blended image
        blended = np.sum(normalized_data * weights[:, None, None], axis=0)
        img.set_data(blended)
        fig.canvas.draw_idle()
    
    # Register update function with all sliders
    for slider in sliders:
        slider.on_changed(update)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def main():
    # Example usage
    data_dir = "data/BRATS/Images"
    sample_file = "BRATS_002_normed_on_mask.nii.gz"
    file_path = os.path.join(data_dir, sample_file)
    
    # Define modalities
    modalities = ["T1", "T1c", "T2", "FLAIR"]
    
    # Load data and print shape
    data = load_modalities(file_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Create output directory
    os.makedirs("visualization_results", exist_ok=True)
    
    # Create different visualizations
    plot_modality_grid(data, modalities, save_path="visualization_results/modality_grid.png")
    plot_intensity_distributions(data, modalities, save_path="visualization_results/intensity_distributions.png")
    create_interactive_blend(data, modalities, save_path="visualization_results/interactive_blend.png")

if __name__ == "__main__":
    main() 