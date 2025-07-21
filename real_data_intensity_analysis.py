import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import random
import os
import glob
from pathlib import Path
from config import Database_config, Augmentation_config, Training_config
from augment_utils import spatial_contrast_aug
from collections import defaultdict
from dataloader import create_dataloader

def load_training_data_by_modality(datasets):
    """
    Load training data directly from files using the database config structure
    """
    print("🔍 Loading training data from specified datasets...")
    
    db_config = Database_config()
    modality_data = defaultdict(list)
    dataset_counts = {}
    
    for dataset in datasets:
        if dataset not in db_config.channels:
            print(f"⚠️  Dataset {dataset} not found in config, skipping...")
            continue
            
        print(f"\n📂 Processing {dataset} training data:")
        modalities = db_config.channels[dataset]
        print(f"   Modalities: {modalities}")
        
        try:
            # Get training data paths directly from config
            img_path = Path(db_config.img_path[dataset])
            
            if not img_path.exists():
                print(f"   ⚠️  Image path {img_path} does not exist, skipping {dataset}")
                continue
            
            # Get all NIfTI files in the directory (training data)
            image_files = []
            for ext in ['*.nii.gz', '*.nii']:
                image_files.extend(glob.glob(str(img_path / ext)))
                image_files.extend(glob.glob(str(img_path / "**" / ext), recursive=True))
            
            print(f"   Found {len(image_files)} image files")
            
            # Process first 100 files as training data (for speed)
            dataset_image_count = 0
            max_files = min(100, len(image_files))
            
            for file_path in image_files[:max_files]:
                try:
                    # Load NIfTI file
                    img = nib.load(file_path)
                    img_data = img.get_fdata()
                    
                    # Handle different data formats
                    if len(img_data.shape) == 4:
                        # Multi-channel data - each channel is a modality
                        n_channels = img_data.shape[-1]
                        
                        if n_channels != len(modalities):
                            # Take only the number of channels we expect
                            n_channels = min(n_channels, len(modalities))
                        
                        for i, modality in enumerate(modalities[:n_channels]):
                            channel_data = img_data[:, :, :, i]
                            
                            # Take middle slice
                            slice_data = channel_data[:, :, channel_data.shape[2] // 2]
                            
                            # Create brain mask (non-zero regions)
                            brain_mask = slice_data != 0
                            
                            if np.any(brain_mask):
                                # Normalize brain region to [0, 1]
                                brain_intensities = slice_data[brain_mask]
                                if brain_intensities.max() > brain_intensities.min():
                                    normalized_brain = (brain_intensities - brain_intensities.min()) / (brain_intensities.max() - brain_intensities.min())
                                    
                                    # Create normalized image
                                    normalized_slice = np.zeros_like(slice_data)
                                    normalized_slice[brain_mask] = normalized_brain
                                    modality_data[modality].append(normalized_slice)
                                    
                    elif len(img_data.shape) == 3:
                        # Single channel data - assume it's the first modality
                        if len(modalities) > 0:
                            modality = modalities[0]
                            
                            # Take middle slice
                            slice_data = img_data[:, :, img_data.shape[2] // 2]
                            
                            # Create brain mask and normalize
                            brain_mask = slice_data != 0
                            if np.any(brain_mask):
                                brain_intensities = slice_data[brain_mask]
                                if brain_intensities.max() > brain_intensities.min():
                                    normalized_brain = (brain_intensities - brain_intensities.min()) / (brain_intensities.max() - brain_intensities.min())
                                    normalized_slice = np.zeros_like(slice_data)
                                    normalized_slice[brain_mask] = normalized_brain
                                    modality_data[modality].append(normalized_slice)
                    
                    dataset_image_count += 1
                    
                except Exception as e:
                    print(f"      ❌ Error loading {Path(file_path).name}: {e}")
                    continue
            
            dataset_counts[dataset] = dataset_image_count
            print(f"   ✅ Processed {dataset_image_count} training images")
            
        except Exception as e:
            print(f"   ❌ Error processing {dataset}: {e}")
            continue
    
    # Convert to regular dict and print summary
    modality_data = dict(modality_data)
    print(f"\n📊 Training data loading summary:")
    for modality, data_list in modality_data.items():
        print(f"   {modality}: {len(data_list)} images")
    
    print(f"\n📈 Dataset image counts:")
    for dataset, count in dataset_counts.items():
        print(f"   {dataset}: {count} images")
    
    return modality_data, dataset_counts

def calculate_modality_distributions(modality_data):
    """
    Calculate average intensity distribution for each modality across training data
    """
    print("\n📈 Calculating average distributions per modality from training data...")
    
    distributions = {}
    
    for modality, images in modality_data.items():
        print(f"   Processing {modality}...")
        
        # Flatten all images and extract brain pixels only (non-zero)
        all_brain_intensities = []
        
        for img in images:
            brain_mask = img > 0  # Non-zero pixels are brain
            if np.any(brain_mask):
                brain_intensities = img[brain_mask]
                all_brain_intensities.extend(brain_intensities)
        
        if len(all_brain_intensities) == 0:
            print(f"   ⚠️  No brain pixels found for {modality}")
            continue
            
        all_brain_intensities = np.array(all_brain_intensities)
        
        # Calculate histogram (probability density)
        hist, bin_edges = np.histogram(all_brain_intensities, bins=100, density=True, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        distributions[modality] = {
            'bin_centers': bin_centers,
            'hist': hist,
            'raw_intensities': all_brain_intensities
        }
        
        print(f"     Training images used: {len(images)}")
        print(f"     Mean: {np.mean(all_brain_intensities):.3f} ± {np.std(all_brain_intensities):.3f}")
        print(f"     Range: [{np.min(all_brain_intensities):.3f}, {np.max(all_brain_intensities):.3f}]")
        print(f"     Total brain pixels: {len(all_brain_intensities):,}")
    
    return distributions

def create_training_augmented_distribution(modality_data, dataset_counts):
    """
    Create distribution from augmented training images, matching the count of training data
    Uses brain masks properly to exclude background values from augmentation results
    """
    total_training_images = sum(dataset_counts.values())
    print(f"\n🔧 Creating augmented distribution from {total_training_images} training images...")
    
    aug_config = Augmentation_config()
    augmented_intensities = []
    
    # Collect all training images from all modalities
    all_images = []
    all_modalities = []
    
    for modality, images in modality_data.items():
        for img in images:
            all_images.append(img)
            all_modalities.append(modality)
    
    # Use all available training images for augmentation
    n_samples = len(all_images)
    print(f"   Applying augmentations to all {n_samples} training images...")
    
    for i in range(n_samples):
        img = all_images[i]
        modality = all_modalities[i]
        
        try:
            # Store original brain mask from the input image
            original_brain_mask = img > 0
            
            # Convert to tensor and add batch/channel dimensions
            img_tensor = torch.from_numpy(img).float()
            
            # Add dimensions: [H, W] -> [1, 1, H, W, 1] (batch, channel, H, W, D)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, H, W, 1]
            
            # Create brain mask tensor (matching the original brain region)
            brain_mask = (img_tensor > 0).float()
            
            # Create random pathology mask (small random regions within brain)
            pathology_mask = torch.zeros_like(brain_mask, dtype=torch.bool)
            if torch.any(brain_mask):
                # Add small random pathology regions ONLY within brain
                h, w = brain_mask.shape[-3], brain_mask.shape[-2]
                brain_regions = brain_mask.squeeze().numpy() > 0
                
                for _ in range(random.randint(0, 3)):  # 0-3 random lesions
                    # Find brain pixels to place lesions
                    brain_coords = np.where(brain_regions)
                    if len(brain_coords[0]) > 0:
                        # Pick random brain pixel as center
                        idx = random.randint(0, len(brain_coords[0]) - 1)
                        center_h, center_w = brain_coords[0][idx], brain_coords[1][idx]
                        radius = random.randint(3, 8)
                        
                        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                        dist = ((y - center_h)**2 + (x - center_w)**2).float()
                        lesion_mask = (dist <= radius**2) & (brain_mask.squeeze() > 0)
                        # Use logical OR instead of bitwise OR
                        pathology_mask[0, 0, :, :, 0] = pathology_mask[0, 0, :, :, 0] | lesion_mask
            
            # Convert pathology mask back to float for the augmentation function
            pathology_mask_float = pathology_mask.float()
            
            # Create multi-channel batch (simulate other modalities)
            batch_img = img_tensor.repeat(4, 1, 1, 1, 1).squeeze(1)  # [4, H, W, D]
            
            # Add noise to other channels to simulate different modalities
            # BUT keep background as zero
            for c in range(1, 4):
                noise = torch.randn_like(batch_img[c]) * 0.1
                batch_img[c] = torch.clamp(batch_img[c] + noise, 0, 1)
                # Ensure background stays zero
                batch_img[c] = batch_img[c] * brain_mask.squeeze()
            
            # Apply augmentation using your actual function
            channel_add = [0]  # Augment first channel
            aug_img = spatial_contrast_aug(
                aug_config=aug_config,
                channel_add=channel_add,
                pathology_label=pathology_mask_float.squeeze(1),  # [1, H, W, D]
                brain_mask=brain_mask.squeeze(1),  # [1, H, W, D]
                batch_img=batch_img,  # [4, H, W, D]
                plot_image=False
            )
            
            # Extract brain intensities from augmented image using ORIGINAL brain mask
            aug_np = aug_img.squeeze().numpy()
            
            # Apply the original brain mask to ensure we only get brain pixels
            if np.any(original_brain_mask):
                # Use the original brain mask to extract only brain pixels
                brain_intensities = aug_np[original_brain_mask]
                
                # Remove any zero or negative values that might have been introduced
                brain_intensities = brain_intensities[brain_intensities > 0]
                
                # Clamp to [0,1] and filter valid intensities
                brain_intensities = np.clip(brain_intensities, 0, 1)
                valid_intensities = brain_intensities[~np.isnan(brain_intensities)]
                
                if len(valid_intensities) > 0:
                    augmented_intensities.extend(valid_intensities)
            
        except Exception as e:
            print(f"   ⚠️  Error augmenting training image {i}: {e}")
            # Fallback: simple contrast/brightness augmentation with proper masking
            try:
                brain_mask_np = img > 0
                if np.any(brain_mask_np):
                    # Apply augmentation only to brain region
                    contrast = random.uniform(0.8, 1.2)
                    brightness = random.uniform(-0.1, 0.1)
                    
                    # Augment the image
                    aug_img = img.copy()
                    aug_img[brain_mask_np] = np.clip(
                        aug_img[brain_mask_np] * contrast + brightness, 0, 1
                    )
                    
                    # Extract only brain pixels (exclude any zeros)
                    brain_intensities = aug_img[brain_mask_np]
                    brain_intensities = brain_intensities[brain_intensities > 0]  # Remove zeros
                    
                    if len(brain_intensities) > 0:
                        augmented_intensities.extend(brain_intensities)
            except:
                continue
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i + 1}/{n_samples} images processed")
    
    if len(augmented_intensities) == 0:
        print("   ❌ No valid augmented intensities generated")
        return None
    
    augmented_intensities = np.array(augmented_intensities)
    
    # Final check: remove any remaining zeros or very small values
    augmented_intensities = augmented_intensities[augmented_intensities > 1e-6]
    
    if len(augmented_intensities) == 0:
        print("   ❌ No valid non-zero augmented intensities generated")
        return None
    
    # Calculate histogram
    hist, bin_edges = np.histogram(augmented_intensities, bins=100, density=True, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    print(f"   ✅ Generated {len(augmented_intensities):,} augmented brain pixels from training data")
    print(f"      Mean: {np.mean(augmented_intensities):.3f} ± {np.std(augmented_intensities):.3f}")
    print(f"      Range: [{np.min(augmented_intensities):.3f}, {np.max(augmented_intensities):.3f}]")
    print(f"      Zero values filtered out: ✅")
    print(f"      Background exclusion: ✅")
    
    return {
        'bin_centers': bin_centers,
        'hist': hist,
        'raw_intensities': augmented_intensities
    }

def plot_training_intensity_distributions(modality_distributions, augmented_distribution, dataset_counts, save_path="training_intensity_distributions.png"):
    """
    Create visualization showing training modality distributions vs augmentation coverage
    """
    print(f"\n📊 Creating intensity distribution visualization...")
    
    # Define colors for modalities
    modality_colors = {
        'T1': '#1f77b4',      # Blue
        'T1c': '#ff7f0e',     # Orange  
        'T2': '#2ca02c',      # Green
        'FLAIR': '#d62728',   # Red
        'PD': '#9467bd',      # Purple
        'DWI': '#8c564b',     # Brown
        'ADC': '#e377c2',     # Pink
        'SWI': '#7f7f7f',     # Gray
    }
    
    # Create single plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual modality distributions (from training data)
    for modality, dist in modality_distributions.items():
        color = modality_colors.get(modality, '#000000')
        plt.plot(dist['bin_centers'], dist['hist'], 
                label=f'{modality} (μ={np.mean(dist["raw_intensities"]):.3f})',
                color=color, linewidth=3, alpha=0.8)
    
    # Plot augmented distribution with fill
    if augmented_distribution is not None:
        plt.fill_between(augmented_distribution['bin_centers'], 
                         augmented_distribution['hist'],
                         alpha=0.25, color='gray', 
                         label=f'Augmentation Coverage (μ={np.mean(augmented_distribution["raw_intensities"]):.3f})')
        
        # Also plot the augmented line for clarity
        plt.plot(augmented_distribution['bin_centers'], 
                 augmented_distribution['hist'],
                 color='black', linewidth=2, linestyle='--', alpha=0.8)
    
    plt.xlabel('Normalized Intensity Values', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Training Data: Modality Intensity Distributions vs Augmentation Coverage\n(Based on Training Data Only)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    
    # Add text box with summary statistics
    if augmented_distribution is not None:
        # Calculate coverage metrics
        all_mod_intensities = np.concatenate([dist['raw_intensities'] for dist in modality_distributions.values()])
        aug_intensities = augmented_distribution['raw_intensities']
        
        mod_range = np.max(all_mod_intensities) - np.min(all_mod_intensities)
        aug_range = np.max(aug_intensities) - np.min(aug_intensities)
        range_expansion = ((aug_range / mod_range) - 1) * 100 if mod_range > 0 else 0
        
        # Add text box
        total_training_images = sum(dataset_counts.values())
        textstr = f'Training Data Analysis:\n'
        textstr += f'Total Training Images: {total_training_images}\n'
        textstr += f'Modality Range: {mod_range:.3f}\n'
        textstr += f'Augmented Range: {aug_range:.3f}\n'
        textstr += f'Range Expansion: {range_expansion:+.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt.gcf()

def analyze_training_data_distributions(datasets):
    """
    Main function to analyze training data intensity distributions
    """
    print("="*80)
    print("🧠 TRAINING DATA INTENSITY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load training data
    modality_data, dataset_counts = load_training_data_by_modality(datasets)
    
    if not modality_data:
        print("❌ No training data loaded. Please check your datasets and training configuration.")
        return None, None, None
    
    # Calculate modality distributions from training data
    modality_distributions = calculate_modality_distributions(modality_data)
    
    if not modality_distributions:
        print("❌ No modality distributions calculated.")
        return None, None, None
    
    # Create augmented distribution using all training images
    augmented_distribution = create_training_augmented_distribution(modality_data, dataset_counts)
    
    # Create visualization
    fig = plot_training_intensity_distributions(modality_distributions, augmented_distribution, dataset_counts)
    
    # Print summary
    print("\n" + "="*80)
    print("📋 TRAINING DATA ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n🗂️  Datasets analyzed: {', '.join(datasets)}")
    print(f"📊 Training data modality distributions:")
    
    for modality, dist in modality_distributions.items():
        intensities = dist['raw_intensities']
        print(f"   {modality:8} - {len(intensities):,} pixels, "
              f"μ={np.mean(intensities):.3f}±{np.std(intensities):.3f}, "
              f"range=[{np.min(intensities):.3f}, {np.max(intensities):.3f}]")
    
    if augmented_distribution is not None:
        aug_intensities = augmented_distribution['raw_intensities']
        print(f"\n🔧 Augmented training data:")
        print(f"   {'Aug':8} - {len(aug_intensities):,} pixels, "
              f"μ={np.mean(aug_intensities):.3f}±{np.std(aug_intensities):.3f}, "
              f"range=[{np.min(aug_intensities):.3f}, {np.max(aug_intensities):.3f}]")
        
        # Coverage analysis
        all_mod_intensities = np.concatenate([dist['raw_intensities'] for dist in modality_distributions.values()])
        mod_range = np.max(all_mod_intensities) - np.min(all_mod_intensities)
        aug_range = np.max(aug_intensities) - np.min(aug_intensities)
        
        print(f"\n📈 Coverage Analysis:")
        print(f"   Combined modality range: {mod_range:.3f}")
        print(f"   Augmented range: {aug_range:.3f}")
        print(f"   Range expansion: {((aug_range / mod_range) - 1) * 100:+.1f}%")
        
        # Check if augmentation covers each modality's range
        print(f"\n🎯 Modality Coverage Check:")
        for modality, dist in modality_distributions.items():
            mod_min, mod_max = np.min(dist['raw_intensities']), np.max(dist['raw_intensities'])
            aug_min, aug_max = np.min(aug_intensities), np.max(aug_intensities)
            
            min_coverage = aug_min <= mod_min
            max_coverage = aug_max >= mod_max
            full_coverage = min_coverage and max_coverage
            
            coverage_status = "✅ Full" if full_coverage else ("⚠️ Partial" if min_coverage or max_coverage else "❌ Poor")
            print(f"   {modality:8}: {coverage_status} coverage")
    
    print(f"\n📈 Training Data Counts per Dataset:")
    for dataset, count in dataset_counts.items():
        print(f"   {dataset:8}: {count} training images")
    
    print(f"\n✅ Analysis complete! Training data visualization saved as 'training_intensity_distributions.png'")
    
    return modality_distributions, augmented_distribution, fig

if __name__ == "__main__":
    # Run analysis with training data only
    datasets_to_analyze = ['BRATS', 'ATLAS', 'WMH','ISLES2022,TBI']  # Datasets with available training data
    
    modality_dists, aug_dist, fig = analyze_training_data_distributions(
        datasets=datasets_to_analyze
    ) 