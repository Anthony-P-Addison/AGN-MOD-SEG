# MultiUnet: Multi-Modal Medical Image Segmentation

![MultiUnet Architecture](diagrams/Diagram%20-%20Page%201.png)

A deep learning framework for medical image segmentation that supports multiple imaging modalities and various medical datasets.

## Features

- Multi-modal image processing supporting various medical imaging modalities:
  - FLAIR (Fluid Attenuated Inversion Recovery)
  - T1-weighted
  - T1-weighted with contrast (T1c)
  - T2-weighted
  - DWI (Diffusion Weighted Imaging)
  - ADC (Apparent Diffusion Coefficient)
  - SWI (Susceptibility Weighted Imaging)
  - PD (Proton Density)

- Supported Datasets:
  - BRATS (Brain Tumor)
  - ATLAS
  - MSSEG (Multiple Sclerosis)
  - ISLES (Ischemic Stroke)
  - WMH (White Matter Hyperintensities)
  - TBI (Traumatic Brain Injury)
  - ISLES2022
  - And more

- Advanced Features:
  - Domain-invariant slot learning
  - Modality dropout training
  - K-fold cross-validation support
  - Flexible model architecture (Deep U-Net variants)
  - Wandb integration for experiment tracking
  - Data augmentation
  - Custom masking generation

## Requirements

- Python 3.x
- PyTorch
- Weights & Biases (wandb)
- Additional dependencies listed in `requirements.txt`

## Project Structure

```
MultiUnet/
├── config.py           # Configuration settings for training and databases
├── train_2.py         # Main training script
├── train_finetune.py  # Fine-tuning script
├── k_fold.py         # K-fold cross-validation implementation
├── masking.py        # Mask generation utilities
└── data/             # Dataset directory
    ├── BRATS/
    ├── ATLAS/
    ├── MSSEG/
    └── ...
```

## Configuration

The project uses configuration classes in `config.py`:

- `Training_config`: Controls training parameters like:
  - Learning rate
  - Batch size
  - Number of epochs
  - Model architecture
  - Input size
  - Wandb settings

- `Database_config`: Manages dataset-specific settings:
  - Available modalities per dataset
  - Dataset sizes
  - File paths
  - Training/testing splits

## Usage

1. Prepare your data:
   ```bash
   # Organize your data in the data/ directory according to the dataset structure
   data/
   ├── DATASET_NAME/
   │   ├── Images/
   │   └── Labels/
   ```

2. Configure settings:
   - Adjust parameters in `config.py` according to your needs
   - Set up your wandb project if using experiment tracking

3. Run training:
   ```bash
   python train_2.py --datasets "DATASET1,DATASET2" [additional arguments]
   ```

4. For k-fold cross-validation:
   ```bash
   python train_2.py --datasets "DATASET1" --k_fold N
   ```

## Model Architecture

The project implements multiple U-Net variants optimized for multi-modal medical image segmentation. The architecture supports:
- Multiple input modalities
- Domain-invariant feature learning
- Modality dropout for robustness
- Flexible slot allocation for different imaging protocols

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Citation

[Add citation information if applicable]

## Contact

[Add contact information if desired] 