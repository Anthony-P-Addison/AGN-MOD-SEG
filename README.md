# Modality-Agnostic Input Channels Enable Segmentation of Brain lesions in Multimodal MRI with Sequences Unavailable During Training

This repository contains the official Pytorch implementation of the following paper:

 <u>[Modality Agnostic Input Channels Enable... ](https://arxiv.org/html/2509.09290v1)</u>

<img src="diagrams/Input_Architecture.png" alt="MultiUnet Architecture" width="800">

## Introduction

This work presents a novel **modality-agnostic architecture** for 3D medical image segmentation that  handles heterogeneous combinations of both **seen and unseen imaging modalities**. 

**Key Innovation**: Our approach enables robust brain lesion segmentation even when encountering imaging sequences that were not available during training, addressing a critical challenge in real-world clinical deployment.

**Architecture Design**: We integrate modality-agnostic input pathways with modality-specific processing channels, creating a flexible framework that can adapt to varying modality combinations without retraining.

 **Training Strategy**: The model is trained using innovative **Modality Dropout** techniques and augmentation strategies that synthesize artificial contrasts between brain tissue and lesions, enhancing generalization across diverse imaging protocols.



## Installation
To install code and dependencies clone the repository:
```bash
git clone https://github.com/Anthony-P-Addison/AGN-MOD-SEG.git
```

Create conda virtual environment: 
```bash
conda env create -f environment_current.yml
conda activate agn_mod_seg
```



## Project Structure

```
MultiUnet/
├── config.py          # Configuration settings for training,testing, finetuning  and database  selection
├── trainer.py         # Main training script
├── train_finetune.py  # Fine-tuning script
├── k_fold_train.py    # K-fold cross-validation implementation
├── masking.py         # Mask generation utilities for datasets. 
├── test.py            # Main test script
├── utils.py           # additional functions 
├── nets               # folder containing network componenets    
├──      
└── data/              # Dataset directory (can add your own datasset folder as appropriate)
    ├── BRATS/
    ├── ATLAS/
    ├── MSSEG/
    └── ...
```

## Usage

The pipeline is set up to be trained on one or more datasets of 3D medical images (in the paper it is segmentation of brain lesions but can be applied to another anatomies/imaging modalities). 


1. Prepare your data:
   ```bash
   # Organize your data in the data/ directory according to the dataset structure. Corresponding labels must be in same order directory as images. Name the same.
   data/
   ├── DATASET_NAME/
   │   ├── Images/
   │   └── Labels/
   ```

2. Create brain masks to allow application of tissue specific augmentations. Corresoponsing masks will be located in the same directoy as the corresponding images and labels. 
   
   ```bash
   python masking.py --datasets "DATASET1"
   ```

   Saved to mask folder as follows:

   ```bash
   data/
   ├── DATASET_NAME/
   │   ├── Images/
   │   ├── Labels/
   │   └── Masks/
   ```


2. Configure settings:
   - Adjust parameters in `config.py` according to your needs
      - add your own databses in the config file under the database config EXAMPLE BRATS (in your instance you would fill in the information for your dataset/s in place of BRATS): 
         ```bash
         channels["BRATS"] = ["FLAIR", "T1", "T1c", "T2"]
         ...
         train_size["BRATS"] = 444 
         ...
         total_size["BRATS"] = 484
         ...
         img_path["BRATS"] = "data/BRATS/Images"
         ...
         seg_path["BRATS"] = "data/BRATS/Labels"
         ...
         val_size["BRATS"] = 40 
         ...
         mask_path["BRATS"] = "data/BRATS/Masks"

         ```

      
   - Set up your wandb project if using experiment tracking


## Model Architecture

The project implements multiple U-Net variants optimized for multi-modal medical image segmentation.

<!-- ![Model Information](diagrams/model_comparison_table.png) -->
<img src="diagrams/model_comparison_table.png" alt="Model Information" width="250">

**Agn. Path:** Additional agnostic channel and pre processing path for input

**Agn. Channel:** Additional agnostic channel to process new modality input



<u>**Baselines**</u>

The following baselines can be trained/tested and adapted as fit:

**Standard Model:** Train the model with with channels equal to the number of modalities across the training database. ** CANNOT PROCESS UNSEEN MODALITY

**Shuffled:**  Modalities are randomly shuffled between input channels during training.

**Single:**   Single input channel to the model during training modalities randomly assigned. 



## Quick Use

Use pre trained model for inference on data with seen modalities, unseen modalities and combination of seen and unseen modality. WHere unseen modality is assigned to the agnostic channel and seen modalities are asssigned to modality specific input channels, the model below was trained on the following modalities ['FLAIR', 'PD', 'SWI', 'T1', 'T1c', 'T2'] hence, if you have a modality not here you can place in agnostic channel
In this example we test with ISLES15 (stroke dataset) with a combination of seen and unseen modalities. Unseen modality DWI is placed in agnostic channel: (ISLES15 is publically available to test). There are a variety of models to choose from in the checkpoint_path_json and can be iniated with the --model_name argument. 
 ```bash
         # Pre trained agnostic path model:
         python test.py --datasets_to_test 'ISLES' --modalities_to_test "0_1_2_3" --device_id 0 --trained_on 'TBI_WMH_BRATS_MSSEG_ATLAS' --model_name 'setting_1_agnostic_path' --agnostic_path --agnostic_channel --modality_for_agnostic_channel 'DWI' 


         # pre trained agnostic channel (no agnostic path only channel) 
         python test.py --datasets_to_test 'ISLES' --modalities_to_test "0_1_2_3" --device_id 1 --trained_on 'TBI_WMH_BRATS_MSSEG_ATLAS' --model_name 'setting_1_agnostic_channel' --agnostic_channel --modality_for_agnostic_channel 'DWI'


         # pre trained standard model (no agnostic channel or path, hence cannot utilise new modality)
         python test.py --datasets_to_test 'ISLES' --modalities_to_test "0_1_2" --device_id 1 --trained_on 'TBI_WMH_BRATS_MSSEG_ATLAS' --model_name 'setting_1_standard' --modality_remove 'DWI'
         
 ```

 There are also checkpoints from setting 2 from the paper- where FLAIR was removed during training and used in agnostic channel at test time.
 For testing your own checkpoint set arg --checkpoint to path of where your checkpoint is located and set --model name to None. 






## Training with Agnostic channel AND path

1. Run training (on predefined train/test split as defined in the config file):
```bash
         python trainer.py --datasets "DATASET1_DATASET2..." --device_id 1 --agnostic_channel True --agnostic_path True --agnostic_chan_augs True --modality_remove None
   ```

2. For k-fold cross-validation:
```bash
       python k_fold_train.py --datasets "DATASET1_DATASET2..." --k_fold 7 --agnostic_channel True --agnostic_path True --agnostic_chan_augs True --modality_remove 'FLAIR'
```
To train on just standard model set: --agnostic_channel False and set --agnostic_path False. In this instance the input channels are all modality specific. 

## Finetuning with Agnostic Channel and Path 

1. Finetuning (allow fintune of dataset with seen, unseen, or combinatino of seen and unseen modalities) by utlising the agnostic channel input for previously inseen modalities:
```bash
            python train_finetune.py --finetune_dataset "ISLES" --device_id 0 --load_model_finetune_path 'setting_1_agnostic_path' --add_agnostic_path_to_pre_trained_model False --add_agnostic_channel_to_pre_trained_model False --pre_trained_agnostic_path True --pre_trained_agnostic_channel False --new_mod_finetune 'DWI' --datasets_trained_initially 'BRATS_MSSEG_ATLAS_TBI_WMH'  --modality_remove_training_set 'DWI' --modality_remove_validation_set None

```




**Citation**  
When Citing this work, please use:
 ```
@inproceedings{
addison2025modalityagnostic,
title={Modality-agnostic input channels enable segmentation of brain lesions in multi-modal {MRI} with sequences unavailable during training},
author={Anthony P. Addison and Felix Wagner and Wentian Xu and Natalie Voets and Konstantinos Kamnitsas},
booktitle={ML-CDS2025:  Multimodal Learning and Fusion Across Scales for Clinical Decision Support},
year={2025},
url={https://openreview.net/forum?id=zDzkXv0O07}
}
 
 ```


## License

This project is licensed under the MIT license

