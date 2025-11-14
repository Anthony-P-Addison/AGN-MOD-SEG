from pathlib import Path
import json
import os
from datetime import datetime


class Training_config():
    # Training Parameters
    epoch:int  = 600
    workers:int = 2 #numworker
    train_batch_size: int = 2
    val_interval:int = 8 #number of epochs between the validation         
    model_type:str = "AGNOSTIC_NET"   # AGNOSTIC_NET, MULTIUNET
    cropped_input_size:tuple = (96,96,96)
    # lr_config
    lr:float =  1e-4  # initial lr
    lr_sched: bool = False  # if False defaults to below
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int =350     
    drop_learning_rate_value:float = 1e-5

    # Load pre trained model:
    load_pre_trained_model:bool = False  # if true will load pre-train model  
    load_model_path:Path =  None
    ## Remove modality from the training set (can use for testing the agnostic channel)
    modality_remove = 'FLAIR' # Can be: str, list, or None. Single modality (str) or multiple modalities (list) to remove. None if no modality to be dropped 
    random_drop:int = 1 # 1 for to be dropped and 0 for not to be dropped. 
    #### admin  ####
    wandb_active:bool = True
    project_name:str = "Agnostic_channel"   # wandb project name
    model_save_path:str = "models/" + project_name + "/_model_remove:_" + str(modality_remove) + "/" # path to save the model
    ##### Baselines #####
    rand_assign_channels:bool = False
    single_slot:bool = False  

    held_out_datasets:list = [] # datasets want to use for validation but not for training


class Database_config():
    BRATS_two_channel_seg:bool = False  # Using different segmentation ground truths for different sets of modalities on BRATS   if true, the input that dropped the FLAIR and T2 will use ground truth that not contain edema     default false
    TBI_multichannel:bool = False    # multichannel labels
    #### All samples from each database placed in a singular database specific folder. Images sorted by file name and  train and test databases split depending on this order.####
    # when using new database the channels and length of datasets has to be specififed here.###
    channels = {}

    # modalities for each database
    channels["BRATS"] = ["FLAIR", "T1", "T1c", "T2"]
    channels["ATLAS"] = ["T1"]
    channels["MSSEG"] = ['FLAIR', "T1", "T1c", "T2", "PD"]
    channels["ISLES"] = ["FLAIR", "T1", "T2", "DWI"]
    channels["WMH"] = ["FLAIR", "T1"]
    channels["ISLES2022"] = ['ADC','DWI','FLAIR'] 
    channels["TBI"] = ["FLAIR", "T1", "T2", "SWI"]

    # training set size
    train_size = {}
    train_size["BRATS"] = 444
    train_size["ATLAS"] = 459
    train_size["MSSEG"] = 37
    train_size["ISLES"] = 0
    train_size["WMH"] = 42
    train_size["ISLES2022"] = 175
    train_size["TBI"] = 42

    #total size 
    total_size = {}
    total_size["BRATS"] = 484
    total_size["ATLAS"] = 654
    total_size["MSSEG"] = 53
    total_size["ISLES"] = 28   
    total_size["WMH"] = 60
    total_size["ISLES2022"] = 250   

    #image and seg map paths
    img_path = {}
    seg_path = {}
    img_path["BRATS"] = "data/BRATS/Images"

    if BRATS_two_channel_seg:
        # Need the ground truth file that has multiple channels, and each channel contains a different ground truth. !!!!Only need when you want to use multiple ground truths for BRATS, or you can ignore this
        # Label file with two channels one for tumor core and one for whole tumor. Flair and T2 better for imaging tumor core and T1 and T1c better imaging whole tumor including edema.
        seg_path["BRATS"] = "data/BRATS/Labels_multiple"
    else:
        # default setting
        seg_path["BRATS"] = "data/BRATS/Labels"
    img_path["ATLAS"] = "data/ATLAS/Images"
    seg_path["ATLAS"] = "data/ATLAS/Labels"
    img_path["ISLES2022"] = "data/ISLES2022/Images"
    seg_path["ISLES2022"] = "data/ISLES2022/Labels"
    img_path["MSSEG"] = "data/MSSEG/Images"
    seg_path["MSSEG"] = "data/MSSEG/Labels"
    img_path["ISLES"] = "data/ISLES/Images"
    seg_path["ISLES"] = "data/ISLES/Labels"
    img_path["WMH"] = "data/WMH/Images"
    seg_path["WMH"] = "data/WMH/Labels"
   
    #only for test
    val_size = {}
    val_size["BRATS"] = 40 
    val_size["ATLAS"] = 195
    val_size["MSSEG"] = 15
    val_size["WMH"] = 18
    val_size["ISLES2022"] = 75
    val_size["ISLES"] = 28   

    #brain mask path 
    mask_path = {}
    mask_path["BRATS"] = "data/BRATS/Masks"
    mask_path["ATLAS"] = "data/ATLAS/Masks"
    mask_path["MSSEG"] = "data/MSSEG/Masks"
    mask_path["ISLES2022"] = "data/ISLES2022/Masks"
    mask_path["WMH"] = "data/WMH/Masks"
    mask_path["ISLES"] = "data/ISLES/Masks"


class Test_config():
    save_segs:bool = False  # True to save the segmentation outputs
    save_path:Path = "save_segs/"  # save niftis generated of segmentation masks
    croppped_input_size: list = [96, 96, 96] # ensure is same as training cropped input size
    # baseline
    single_slot:bool = False   # True to use a single slot for the model
    rand_assign:bool = False   # True to randomly assign the channels to the model

    model_net_type:str = "agnostic_net"   


class Finetune_config:
    epoch:int = 600
    workers:int = 2
    randomly_drop: bool = True
    train_batch_size:int = 2
    val_interval:int = 4
    lr:float = 1e-5
    model_type: str = "AGNOSTIC_NET"
    cropped_input_size:list = [96, 96, 96]
    drop_learning_rate:bool = True
    drop_learning_rate_epoch: int = 350
    drop_learning_rate_value: float = 5e-6

    # model_save_path
    project_name: str = "Agnostic_finetune"
    model_save_path:str = "models/" + project_name + "/" # path to save the model
    wandb_report:bool = True


class Augmentation_config:
    ## augmentaitons for the input to agnostic channel.
    ##### Brain Tissue Augmentations #####
    prob_brain_invert: float = 0.5
    prob_brain_mixup: float = 0.5
    prob_brain_scale_shift: bool = True

    brain_factor_multiply: tuple = (0.8, 1.2)
    brain_factor_intensity: tuple = (-0.2, 0.2)

    ##### Pathology Tissue Augmentations #####
    prob_lesion_switch: float = 0.75
    prob_pathology_invert: float = 0.5
    prob_pathology_mixup: float = 0.5
    prob_tumor_scale_shift: bool = False

    tumor_factor_multiply: tuple = (0.8, 1.2)
    tumor_factor_intensity: tuple = (-0.2, 0.2)

    ##### Uniform Augmentations #####
    # uniform augmentations apply same probability to both pathology and the healthy brain tissue,
    uniform_augs: bool = False
    uniform_scale_shift: bool = False
    uniform_mix_up: float = 0
    uniform_invert: float = 0
