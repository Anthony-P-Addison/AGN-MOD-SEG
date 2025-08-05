from pathlib import Path
import json
import os
from datetime import datetime



class Training_config():

    # Training Parameters
    epoch:int  = 600
    workers:int = 2 #numworker
    train_batch_size: int = 2 
    val_interval:int = 4 #number of epochs between the validation         
    model_type:str = "deep_unet"   # deep_unet, old_unet
    cropped_input_size:tuple = (96,96,96)
    # lr_config
    lr:float =  1e-4  # initial lr
    lr_sched: bool = False # if False defaults to below

    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int =350     
    drop_learning_rate_value:float = 1e-5
  
    # Load pre trained model:
    load_pre_trained_model:bool = False  # if true will load pre-train model  
    load_model_path:Path =  'models/WMH_PRELIM_TEST/_model_remove:_FLAIR/TBI_ISLES2022_BRATS_MSSEG_ATLAS/2025-06-30_12-04/WMH_PRELIM_TEST_random_drop_True_2025-06-30_12-04_Epoch_49.pth'
    ## Remove modality from the training set (can use for testing the agnostic channel)
    modality_remove ='FLAIR' # Can be: str, list, or None. Single modality (str) or multiple modalities (list) to remove. None if no modality to be dropped 
    random_drop:int = 1 # 1 for to be dropped and 0 for not to be dropped. 
    #### admin  ####
    wandb_active:bool = True
    project_name:str = "Agnostic_channel"   # wandb project name
    model_save_path:str = "models/" + project_name + "/_model_remove:_" + str(modality_remove) + "/" # path to save the model
    ### Layers in model specific to domain invariant slot ###
    domain_invariant_layers:bool = True
    domain_invariant_slot:bool = True
    contrast_augmentation: bool = True
    ##### Baselines #####
    rand_assign_channels:bool = False
    single_slot:bool = False  
    
    ### HELD OUT DATASET WANT TO VALIDATE AS TRAIN WITH OR WITHOUT UNSEEN MODALITY ####
    held_out_datasets:list = ['WMH']  # datasets want to use for validation but not for training


class Database_config():
    
    BRATS_two_channel_seg:bool = False  # Using different segmentation ground truths for different sets of modalities on BRATS   if true, the input that dropped the FLAIR and T2 will use ground truth that not contain edema     default false
    TBI_multichannel:bool = False    # multichannel labels
    #### All samples from each database placed in a singular database specific folder. Images sorted by file name and  train and test databases split depending on this order.####
    # when using new database the channels and length of datasets has to be specififed here.###
    channels = {}

    # the modalities for each database
    channels["BRATS"] = ["FLAIR", "T1", "T1c", "T2"]
    channels["ATLAS"] = ["T1"]
    channels["MSSEG"] = ["FLAIR", "T1", "T1c", "T2", "PD"]
    channels["ISLES"] = ["FLAIR", "T1", "T2","DWI"]      # DWI change to invar channel   invar 
    channels["WMH"] = ["FLAIR", "T1"]
    channels["VOETS2"] = ["FLAIR","T2","T1c"]
    channels["TBI"] = ["FLAIR", "T1", "T2", "SWI"]
    channels["ISLES2022"] = ['ADC','DWI','FLAIR'] 
    channels["TUMOUR2"]  = ['T1']
    channels['VESTIBS'] = ['T1c','T2']  
    train_size = {}
    # size for each database
    # training set size
    train_size["BRATS"] = 444 #444  # 50
    train_size["ATLAS"] = 459  #459   # 50
    train_size["MSSEG"] = 37   # 37 
    train_size["ISLES"] =1  #19
    train_size["WMH"] = 42
    train_size["TBI"] = 156   #156  # 50
    train_size["VOETS2"] = 3
    train_size["ISLES2022"] = 175 #175    #50
    train_size["TUMOUR2"] = 1 
    train_size["VESTIBS"] = 176
    total_size = {}
    total_size["BRATS"] = 484
    total_size["ATLAS"] = 654
    total_size["MSSEG"] = 53
    total_size["ISLES"] = 28   # test 9 and train 19
    total_size["WMH"] = 60
    total_size["TBI"] = 281
    total_size["VOETS2"] = 7 
    total_size["ISLES2022"] = 250   
    total_size["TUMOUR2"] = 57   # 51 
    total_size["VESTIBS"] = 242
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
    img_path["TUMOUR2"] = "data/TUMOUR2/Images"
    seg_path["TUMOUR2"] = "data/TUMOUR2/Labels"
    img_path["VESTIBS"] = "data/VESTIBS/Images"
    seg_path["VESTIBS"] = "data/VESTIBS/Labels"

    img_path["MSSEG"] = "data/MSSEG/Images"
    seg_path["MSSEG"] = "data/MSSEG/Labels"
    img_path["ISLES"] = "data/ISLES/Images"
    seg_path["ISLES"] = "data/ISLES/Labels"
    img_path["WMH"] = "data/WMH/Images"
    seg_path["WMH"] = "data/WMH/Labels"
    img_path["VOETS2"] = "data/VOETS2/Images"
    seg_path["VOETS2"] = "data/VOETS2/Labels"
    img_path["TBI"] = "data/TBI/Images"  
    if TBI_multichannel:
        seg_path["TBI"] = "data/TBI_multichannel/Labels" 
    else:
        seg_path["TBI"] = "data/TBI/Labels"
                   
    #!!!only for test.py
    val_size = {}
    val_size["BRATS"] = 40 
    val_size["ATLAS"] = 195
    val_size["MSSEG"] = 15
    val_size["WMH"] = 18
    val_size["TBI"] = 125
    val_size["VOETS2"] = 4
    val_size["ISLES2022"] = 75
    val_size["ISLES"] = 28   #28  #9
    val_size["TUMOUR2"] = 57  #10
    val_size["VESTIBS"] = 66

    mask_path = {}
    mask_path["BRATS"] = "data/BRATS/Masks"
    mask_path["ATLAS"] = "data/ATLAS/Masks"
    mask_path["MSSEG"] = "data/MSSEG/Masks"
    mask_path["ISLES2022"] = "data/ISLES2022/Masks"
    mask_path["TUMOUR2"] = "data/TUMOUR2/Masks"
    mask_path["WMH"] = "data/WMH/Masks"
    mask_path["VOETS2"] = "data/VOETS2/Masks"
    mask_path["TBI"] = "data/TBI/Masks"
    mask_path["ISLES"] = "data/ISLES/Masks"
    mask_path["VESTIBS"] = "data/VESTIBS/Masks"

class Test_config():
    save_segs:bool = False  # True to save the segmentation outputs
    save_path:Path = "save_segs/"  # save niftis generated
    #model_file_path:Path = 'models/Mixup/_model_remove:_None/ATLAS_MSSEG_TBI_BRATS_WMH/2025-02-15_19-10/Mixup_random_drop_TrueATLAS_MSSEG_TBI_BRATS_WMH2025-02-15_19-10_BEST_AVERAGE.pth'#'models/Mixup/_model_remove:_FLAIR/MSSEG_BRATS_ATLAS_TBI_ISLES/2025-02-04_23-08/Mixup_random_drop_TrueMSSEG_BRATS_ATLAS_TBI_ISLES2025-02-04_23-08_BEST_AVERAGE.pth'  #models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_Epoch_599.pth' #'models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth' # "models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_checkpoint_Epoch_599.pt"  # "models/MSSEG/_random_drop_0_2024-12-04_16-15_checkpoint_Epoch_99.pt" #"models/MSSEG/rand_slot_allocation_random_drop_0_2024-12-09_16-01_checkpoint_Epoch_99.pt"
    model_net_type: str = "unet_deep"    # 'unet_old', 'unet_deep' # the type of the pre-train model
    num_modalities_trained_on: int = 7 # the number of modalities included in training
    #: dict[str,list[int]] = {"VOETS2":[3,5,4],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]} #The allocated channel index of the modalities(each channel) in the testing databases (start from 0) For example "ATLAS": [3] means the T1 modality in ATLAS will be allocated to the forth channel "VOETS2":[1,5],
    
    # Slot 
    rand_assign:bool = False # True to randomly assign the channels to the model
    domain_invariant_slot:bool =  True
    single_slot:bool = False
    modality_remove = 'DWI' #Can be: str, list, or None. Modalities to be completely removed during test in dataloader. 
    invariant_layers:bool = False
    modality_rem_train: str = 'ADC' # the modality that was removed during training and now want to test in the invariant slot.



class Finetune_config:
    # similar to training
    epoch:int = 600
    workers:int = 2
    train_batch_size:int = 2
    val_interval:int = 4
    lr:float = 1e-5
    model_type:str = "UNET"
    cropped_input_size:list = [96, 96, 96]
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int = 350           # TODO: change to a higher value for finetuning of ISLES
    drop_learning_rate_value:float = 5e-6   # can lower this for finetuning
   
    # model_save_path
    modality_remove:str = None  # the modality to be removed (useful for testing invariant slot on this modality). None if no modality to be dropped
    project_name:str = "UPPER_BOUND_FINETUNE"   # wandb project name:   # all_in_one   # shuffle_slots  # modality_invariant_slot
    model_save_path:str = "models/" + project_name + "/" # path to save the model
    wandb_report:bool = True

    # add new modality to invar channel, trained from previous model
    new_mod_finetune:str = None  # modality to be added to the invar channel for finetuning
    modality_remove_training_set: str = None  # the modality to be removed from the training set of pre trained model. None if no modality to be dropped, 
   

    # add invar channel, train from scratch
    add_invar_channel_to_pre_trained_model:bool = False
    add_invar_layers_to_pre_trained_model:bool = False

    invariant_layers:bool = False






class Augmentation_config:
    ##### Brain Tissue Augmentations #####
    prob_brain_invert: float = 0 #0.5 
    prob_brain_mixup: float = 0 #0.5
    prob_brain_scale_shift: bool = False

    brain_factor_multiply: tuple = (0.8, 1.2)
    brain_factor_intensity: tuple = (-0.2, 0.2)

    ##### Pathology Tissue Augmentations #####
    prob_pathology_switch: float = 0 #0.75 
    prob_pathology_invert: float = 0 #0.5  
    prob_pathology_mixup: float = 0 #0.5 
    prob_tumor_scale_shift: bool = False

    tumor_factor_multiply: tuple = (0.8, 1.2)
    tumor_factor_intensity: tuple = (-0.2, 0.2)

    ##### Uniform Augmentations #####
    # uniform augmentations apply same probability to both pathology and the healthy brain tissue,
    uniform_augs: bool = True
    uniform_scale_shift: bool = True
    uniform_mix_up: float = 0.5
    uniform_invert: float = 0.5






    


def save_config_file(model_save_path):
    """
    Save the entire config.py file as a text file to the model save directory.
    This preserves the exact configuration used for training.
    
    Args:
        model_save_path: Path where the model is being saved
    """
    import inspect
    
    # Get the path of the current config.py file
    current_file = inspect.getfile(inspect.currentframe())
    
    # Create configs directory in the model save path
    config_dir = os.path.join(os.path.dirname(model_save_path), 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_filename = f'config_{timestamp}.txt'
    destination_path = os.path.join(config_dir, config_filename)
    
    # Read the config.py file and save as text
    with open(current_file, 'r') as source_file:
        config_content = source_file.read()
    
    with open(destination_path, 'w') as dest_file:
        dest_file.write(config_content)
    
    print(f"Config file saved as text to: {destination_path}")
    return destination_path

