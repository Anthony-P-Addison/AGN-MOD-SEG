
from pathlib import Path

class Training_config():

    wandb_active:bool = True 

    epoch:int  = 600
    workers:int = 2  # numworker
    train_batch_size: int = 2
    val_interval:int = 1 # the number of epochs between the validation
    lr:float = 1e-3            
    model_type:str = "UNET"  # default model type unet
    cropped_input_size:list = [128, 128, 128]
    # lr_config
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int = 150  # epoch at which to decrease the learning rate
    drop_learning_rate_value:float = 1e-4
    project_name:str = "single_slot"   # wandb project name   # all_in_one   # shuffle_slots  # modality_invariant_slot 

    modality_remove: str = None #'T1'      # the modality to be removed (useful for testing invariant slot on this modality). None if no modality to be dropped
    model_save_path:str = "models/" + project_name + "/_model_remove:_" + str(modality_remove) + "/"  # path to save the model

    load_pre_trained_model:bool = False  # if true will load pre-train model  
    load_model_path:Path ='models/modality_invariant_slot/_model_remove:_FLAIR/BRATS/modality_invariant_slot_random_drop_1_2025-01-09_23-32_Epoch_299.pth' #"models/modality_invariant_slot/WMH_MSSEG/modality_invariant_slot_random_drop_1WMH_MSSEG_TOTAL_AVERAGE.pth"  # path to model .pt file
    # save_name = "MSSEG_NO_DROP_3" #the name of the model

    random_drop = 0  # 1 for to be dropped and 0 for not to be dropped. 
    # slot allocation
    rand_assign_channels:bool = False
    domain_invariant_slot:bool = False
    
    single_slot:bool = True
    
    

class Database_config():
    
    BRATS_two_channel_seg:bool = False  # Using different segmentation ground truths for different sets of modalities on BRATS   if true, the input that dropped the FLAIR and T2 will use ground truth that not contain edema     default false
    TBI_multichannel:bool = True    # multichannel labels
    # In our work, we put all the samples in one folder. As a quick setting, We sorted the images with the file name and split training and testing databases depending on this order.
    # when using new database the channels and length of datasets has to be specififed here.
    channels = {}
    # the modalities for each database
    channels["BRATS"] = ["FLAIR", "T1", "T1c", "T2"]
    channels["ATLAS"] = ["T1"]
    channels["MSSEG"] = ["FLAIR", "T1", "T1c", "T2", "PD"]
    channels["ISLES"] = ["FLAIR", "T1", "T2","invar"]      # DWI change to invar channel   invar 
    channels["WMH"] = ["FLAIR", "T1"]
    channels["VOETS2"] = ["FLAIR","T2","T1c"]
    channels["TBI"] = ["FLAIR", "T1", "T2", "SWI"]
    train_size = {}
    # size for each database
    # training set size
    train_size["BRATS"] = 444
    train_size["ATLAS"] = 459
    train_size["MSSEG"] = 37 
    train_size["ISLES"] = 0
    train_size["WMH"] = 42
    train_size["TBI"] = 156
    train_size["VOETS2"] = 3
    total_size = {}
    total_size["BRATS"] = 484
    total_size["ATLAS"] = 654
    total_size["MSSEG"] = 53
    total_size["ISLES"] = 28
    total_size["WMH"] = 60
    total_size["TBI"] = 281
    total_size["VOETS2"] = 7
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
    val_size["ISLES"] = 28
    val_size["MSSEG"] = 15
    val_size["WMH"] = 18
    val_size["TBI"] = 125
    val_size["VOETS2"] = 4


class Test_config():
    save_segs:bool = False  # True to save the segmentation outputs
    save_path:Path = "save_segs/"  # save niftis generated
    model_file_path:Path = 'models/single_slot/_model_remove:_None/MSSEG/single_slot_random_drop_1_2025-01-12_18-00_BEST_MSSEG.pth' # 'models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_Epoch_599.pth' #'models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth' # "models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_checkpoint_Epoch_599.pt"  # "models/MSSEG/_random_drop_0_2024-12-04_16-15_checkpoint_Epoch_99.pt" #"models/MSSEG/rand_slot_allocation_random_drop_0_2024-12-09_16-01_checkpoint_Epoch_99.pt"
    model_net_type: str = "UNET"  # the type of the pre-train model
    num_modalities_trained_on: int = 1 # the number of modalities include in training
    #model_channel_map: dict[str,list[int]] = {"VOETS2":[3,5,4],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]} #The allocated channel index of the modalities(each channel) in the testing databases (start from 0) For example "ATLAS": [3] means the T1 modality in ATLAS will be allocated to the forth channel "VOETS2":[1,5],
    
    # Slot allocation
    rand_assign:bool = False  # True to randomly assign the channels to the model
    domain_invariant_slot:bool = False
    single_slot:bool = True
    modality_remove:str = None #  'FLAIR'  #'T1'   # None if no modality is to be removed.  


class Finetune_config:
    # simialr to training
    epoch:int = 600
    workers:int = 2
    train_batch_size:int = 2
    val_interval:int = 4
    lr:float = 1e-3
    BRATS_two_channel_seg:bool = False
    model_type:str = "UNET"
    cropped_input_size:list = [128, 128, 128]
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int = 150
    drop_learning_rate_value:float = 1e-4
    # model_save_path
    model_save_path:Path = "models/finetune_checkpoints/"
    project_name:str = "all_in_one"


