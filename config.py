from pathlib import Path

class Training_config():

    wandb_active:bool = False
    epoch:int  = 600
    workers:int = 2 # numworker
    train_batch_size: int = 8
    val_interval:int = 4 # the number of epochs between the validation   # 4 
    lr_sched: bool = True
    lr:float = 1e-3           
    model_type:str = "UNET"  # default model type unet
    cropped_input_size:tuple = (96,96,96) # (128, 128, 128)
    # lr_config
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int = 150 # 150 # epoch at which to decrease the learning rate
    drop_learning_rate_value:float = 1e-4
    # pre trained model:
    load_pre_trained_model:bool = False  # if true will load pre-train model  
    load_model_path:Path =  'models/Mixup/_model_remove:_None/MSSEG_TBI_BRATS_WMH_ATLAS/2025-02-21_23-40/Mixup_random_drop_True_2025-02-21_23-40_Epoch_149.pth'   ##"models/modality_invariant_slot/WMH_MSSEG/modality_invariant_slot_random_drop_1WMH_MSSEG_TOTAL_AVERAGE.pth"  # path to model .pt file
    
    random_drop:int = 1  # 1 for to be dropped and 0 for not to be dropped. 

    ######### slot allocation #############

    mixup :bool = False
    gin_mix = False
    gin_ipa: str = 'GIN_IPA'   # gin
    rand_assign_channels:bool = False
    domain_invariant_slot:bool = True
    Two_domain_invariant_slot:bool = False     # TODO: see if the presence of an extra slot can help training
    single_slot:bool = False
    modality_remove: str = 'ADC' #'T1' #None    # the modality to be removed (useful for testing invariant slot on this modality). None if no modality to be dropped 
    #### admin  ####
    project_name:str = "all_in_one"   # wandb project name:   # all_in_one   # shuffle_slots  # modality_invariant_slot  # Mixup
    model_save_path:str = "models/" + project_name + "/_model_remove:_" + str(modality_remove) + "/" # path to save the model



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
    train_size = {}
    # size for each database
    # training set size
    train_size["BRATS"] = 444 
    train_size["ATLAS"] = 459
    train_size["MSSEG"] = 37   # 37 
    train_size["ISLES"] =19   #19   # 19 
    train_size["WMH"] = 42
    train_size["TBI"] = 156   #156
    train_size["VOETS2"] = 3
    train_size["ISLES2022"] = 175   
    train_size["TUMOUR2"] = 41  
    total_size = {}
    total_size["BRATS"] = 484
    total_size["ATLAS"] = 654
    total_size["MSSEG"] = 53
    total_size["ISLES"] = 28   # test 9 and train 19
    total_size["WMH"] = 60
    total_size["TBI"] = 281
    total_size["VOETS2"] = 7 
    total_size["ISLES2022"] = 250   
    total_size["TUMOUR2"] = 51
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
    img_path["ISLES2022"] = "data/ISLES_2022/Images"
    seg_path["ISLES2022"] = "data/ISLES_2022/Labels"
    img_path["TUMOUR2"] = "data/TUMOUR2/Images"
    seg_path["TUMOUR2"] = "data/TUMOUR2/Labels"

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
    val_size["ISLES"] = 9   #28  #9
    val_size["TUMOUR2"] = 10


class Test_config():
    save_segs:bool = False  # True to save the segmentation outputs
    save_path:Path = "save_segs/"  # save niftis generated
    model_file_path:Path = 'models/Mixup/_model_remove:_None/ATLAS_MSSEG_TBI_BRATS_WMH/2025-02-15_19-10/Mixup_random_drop_TrueATLAS_MSSEG_TBI_BRATS_WMH2025-02-15_19-10_BEST_AVERAGE.pth'#'models/Mixup/_model_remove:_FLAIR/MSSEG_BRATS_ATLAS_TBI_ISLES/2025-02-04_23-08/Mixup_random_drop_TrueMSSEG_BRATS_ATLAS_TBI_ISLES2025-02-04_23-08_BEST_AVERAGE.pth'  #models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_Epoch_599.pth' #'models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth' # "models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_checkpoint_Epoch_599.pt"  # "models/MSSEG/_random_drop_0_2024-12-04_16-15_checkpoint_Epoch_99.pt" #"models/MSSEG/rand_slot_allocation_random_drop_0_2024-12-09_16-01_checkpoint_Epoch_99.pt"
    model_net_type: str = "UNET"  # the type of the pre-train model
    num_modalities_trained_on: int = 6 # the number of modalities included in training
    #model_channel_map: dict[str,list[int]] = {"VOETS2":[3,5,4],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]} #The allocated channel index of the modalities(each channel) in the testing databases (start from 0) For example "ATLAS": [3] means the T1 modality in ATLAS will be allocated to the forth channel "VOETS2":[1,5],
    
    # Slot allocation
    rand_assign:bool = False  # True to randomly assign the channels to the model
    domain_invariant_slot:bool = False
    single_slot:bool = False
    modality_remove:str = None #'FLAIR'     #'T1'   # Modlality completed removed during test in dataloader. 

    modality_rem_train: str = None # 'DWI' # the modality that was removed during training and now want to test in the invariant slot. 



class Finetune_config:
    # similar to training
    epoch:int = 600
    workers:int = 2
    train_batch_size:int = 2
    val_interval:int = 4
    lr:float = 1e-3
    model_type:str = "UNET"
    cropped_input_size:list = [128, 128, 128]
    drop_learning_rate:bool = True
    drop_learning_rate_epoch:int = 150           # TODO: change to a higher value for finetuning of ISLES
    drop_learning_rate_value:float = 1e-4   # can lower this for finetuning
   
    # model_save_path
    modality_remove:str = None  # the modality to be removed (useful for testing invariant slot on this modality). None if no modality to be dropped
    project_name:str = "Fine_Tune"   # wandb project name:   # all_in_one   # shuffle_slots  # modality_invariant_slot
    model_save_path:str = "models/" + "finetune_checkpoints/"   # path to save the model
    add_slot_to_pre_trained_model:bool = False  # if true will add slot to pre-train model
    new_mod_finetune:str = None  #"DWI"   # for randomly initiating a new slot
    add_invar_channel: bool = True





