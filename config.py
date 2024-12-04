
class Training_config:
        
        epoch=600
        workers = 2 #numworker
        train_batch_size = 2 
        val_interval = 4 # the number of epochs between the validation  
        lr = 1e-3
        load_pre_trained_model = False # if true will load pre-train model
        BRATS_two_channel_seg = False # Using different segmentation ground truths for different sets of modalities on BRATS   if true, the input that dropped the FLAIR and T2 will use ground truth that not contain edema     default false 
        model_type = "UNET" # default model type unet
        cropped_input_size = [128,128,128]    

        drop_learning_rate = True
        drop_learning_rate_epoch = 150 # epoch at which to decrease the learning rate
        drop_learning_rate_value = 1e-4
        # model_save_path 
        # save model  pat
        model_save_path = "models/"
        load_model_path="models/new_test_MSSEG_random_drop_0_checkpoint_Epoch_599.pt"  # path to model .pt file
        save_name = "MSSEG_NO_DROP_3" #the name of the model
        project_name = "all_in_one" #wandb project name

class Database_config:
        # In our work, we put all the samples in one folder. As a quick setting, We sorted the images with the file name and split training and testing databases depending on this order.
        # when using new database the channels and length of datasets has to be specififed here.
        channels={}
        #the modalities for each database
        channels['BRATS'] = ["FLAIR", "T1", "T1c", "T2"]
        channels['ATLAS'] = ["T1"]
        channels['MSSEG'] = ["FLAIR","T1","T1c","T2","PD"] 
        channels['ISLES'] = ["FLAIR", "T1", "T2", "DWI"]
        channels['WMH'] = ["FLAIR", "T1"]
        channels['VOETS2'] = ["T1","T2","FLAIR"]     # t1,t2, t2c
        channels['TBI'] = ["FLAIR", "T1", "T2", "SWI"]
        train_size={}
        #size for each database
        #training set size
        train_size['BRATS'] = 444
        train_size['ATLAS'] = 459
        train_size['MSSEG'] = 37
        train_size['ISLES'] = 20
        train_size['WMH'] = 42
        train_size['TBI'] = 156
        train_size['VOETS2'] = 0
        total_size={}
        total_size['BRATS'] = 484
        total_size['ATLAS'] = 654
        total_size['MSSEG'] = 53
        total_size['ISLES']= 28  
        total_size['WMH']= 60
        total_size['TBI']= 281
        total_size['VOETS2']= 10
        img_path={}
        seg_path={}
        img_path["BRATS"] = "data/BRATS/Images"
        
        if Training_config.BRATS_two_channel_seg:
                    #Need the ground truth file that has multiple channels, and each channel contains a different ground truth. !!!!Only need when you want to use multiple ground truths for BRATS, or you can ignore this
                    # Label file with two channels one for tumor core and one for whole tumor. Flair and T2 better for imaging tumor core and T1 and T1c better imaging whole tumor including edema. 
                    seg_path["BRATS"] = "data/BRATS/Labels_multiple"
        else:
                    #default setting
                    seg_path["BRATS"] = "data/BRATS/Labels"
        img_path["ATLAS"]= "data/ATLAS/Images"
        seg_path["ATLAS"]= "data/ATLAS/Labels"
        img_path["MSSEG"]="data/MSSEG/Images"
        seg_path["MSSEG"]="data/MSSEG/Labels"
        img_path["ISLES"]= "data/ISLES/Images"
        seg_path["ISLES"]= "data/ISLES/Labels"
        img_path["WMH"]="data/WMH/Images"
        seg_path["WMH"]="data/WMH/Labels"
        img_path["TBI"]="data/TBI/Images"
        seg_path["TBI"]="data/TBI/Labels"
        img_path["VOETS2"]="data/VOETS2/Images"
        seg_path["VOETS2"]="data/VOETS2/Labels"
        #!!!only for test.py
        val_size={}
        val_size["BRATS"]=40
        val_size["ATLAS"]=195
        val_size["ISLES"]=28
        val_size["MSSEG"]=16   
        val_size["WMH"]=18
        val_size["TBI"]=125
        val_size["VOETS2"]=11


class Test_config:
        save_segs= False # True to save the segmentation outputs 
        save_path = ""    
        model_file_path = "models/MSSEG_random_drop_1_BEST_MSSEG.pth"  #the path of the train model
        model_net_type = "UNet" #the type of the pre-train model
        model_modalities_trained_on = 1 #the number of modalities include in training 
        model_channel_map = {'ATLAS':[0]} #{"VOETS2":[1,5],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]} #The allocated channel index of the modalities(each channel) in the testing databases (start from 0) For example "ATLAS": [3] means the T1 modality in ATLAS will be allocated to the forth channel
        # path of testing dataset ONLY.   {"TBI":[0,1,2,3]}
class Finetune_config:
        # simialr to training    
        epoch=600
        workers = 2
        train_batch_size = 2 
        val_interval = 4
        lr = 1e-3
        BRATS_two_channel_seg = False 
        model_type = "UNET"
        cropped_input_size = [128,128,128]     
        drop_learning_rate = True
        drop_learning_rate_epoch = 150 
        drop_learning_rate_value = 1e-4
        # model_save_path
        model_save_path = "models/"




