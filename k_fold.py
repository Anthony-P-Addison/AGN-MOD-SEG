from train_2 import main
from train_finetune import main as main_finetune
from config import Training_config, Database_config, Augmentation_config , Finetune_config
import argparse
import numpy as np
import copy
import torch
import wandb


####### k_fold cross validation #######

# create function which goes into dataset of argsdatsaet and return a dictionary of 3 splits of the data

def split_dataset(dataset_size, k_fold=int):
    indices = np.arange(dataset_size)

    if k_fold:
        fold_size = dataset_size // k_fold
        splits = []
        for i in range(k_fold):
            val_idx = indices[i * fold_size : (i + 1) * fold_size]
            train_idx = np.concatenate(
                (indices[: i * fold_size], indices[(i + 1) * fold_size :])
            )
            splits.append({"train": train_idx, "val": val_idx})

    return splits

#################################################



##########

 #command line argument
par = argparse.ArgumentParser()
par.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
par.add_argument("--datasets", help="datasets for training, using '_' to separate", type=str)
#par.add_argument("--save_name", help="File name for saving model weights and checkpoints", type=str, default='save')
par.add_argument("--randomly_drop", help="0 or 1, 1 if random dropping modalities when training", type=int, default='1')
par.add_argument("--load_model_finetune_path", help="The path of the pretrained model", type=str)
par.add_argument("--manual_channel_map", help="The allocated channel index of the modalities(each channel) in the finetuning input (start from 0)     Using '_' to separate.  For example, 1_3 means the first modality in the finetuning input goes to the second channel of the model, and the second modality goes to the fourth channel of the model.", type=str)
par.add_argument("--datasets_trained_initially", help="modalities used for training the pre-train model using '_' to separate", type=str)
finetune_args = par.parse_args()



finetune_args.datasets = "ISLES"
finetune_args.randomly_drop = 1
finetune_args.load_model_finetune_path = 'models/BASELINE/_model_remove:_None/TBI_WMH_BRATS_MSSEG_ATLAS/2025-06-12_15-01/WMH_PRELIM_TEST_random_drop_True_2025-06-12_15-01_Epoch_599.pth'
finetune_args.datasets_trained_initially = 'TBI_WMH_BRATS_MSSEG_ATLAS'  
finetune_args.device_id = 1

k_fold = 4

total_size = Database_config.total_size[finetune_args.datasets]
split_datasets = split_dataset(total_size, k_fold)

train_config =  Training_config()
database_config = Database_config()
aug_config = Augmentation_config()
channels_copy = copy.deepcopy(database_config.channels)

wandb_report = True
finetune = True
###########

for i, split in enumerate(split_datasets):

    if finetune:
        main_finetune(finetune_args,k_fold=split)
        wandb.finish()
    else:
        main(train_config,aug_config,database_config,split,finetune_args,channels_copy,optuna_trial=None)

     # Clean up after each fold
    import gc
    import torch
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()





