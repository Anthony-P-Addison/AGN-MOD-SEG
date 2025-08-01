from train_2 import main as main_train
from train_finetune import main as main_finetune
from config import Training_config, Database_config, Augmentation_config , Finetune_config
import argparse
import numpy as np
import copy
import torch
import wandb
import gc


def k_fold_split(dataset_size, k_fold=int):
    """
    Split the dataset into k_fold splits.
    """
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




def main(finetune_args):
    """Main execution function."""

    total_size = Database_config.total_size[finetune_args.datasets]
    k_fold_splits = k_fold_split(total_size, finetune_args.k_fold)

    train_config =  Training_config()
    database_config = Database_config()
    aug_config = Augmentation_config()
    channels_copy = copy.deepcopy(database_config.channels)
    for i, split in enumerate(k_fold_splits):
        if finetune_args.finetune:
            main_finetune(finetune_args,k_fold=split)
            wandb.finish()
        else:
            main_train(train_config,aug_config,database_config,split,finetune_args,channels_copy)
        # Clean up after each fold
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()



if __name__ == "__main__":
    #command line argument
    par = argparse.ArgumentParser()
    par.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    par.add_argument("--datasets", help="datasets for training, using '_' to separate", type=str)
    par.add_argument("--randomly_drop", help="0 or 1, 1 if random dropping modalities when training", type=int, default='1')
    par.add_argument("--load_model_finetune_path", help="The path of the pretrained model", type=str)
    par.add_argument("--manual_channel_map", help="The allocated channel index of the modalities(each channel) in the finetuning input (start from 0)     Using '_' to separate.  For example, 1_3 means the first modality in the finetuning input goes to the second channel of the model, and the second modality goes to the fourth channel of the model.", type=str)
    par.add_argument("--datasets_trained_initially", help="modalities used for training the pre-train model using '_' to separate", type=str)
    par.add_argument("--k_fold", help="number of folds", type=int, default=4)
    par.add_argument("--finetune", help="0 or 1, 1 if finetuning the model", type=bool, default=False)

    finetune_args = par.parse_args()

    finetune_args.datasets = "ISLES"
    finetune_args.randomly_drop = 1
    finetune_args.load_model_finetune_path = 'models/BASELINE/_model_remove:_None/TBI_WMH_BRATS_MSSEG_ATLAS/2025-06-12_15-01/WMH_PRELIM_TEST_random_drop_True_2025-06-12_15-01_Epoch_599.pth'
    finetune_args.datasets_trained_initially = 'TBI_WMH_BRATS_MSSEG_ATLAS'  
    finetune_args.device_id = 1
    finetune_args.k_fold = 4
    finetune_args.finetune = False
    
    main(finetune_args)