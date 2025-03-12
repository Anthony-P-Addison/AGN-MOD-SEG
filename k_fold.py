from train_2 import main
from config import Training_config, Database_config
import argparse
import numpy as np
import copy


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



# command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
parser.add_argument(
    "--datasets", help="datasets for training, using '_' to separate", type=str
)
parser.add_argument(
    "--k_fold", help="k_fold cross validation number fo folds", type=int, default=None
)

#########################
args = parser.parse_args()
args.device_id = 0
args.datasets = "TBI"
args.k_fold = 5

######################################

total_size = Database_config.total_size[args.datasets]

split_datasets = split_dataset(total_size, args.k_fold)

train_config =  Training_config()
database_config = Database_config()
channels_copy = copy.deepcopy(database_config.channels)




for i, split in enumerate(split_datasets):

    main(train_config,database_config,split,args,channels_copy)



#TODO: fors isles try a batch size of 8/10 and  try a slightly smaller max learning rate. 