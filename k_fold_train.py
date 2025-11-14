from train_class import main as main_train
from config import Training_config, Database_config, Augmentation_config
import argparse
import numpy as np
import copy
import torch
import gc
import utils


def k_fold_split(dataset_size, k_fold: int):
    """
    Split the dataset into k_fold splits.
    """
    indices = np.arange(dataset_size)
    splits = []
    if k_fold and k_fold > 1:
        fold_size = dataset_size // k_fold
        for i in range(k_fold):
            val_idx = indices[i * fold_size : (i + 1) * fold_size]
            train_idx = np.concatenate(
                (indices[: i * fold_size], indices[(i + 1) * fold_size :])
            )
            splits.append({"train": train_idx, "val": val_idx})
    else:
        # Single split: all train, empty val
        splits.append({"train": indices, "val": np.array([], dtype=int)})
    return splits


def main(args):
    """Run k-fold training using train_class.py"""
    train_config = Training_config()
    database_config = Database_config()
    aug_config = Augmentation_config()
    channels_copy = copy.deepcopy(database_config.channels)

    # Choose dataset for splitting; if multiple provided, use the last (consistent with data loader usage)
    dataset_list = args.datasets.split("_")
    split_dataset = dataset_list[-1]

    total_size = database_config.total_size[split_dataset]
    splits = k_fold_split(total_size, args.k_fold)

    for split in splits:
        main_train(train_config, aug_config, database_config, split, args, channels_copy)
        # Cleanup after each fold
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument(
        "--datasets", help="datasets for training, using '_' to separate", type=str
    )
    parser.add_argument(
        "--k_fold", help="k_fold cross validation number of folds", type=int, default=4
    )
    parser.add_argument(
        "--agnostic_channel", help="use agnostic channel", type=utils.str2bool, default=False
    )
    parser.add_argument(
        "--agnostic_path", help="use agnostic path", type=utils.str2bool, default=False
    )
    parser.add_argument(
        "--agnostic_chan_augs",
        help="use agnostic channel augmentations",
        type=utils.str2bool,
        default=False,
    )
    parser.add_argument(
        "--modality_remove", help="modality to be removed", type=str, default=None
    )

    args = parser.parse_args()

    args.datasets = "WMH"
    args.k_fold = 7
    args.agnostic_channel = False
    args.agnostic_path = True
    args.agnostic_chan_augs = True
    args.modality_remove = 'FLAIR'

    if args.agnostic_path:
        args.agnostic_channel = True

    if arg.agnostic_channel = True
        assert len()

    main(args)

python k_fold_train.py --datasets "DATASET1_DATASET2..." --k_fold 7 --agnostic_channel True --agnostic_path True --agnostic_chan_augs True --modality_remove 'FLAIR'