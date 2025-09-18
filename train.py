import config
import argparse
import copy
from trainer import Trainer


def main(train_config, aug_config, database_config, k_fold, args, channels_copy):
    trainer = Trainer(train_config, aug_config, database_config, k_fold, args, channels_copy)
    trainer.train()

if __name__ == "__main__":

    # command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument(
        "--datasets", help="datasets for training, using '_' to separate", type=str
    )
    parser.add_argument(
        "--k_fold",
        help="k_fold cross validation number fo folds",
        type=int,
        default=None,
    )

    #########################
    args = parser.parse_args()
    args.device_id = 1
    args.datasets =   "ISLES2022_MSSEG_BRATS_TBI_ATLAS_WMH"
    ######################################

    train_config = config.Training_config()
    database_config = config.Database_config()
    channels_copy = copy.deepcopy(database_config.channels)
    aug_config = config.Augmentation_config()

    main(
        train_config,
        aug_config,
        database_config,
        k_fold=None,
        args=args,
        channels_copy=channels_copy,
    )
