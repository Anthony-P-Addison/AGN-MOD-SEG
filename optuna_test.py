from train_2 import main
import config
import argparse
import copy
import optuna


if __name__ == "__main__":

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
    args.device_id = 0
    args.datasets = "TBI_ISLES2022_BRATS_MSSEG_ATLAS"  #'ISLES2022'

    ######################################

    train_config = config.Training_config()
    database_config = config.Database_config()
    augmentation_config = config.Augmentation_config()
    channels_copy = copy.deepcopy(database_config.channels)

    def objective(trial):
        augmentation_config.prob_brain_invert = trial.suggest_float(
            "prob_brain_invert", 0, 1, step=0.1
        )
        augmentation_config.prob_brain_mixup = trial.suggest_float(
            "prob_brain_mixup", 0, 1, step=0.1
        )
        augmentation_config.prob_pathology_switch = trial.suggest_float(
            "prob_pathology_switch", 0, 1, step=0.1
        )
        augmentation_config.prob_pathology_invert = trial.suggest_float(
            "prob_pathology_invert", 0, 1, step=0.1
        )
        augmentation_config.prob_pathology_mixup = trial.suggest_float(
            "prob_pathology_mixup", 0, 1, step=0.1
        )

        val_dice = main(
            train_config,
            augmentation_config,
            database_config,
            k_fold=None,
            args=args,
            channels_copy=channels_copy,
            optuna_trial=trial,
        )

        return val_dice

    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="Tuning_Augmentation_Parameters",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    print("Starting Optuna optimization for augmentation parameters...")

    study.optimize(objective, n_trials=20, sampler=sampler)

    print(study.best_trial.params)
    print(study.best_trial.value)
