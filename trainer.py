import torch
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.losses import DiceCELoss
from nets.multi_unet import res_unet as unet_old
from nets.agnostic_unet import res_unet as unet_deep
import numpy as np
import utils
import wandb
import config
import argparse
import datetime
from dataloader import get_dataloader
import copy
from tqdm import tqdm


class Trainer:
    def __init__(self, train_config, aug_config, database_config, k_fold, args, channels_copy):
        self.train_config = train_config
        self.aug_config = aug_config
        self.database_config = database_config
        self.k_fold = k_fold
        self.args = args
        self.channels_copy = channels_copy
        
        torch.multiprocessing.set_sharing_strategy("file_system")

        # load config
        self.rand_assign_channels = self.train_config.rand_assign_channels
        self.domain_invariant_slot = self.train_config.domain_invariant_slot
        self.load_model_path = self.train_config.load_model_path
        self.modality_remove = self.train_config.modality_remove
        self.randomly_drop = bool(self.train_config.random_drop)
        self.single_slot = self.train_config.single_slot
        self.wandb_active = self.train_config.wandb_active
        self.agnostic_chan_augs = self.train_config.agnostic_chan_augs
        self.lr_sched = self.train_config.lr_sched
        self.held_out_datasets = self.train_config.held_out_datasets
        self.dropped_modality = self.train_config.modality_remove
        self.cropped_input_size = self.train_config.cropped_input_size
        self.epochs = self.train_config.epoch

        if self.randomly_drop and self.single_slot:
            raise ValueError("Cannot have both random drop and single slot")

        self.model_save_path = None
        self.date = None
        if self.wandb_active:
            now = datetime.datetime.now()
            self.date = now.strftime("%Y-%m-%d_%H-%M")
            self.model_save_path = os.path.join(
                self.train_config.model_save_path, self.args.datasets + "/" + self.date + "/"
            )
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            config.save_config_file(self.model_save_path)
            self._init_wandb()

        self._print_settings()
        self._setup_data()
        self._setup_model()
        self._setup_metrics()
        self._setup_channel_maps()
    
    def _init_wandb(self):
        wandb.init(
            project=self.train_config.project_name,
            name=(
                self.train_config.project_name
                + self.args.datasets
                + "_random_drop_"
                + str(self.randomly_drop)
                + "_"
                + "modality_remove:_"
                + str(self.modality_remove)
                + self.date
            ),
        )

    def _print_settings(self):
        print("lr: ", self.train_config.lr)
        print("Workers: ", self.train_config.workers)
        print("Batch Size: ", self.train_config.train_batch_size)
        print("RANDOM DROP: ", self.randomly_drop)
        print("\n #######  Training_methods #######")
        print("Domain Invariant Slot: ", self.domain_invariant_slot)
        print("Training with single input channel/slot: ", self.single_slot)
        print("Randomly assign channels: ", self.rand_assign_channels)
        print("Modality to remove: ", self.modality_remove, "\n")
        if self.k_fold:
            print(f"Training split___: {self.k_fold}")

    def _setup_data(self):
        self.img_index = 0
        self.label_index = 1
        self.mask_index = 2

        self.channels = self.database_config.channels
        if self.modality_remove is not None:
            for key in self.channels:
                self.channels[key] = [x for x in self.channels[key] if x != self.modality_remove]
            print(f"Removed {str(self.modality_remove)} from datasets")

        self.datasetlist = self.args.datasets.split("_")
        self.total_modalities = sorted(list(set(mod for d in self.datasetlist for mod in self.channels[d])))
        if self.domain_invariant_slot:
             self.total_modalities.append("invar")
        
        data_size = 0
        for dataset in self.datasetlist:
            if self.k_fold is not None:
                data_size = len(self.k_fold["train"])
            else:
                data_size = max(data_size, self.database_config.train_size[dataset])
        self.data_size = data_size
        print("Data_size", self.data_size)

        self.train_loaders, self.val_loader, self.data_loader_map = get_dataloader(
            self.train_config, self.database_config, self.datasetlist, self.cropped_input_size,
            self.data_size, self.channels_copy, self.k_fold, dataset_use="Train"
        )

        self.val_only_loader = {}
        if self.held_out_datasets:
            original_modality_remove = self.train_config.modality_remove
            self.train_config.modality_remove = None
            _, self.val_only_loader, _ = get_dataloader(
                self.train_config, self.database_config, self.held_out_datasets, self.cropped_input_size,
                self.data_size, self.channels_copy, self.k_fold, dataset_use="Val ONLY"
            )
            self.train_config.modality_remove = original_modality_remove

    def _setup_model(self):
        print("Running on GPU:" + str(self.args.device_id))
        print("Running for epochs:" + str(self.epochs))
        self.cuda_id = "cuda:" + str(self.args.device_id)
        self.device = torch.device(self.cuda_id)
        torch.cuda.set_device(self.cuda_id)

        in_channel = 1 if self.single_slot else len(self.total_modalities)
        if self.train_config.model_type == "MULTIUNET":
            print("TRAINING WITH MULTIUNET")
            self.model = unet_old(in_channels=in_channel).to(self.device)
        elif self.train_config.model_type == "AGNOSTIC_NET":
            print("TRAINING WITH AGNOSTIC NET")
            self.model = unet_deep(
                in_channels=in_channel,
                invariant_channel=self.train_config.domain_invariant_layers,
            ).to(self.device)

        print("In Channels= ", len(self.total_modalities))
        print("Batch size = ", self.train_config.train_batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.epoched = 0
        if self.train_config.load_pre_trained_model:
            print("LOADING MODEL: ", self.load_model_path)
            checkpoint = torch.load(
                self.load_model_path, map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id}
            )
            self.model.load_state_dict(checkpoint)

        self.loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
        
        if self.lr_sched:
            self.scheduler = utils.lr_schedule(epochs=self.epochs, optimizer=self.optimizer)

    def _setup_metrics(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.sensitivity_metric = ConfusionMatrixMetric(
            include_background=True, metric_name="sensitivity", reduction="mean", get_not_nans=False
        )
        self.precision_metric = ConfusionMatrixMetric(
            include_background=True, metric_name="precision", reduction="mean", get_not_nans=False
        )
        self.IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
        self.best_metric = {dataset: -1 for dataset in self.datasetlist + self.held_out_datasets}
        self.best_metric_epoch = {dataset: -1 for dataset in self.datasetlist + self.held_out_datasets}
        self.best_avg_dice = 0

    def _setup_channel_maps(self):
        self.channel_map = {
            dataset: utils.map_channels(
                self.channels[dataset], self.total_modalities, rand_assign=self.rand_assign_channels
            ) for dataset in self.datasetlist
        }

        self.combination_map = {
            dataset: utils.map_combinations(
                self.channels[dataset], invar_ratio=0.2 if self.domain_invariant_slot else 0
            ) for dataset in self.datasetlist
        }

    def train(self):
        for epoch in tqdm(range(self.epoched, self.epochs)):
            print("-" * 10)
            self.model.train()
            epoch_loss = 0
            step = 0

            if not self.lr_sched and self.train_config.drop_learning_rate and epoch >= self.train_config.drop_learning_rate_epoch:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.train_config.drop_learning_rate_value

            for batch_data in zip(*self.train_loaders):
                step += 1
                modality_outputs, labels = [], []
                
                for i, dataset in enumerate(self.datasetlist):
                    batch = batch_data[i]
                    input_data, label = self._process_batch(dataset, batch)
                    
                    modality_output = self.model(input_data)
                    modality_outputs.append(modality_output)
                    labels.append(label)

                self.optimizer.zero_grad()
                modality_outs = torch.cat(modality_outputs, dim=0)
                combined_labels = torch.cat(labels, dim=0)
                loss = self.loss_function(modality_outs, combined_labels)

                if torch.isnan(loss):
                    raise ValueError("Loss produced NaN value")
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = self.data_size // self.train_config.train_batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

                if self.wandb_active:
                    wandb.log({"loss": loss.item(), "epoch": epoch + 1, "lr": self.optimizer.param_groups[0]["lr"]})

            if self.lr_sched:
                self.scheduler.step()

            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            print("\n------------------------\n")

            if (epoch + 1) % 50 == 0:
                self._save_model(epoch, loss=epoch_loss)

            if (epoch + 1) % self.train_config.val_interval == 0:
                self._validate(epoch)
        
        if self.wandb_active:
            wandb.finish()

    def _process_batch(self, dataset, batch):
        label = batch[self.label_index].to(self.device)
        if self.randomly_drop:
            _, modalities_remaining, batch[self.img_index] = utils.rand_set_channels_to_zero_with_invar(
                self.channels[dataset], batch[self.img_index],
                mask_data=batch[self.mask_index], domain_invariant=self.domain_invariant_slot,
                batch_label_data=batch[self.label_index], device_id=self.args.device_id,
                agnostic_chan_augs=self.agnostic_chan_augs, combination_map=self.combination_map[dataset],
                augmentation_config=self.aug_config
            )
            if dataset == "BRATS" and self.database_config.BRATS_two_channel_seg:
                seg_channel = 1 if (0 not in modalities_remaining[0]) and (3 not in modalities_remaining[0]) else 0
                label = batch[self.label_index][:, [seg_channel], :, :, :].to(self.device)

        if self.single_slot:
            input_data, _ = utils.single_slot(batch[self.img_index])
        else:
            input_data = torch.zeros(
                (batch[self.img_index].shape[0], len(self.total_modalities), *self.cropped_input_size),
                dtype=torch.float32
            )
            if self.rand_assign_channels:
                self.channel_map[dataset] = utils.rand_assign_channels(self.channels[dataset], self.total_modalities)
            
            input_data[:, self.channel_map[dataset], :, :, :] = batch[self.img_index]

        return input_data.to(self.device), label

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_av_dice = []
            for dataset in [*self.datasetlist, *self.held_out_datasets]:
                metric = self._run_validation_on_dataset(dataset)
                total_av_dice.append(metric["dice"])
                self._log_and_save_best_model(dataset, metric, epoch)

            if len(self.datasetlist) > 1 and np.mean(total_av_dice) > self.best_avg_dice:
                self.best_avg_dice = np.mean(total_av_dice)
                self._save_model(epoch, best=True, avg_dice=self.best_avg_dice)
                print(f"Saved new best average dice model: {self.best_avg_dice}")
    
    def _run_validation_on_dataset(self, dataset):
        self.dice_metric.reset()
        self.sensitivity_metric.reset()
        self.precision_metric.reset()
        self.IOU_metric.reset()

        current_loader = self.val_only_loader.get(dataset) or self.val_loader.get(dataset)
        for val_data in current_loader:
            if self.single_slot:
                input_data, _ = utils.single_slot(val_data[0])
            else:
                input_data = torch.zeros((1, len(self.total_modalities), *val_data[0].shape[2:]), dtype=torch.float32)
                channel_map = utils.map_channels(self.channels_copy[dataset], self.total_modalities, rand_assign=False)
                input_data[:, channel_map, :, :, :] = val_data[0]
            
            input_data = input_data.to(self.device)
            label = val_data[1].to(self.device)

            val_outputs = sliding_window_inference(input_data, self.cropped_input_size, 1, self.model)
            val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
            
            self.dice_metric(y_pred=val_outputs, y=label)
            self.sensitivity_metric(y_pred=val_outputs, y=label)
            self.precision_metric(y_pred=val_outputs, y=label)
            self.IOU_metric(y_pred=val_outputs, y=label)
        
        return {
            "dice": self.dice_metric.aggregate().item(),
            "sensitivity": self.sensitivity_metric.aggregate()[0].item(),
            "precision": self.precision_metric.aggregate()[0].item(),
            "IOU": self.IOU_metric.aggregate().item()
        }

    def _log_and_save_best_model(self, dataset, metric, epoch):
        if metric["dice"] > self.best_metric[dataset]:
            self.best_metric[dataset] = metric["dice"]
            self.best_metric_epoch[dataset] = epoch + 1
            if epoch > 1:
                self._save_model(epoch, best=True, dataset=dataset)
                print(f"Saved new best dice model for {dataset}: {self.best_metric[dataset]}")
        
        print(
            "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                epoch + 1, dataset, metric["dice"], dataset, self.best_metric[dataset], self.best_metric_epoch[dataset]
            )
        )

        if self.wandb_active:
            wandb.log({
                "epoch_val": epoch + 1,
                f"mdice_{dataset}": metric["dice"],
                f"sensitivity_{dataset}": metric["sensitivity"],
                f"precision_{dataset}": metric["precision"],
                f"mIOU_{dataset}": metric["IOU"],
            })

    def _save_model(self, epoch, best=False, dataset=None, avg_dice=None, loss=None):
        if best:
            if dataset:
                model_save_name = f"{self.model_save_path}{self.train_config.project_name}_random_drop_{self.randomly_drop}_{self.date}_BEST_{dataset}.pth"
            elif avg_dice:
                model_save_name = f"{self.model_save_path}{self.train_config.project_name}_random_drop_{self.randomly_drop}_{self.args.datasets}{self.date}_BEST_AVERAGE.pth"
        else:
            model_save_name = f"{self.model_save_path}{self.train_config.project_name}_random_drop_{self.randomly_drop}_{self.date}_Epoch_{epoch}.pth"
            opt_save_name = f"{self.model_save_path}{self.train_config.project_name}_random_drop_{self.randomly_drop}_{self.date}_checkpoint_Epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch, "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(), "loss": loss,
            }, opt_save_name)

        torch.save(self.model.state_dict(), model_save_name)
        print("Saved Model")


if __name__ == "__main__":
    import argparse
    import config

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MultiUnet model')
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument(
        "--datasets", 
        help="datasets for training, using '_' to separate", 
        type=str,
        default=None
    )
 
  

    args = parser.parse_args()

    # Set default values (as in original script)
    args.device_id = 1
    args.datasets = 'WMH'  # 'ISLES2022'
    args.k_fold = None

    # Load configurations
    train_config = config.Training_config()
    database_config = config.Database_config()
    aug_config = config.Augmentation_config()

    args.device_id = 1
    args.datasets = 'WMH'   #'ISLES2022'
    args.k_fold = None


    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Device ID: {args.device_id}")
    print(f"Datasets: {args.datasets}")
    
    print(f"Learning Rate: {train_config.lr}")
    print(f"Batch Size: {train_config.train_batch_size}")
    print(f"Random Drop: {train_config.random_drop}")
    print(f"Single Slot: {train_config.single_slot}")
    print(f"Model Type: {train_config.model_type}")
    print("===========================\n")


    # Initialize and run trainer
    # The original script had a 'channels_copy' argument, but it's not defined here.
    # Assuming it's meant to be passed or derived.
    # For now, we'll pass a dummy value or remove it if not needed.
    # Given the original code, 'channels_copy' was derived from 'database_config.channels'.
    # Let's re-derive it here for consistency with the new code.
    datasetlist = args.datasets.split("_")
    channels_copy = {dataset: copy.deepcopy(database_config.channels[dataset]) for dataset in datasetlist}
    if train_config.modality_remove is not None:
        for dataset in datasetlist:
            channels_copy[dataset] = [x for x in channels_copy[dataset] if x != train_config.modality_remove]

    trainer = Trainer(train_config, aug_config, database_config, args.k_fold, args, channels_copy)
    trainer.train()