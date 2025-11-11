import argparse
import copy
import datetime
import os

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import ConfusionMatrixMetric, DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from tqdm import tqdm

import config
import utils
import wandb
from dataloader import get_dataloader
from nets.agnostic_unet import res_unet as unet_deep
from nets.multi_unet import res_unet as unet_old


class ModelTrainer:
    def __init__(self, train_config, aug_config, database_config, k_fold, args, channels_copy):
        self.train_config = train_config
        self.aug_config = aug_config
        self.database_config = database_config
        self.k_fold = k_fold
        self.args = args
        self.channels_copy = channels_copy

        torch.multiprocessing.set_sharing_strategy("file_system")

        self._extract_config_values()
        self._setup_save_path_and_wandb()
        self._print_training_settings()
        self._setup_datasets_and_channels()
        self._prepare_dataloaders()
        self._initialize_device_and_model()
        self._initialize_metrics()
        self._initialize_channel_maps()

    def _extract_config_values(self):
        self.rand_assign_channels = self.train_config.rand_assign_channels
        self.agnostic_channel = self.args.agnostic_channel
        self.agnostic_path = self.args.agnostic_path
        self.load_model_path = self.train_config.load_model_path
        self.modality_remove = self.train_config.modality_remove
        self.randomly_drop = bool(self.train_config.random_drop)
        self.single_slot = self.train_config.single_slot
        self.wandb_active = self.train_config.wandb_active
        self.agnostic_chan_augs = self.train_config.agnostic_chan_augs
        self.lr_sched = self.train_config.lr_sched
        self.held_out_datasets = self.train_config.held_out_datasets or []
        self.cropped_input_size = self.train_config.cropped_input_size
        self.epochs = self.train_config.epoch

        if self.randomly_drop and self.single_slot:
            raise ValueError("Cannot have both random drop and single slot")

        self.img_index = 0
        self.label_index = 1
        self.mask_index = 2

    def _setup_save_path_and_wandb(self):
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

    def _print_training_settings(self):
        print("\n #######  Training Hyperparameters #######")
        print("lr: ", self.train_config.lr)
        print("Batch Size: ", self.train_config.train_batch_size)
        print("Random Modality Drop: ", self.randomly_drop)
        print("\n #######   Training Settings  #######")
        print("Agnostic Channel: ", self.agnostic_channel)
        print("Agnostic Path: ", self.agnostic_path)
        print("Training with single input channel: ", self.single_slot)
        print("Randomly assign channels: ", self.rand_assign_channels)
        print("Modality to remove from training datasets: ", self.modality_remove, "\n")
        if self.k_fold:
            print(f"Training split___: {self.k_fold}")

    def _setup_datasets_and_channels(self):
        if self.modality_remove is not None:
            for key, value in self.database_config.channels.items():
                self.database_config.channels[key] = [x for x in value if x != self.modality_remove]
            print(f"Removed {str(self.modality_remove)} from datasets")

        self.channels = self.database_config.channels
        self.train_size = self.database_config.train_size
        self.datasetlist = self.args.datasets.split("_")

        total_modalities = set()
        data_size = 0

        for dataset in self.datasetlist:
            if self.agnostic_channel:
                self.channels[dataset].append("invar")

            if self.k_fold is not None:
                data_size = len(self.k_fold["train"])
            else:
                data_size = max(data_size, self.train_size[dataset])

            total_modalities = total_modalities.union(set(self.channels[dataset]))

        self.total_modalities = sorted(list(total_modalities))
        self.data_size = data_size
        print('\n #######  Data Information #######')
        print("Data_size", self.data_size)

    def _prepare_dataloaders(self):
        (
            self.train_loaders,
            self.val_loader,
            self.data_loader_map,
        ) = get_dataloader(
            self.train_config,
            self.database_config,
            self.datasetlist,
            self.cropped_input_size,
            self.data_size,
            self.channels_copy,
            self.k_fold,
            dataset_use="Train",
        )

        self.val_only_loader = {}
        if self.held_out_datasets:
            original_modality_remove = self.train_config.modality_remove
            self.train_config.modality_remove = None
            _, self.val_only_loader, _ = get_dataloader(
                self.train_config,
                self.database_config,
                self.held_out_datasets,
                self.cropped_input_size,
                self.data_size,
                self.channels_copy,
                self.k_fold,
                dataset_use="Val ONLY",
            )
            self.train_config.modality_remove = original_modality_remove

    def _initialize_device_and_model(self):
        print("Running on GPU:" + str(self.args.device_id))
        print("Running for epochs:" + str(self.epochs))

        self.cuda_id = "cuda:" + str(self.args.device_id)
        self.device = torch.device(self.cuda_id)
        torch.cuda.set_device(self.cuda_id)

        in_channel = 1 if self.single_slot else len(self.total_modalities)

        print('\n #######  Model Information #######')
        if self.train_config.model_type == "MULTIUNET":
            print("Model: MultiUnet")
            self.model = unet_old(in_channels=in_channel).to(self.device)
        elif self.train_config.model_type == "AGNOSTIC_NET":
            print("Model: Agnostic Net")
            print("Agnostic Path: ", self.args.agnostic_path)
            self.model = unet_deep(
                in_channels=in_channel,
                invariant_channel=self.args.agnostic_path,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.train_config.model_type}")

        print('Input channels:', self.total_modalities)
        print("Number Input Channels:", len(self.total_modalities),"\n")
    

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr)
        self.epoched = 0

        if self.train_config.load_pre_trained_model:
            print("Loading Model: ", self.load_model_path)
            checkpoint = torch.load(
                self.load_model_path,
                map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
            )
            self.model.load_state_dict(checkpoint)

        self.loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)

        self.scheduler = None
        if self.lr_sched:
            self.scheduler = utils.lr_schedule(epochs=self.epochs, optimizer=self.optimizer)

    def _initialize_metrics(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.sensitivity_metric = ConfusionMatrixMetric(
            include_background=True,
            metric_name="sensitivity",
            reduction="mean",
            get_not_nans=False,
        )
        self.precision_metric = ConfusionMatrixMetric(
            include_background=True,
            metric_name="precision",
            reduction="mean",
            get_not_nans=False,
        )
        self.IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        self.best_metric = {}
        self.best_metric_epoch = {}
        for dataset in self.datasetlist + self.held_out_datasets:
            self.best_metric[dataset] = -1
            self.best_metric_epoch[dataset] = -1

        self.best_avg_dice = 0

    def _initialize_channel_maps(self):
        self.channel_map = {}
        for dataset in self.datasetlist:
            self.channel_map[dataset] = utils.map_channels(
                self.channels[dataset],
                self.total_modalities,
                rand_assign=self.rand_assign_channels,
            )

        self.combination_map = {}
        for dataset in self.datasetlist:
            if self.agnostic_channel:
                self.combination_map[dataset] = utils.map_combinations(
                    self.channels[dataset], invar_ratio=0.2
                )
            else:
                self.combination_map[dataset] = utils.map_combinations(
                    self.channels[dataset], invar_ratio=0
                )

    def train(self):
        print('\n Total Progress Bar:') 
        for epoch in tqdm(range(self.epoched, self.epochs)):
            print("-" * 10)
            self.model.train()
            epoch_loss = 0
            step = 0

            if (
                not self.lr_sched
                and self.train_config.drop_learning_rate
                and epoch >= self.train_config.drop_learning_rate_epoch
            ):
                for g in self.optimizer.param_groups:
                    g["lr"] = self.train_config.drop_learning_rate_value

            for batch_data in zip(*self.train_loaders):
                step += 1
                epoch_len = max(1, self.data_size // self.train_config.train_batch_size)
                progress_bar = tqdm(
                enumerate(zip(*self.train_loaders), start=1),
                total=epoch_len,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=False,)

                for step,batch_data in progress_bar:


                    modality_outputs = []
                    aux_outputs = []
                    labels = []
                    total_modalities_dropped = []

                    for dataset in self.datasetlist:
                        loader_index = self.data_loader_map[dataset]
                        batch = batch_data[loader_index]

                        if dataset == "BRATS":
                            if self.randomly_drop:
                                (
                                    modalities_dropped,
                                    modalities_remaining,
                                    batch[self.img_index],
                                ) = utils.rand_set_channels_to_zero_with_invar(
                                    self.channels["BRATS"],
                                    batch[self.img_index],
                                    mask_data=batch[self.mask_index],
                                    agnostic_channel=self.agnostic_channel,
                                    batch_label_data=batch[self.label_index],
                                    device_id=self.args.device_id,
                                    agnostic_chan_augs=self.agnostic_chan_augs,
                                    combination_map=self.combination_map["BRATS"],
                                    augmentation_config=self.aug_config,
                                )
                                for i in range(batch[self.label_index].shape[0]):
                                    if self.database_config.BRATS_two_channel_seg:
                                        if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                                            seg_channel = 1
                                        else:
                                            seg_channel = 0
                                        if self.database_config.BRATS_two_channel_seg:
                                            label = batch[self.label_index][
                                                i, [seg_channel], :, :, :
                                            ].to(self.device)
                                    else:
                                        label = batch[self.label_index].to(self.device)
                            else:
                                label = batch[self.label_index].to(self.device)

                            if self.single_slot:
                                input_data, _ = utils.single_slot(batch[self.img_index])
                            else:
                                input_data = torch.from_numpy(
                                    np.zeros(
                                        (
                                            batch[self.img_index].shape[0],
                                            len(self.total_modalities),
                                            self.cropped_input_size[0],
                                            self.cropped_input_size[1],
                                            self.cropped_input_size[2],
                                        ),
                                        dtype=np.float32,
                                    )
                                )

                                if self.rand_assign_channels:
                                    self.channel_map["BRATS"] = utils.rand_assign_channels(
                                        self.channels["BRATS"], self.total_modalities
                                    )

                                input_data[:, self.channel_map["BRATS"], :, :, :] = batch[self.img_index]

                            input_data = input_data.to(self.device)
                            modality_output = self.model(input_data)
                            modality_outputs.append(modality_output)
                            labels.append(label)
                            if self.randomly_drop:
                                total_modalities_dropped.append(modalities_dropped)

                        elif dataset == "TBI":
                            loader_index = self.data_loader_map["TBI"]
                            batch = batch_data[loader_index]
                            TBI_multi_channel_seg = self.database_config.TBI_multichannel

                            if self.single_slot or self.randomly_drop:
                                if self.single_slot:
                                    input_data, modalities_remaining = utils.single_slot(batch[self.img_index])

                                if self.randomly_drop:
                                    (
                                        modalities_dropped,
                                        modalities_remaining,
                                        batch[self.img_index],
                                    ) = utils.rand_set_channels_to_zero_with_invar(
                                        self.channels["TBI"],
                                        batch[self.img_index],
                                        mask_data=batch[self.mask_index],
                                        agnostic_channel=self.agnostic_channel,
                                        batch_label_data=batch[self.label_index],
                                        device_id=self.args.device_id,
                                        agnostic_chan_augs=self.agnostic_chan_augs,
                                        combination_map=self.combination_map["TBI"],
                                        augmentation_config=self.aug_config,
                                    )

                                if (
                                    (0 not in modalities_remaining)
                                    and (2 not in modalities_remaining)
                                    and (3 not in modalities_remaining)
                                ):
                                    seg_channel = 2
                                elif (
                                    (0 not in modalities_remaining)
                                    and (2 not in modalities_remaining)
                                    and (3 in modalities_remaining)
                                ):
                                    seg_channel = 1
                                elif 3 not in modalities_remaining:
                                    seg_channel = 0
                                else:
                                    seg_channel = 2

                                if TBI_multi_channel_seg:
                                    label = batch[self.label_index][:, [seg_channel], :, :, :].to(self.device)
                                else:
                                    label = batch[self.label_index].to(self.device)

                            else:
                                label = batch[self.label_index][:, 0, :, :, :].to(self.device)
                                label = label[:, None, :, :, :]

                            if not self.single_slot:
                                input_data = torch.from_numpy(
                                    np.zeros(
                                        (
                                            batch[self.img_index].shape[0],
                                            len(self.total_modalities),
                                            self.cropped_input_size[0],
                                            self.cropped_input_size[1],
                                            self.cropped_input_size[2],
                                        ),
                                        dtype=np.float32,
                                    )
                                )

                                if self.rand_assign_channels:
                                    self.channel_map["TBI"] = utils.rand_assign_channels(
                                        self.channel_map["TBI"], self.total_modalities
                                    )

                                input_data[:, self.channel_map["TBI"], :, :, :] = batch[self.img_index]

                            input_data = input_data.to(self.device)
                            modality_output = self.model(input_data)
                            modality_outputs.append(modality_output)
                            labels.append(label)
                            if self.randomly_drop:
                                total_modalities_dropped.append(modalities_dropped)

                        else:
                            loader_index = self.data_loader_map[dataset]
                            batch = batch_data[loader_index]

                            if self.single_slot:
                                input_data, _ = utils.single_slot(batch[self.img_index])
                            else:
                                if self.randomly_drop:
                                    (
                                        modalities_dropped,
                                        modalities_remaining,
                                        batch[self.img_index],
                                    ) = utils.rand_set_channels_to_zero_with_invar(
                                        self.channels[dataset],
                                        batch[self.img_index],
                                        mask_data=batch[self.mask_index],
                                        agnostic_channel=self.agnostic_channel,
                                        batch_label_data=batch[self.label_index],
                                        device_id=self.args.device_id,
                                        agnostic_chan_augs=self.agnostic_chan_augs,
                                        combination_map=self.combination_map[dataset],
                                        augmentation_config=self.aug_config,
                                    )

                                input_data = torch.from_numpy(
                                    np.zeros(
                                        (
                                            batch[self.img_index].shape[0],
                                            len(self.total_modalities),
                                            self.cropped_input_size[0],
                                            self.cropped_input_size[1],
                                            self.cropped_input_size[2],
                                        ),
                                        dtype=np.float32,
                                    )
                                )

                                if self.rand_assign_channels:
                                    self.channel_map[dataset] = utils.rand_assign_channels(
                                        self.channel_map[dataset], self.total_modalities
                                    )

                                input_data[:, self.channel_map[dataset], :, :, :] = batch[self.img_index]

                            input_data = input_data.to(self.device)
                            label = batch[self.label_index].to(self.device)
                            combined_outs = self.model(input_data)
                            modality_outputs.append(combined_outs)
                            labels.append(label)
                            if self.randomly_drop:
                                total_modalities_dropped.append(modalities_dropped)

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
                    #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                    progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=self.optimizer.param_groups[0]["lr"],)
                    if self.wandb_active:
                        wandb.log(
                            {"loss": loss.item(), "epoch": epoch + 1, "lr": self.optimizer.param_groups[0]["lr"]}
                        )
            progress_bar.close()
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss /= step
            epoch_loss/= max(step,1)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            print("\n------------------------\n")

            if (epoch + 1) % 50 == 0 and self.model_save_path is not None:
                model_save_name = (
                    self.model_save_path
                    + self.train_config.project_name
                    + "_random_drop_"
                    + str(self.randomly_drop)
                    + "_"
                    + self.date
                    + "_Epoch_"
                    + str(epoch)
                    + ".pth"
                )
                opt_save_name = (
                    self.model_save_path
                    + self.train_config.project_name
                    + "_random_drop_"
                    + str(self.randomly_drop)
                    + "_"
                    + self.date
                    + "_checkpoint_Epoch_"
                    + str(epoch)
                    + ".pt"
                )
                torch.save(self.model.state_dict(), model_save_name)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": epoch_loss,
                    },
                    opt_save_name,
                )
                print("Saved Model")

            if (epoch + 1) % self.train_config.val_interval == 0:
                self._validate(epoch)

        if self.wandb_active:
            wandb.finish()

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            seg_channel = 0
            val_outputs = None
            total_av_dice = list()

            for dataset in [*self.datasetlist, *self.held_out_datasets]:
                metric = {}

                if dataset in self.held_out_datasets:
                    self.channels[dataset] = self.channels_copy[dataset]
                    self.channels[dataset] = [
                        x if x != self.train_config.modality_remove else "invar" for x in self.channels[dataset]
                    ]

                if dataset in self.held_out_datasets:
                    current_loader = self.val_only_loader[dataset]
                elif dataset in self.datasetlist:
                    current_loader = self.val_loader[dataset]
                else:
                    continue

                for val_data in current_loader:
                    self.channel_map[dataset] = utils.map_channels(
                        self.channels[dataset],
                        self.total_modalities,
                        rand_assign=self.rand_assign_channels,
                    )

                    if self.single_slot:
                        input_data, _ = utils.single_slot(val_data[0])
                    else:
                        input_data = torch.from_numpy(
                            np.zeros(
                                (
                                    1,
                                    len(self.total_modalities),
                                    val_data[0].shape[2],
                                    val_data[0].shape[3],
                                    val_data[0].shape[4],
                                ),
                                dtype=np.float32,
                            )
                        )
                        if self.agnostic_channel:
                            if dataset in self.held_out_datasets:
                                input_data[:, self.channel_map[dataset], :, :, :] = val_data[0]
                            elif dataset in self.datasetlist:
                                input_data[:, self.channel_map[dataset][:-1], :, :, :] = val_data[0]
                        else:
                            input_data[:, self.channel_map[dataset], :, :, :] = val_data[0]

                    input_data = input_data.to(self.device)

                    if dataset == "BRATS" and self.database_config.BRATS_two_channel_seg:
                        label = val_data[1][:, [0], :, :, :].to(self.device)
                    elif dataset == "TBI" and self.database_config.TBI_multichannel:
                        label = val_data[1][:, [2], :, :, :].to(self.device)
                    else:
                        label = val_data[1].to(self.device)

                    roi_size = (
                        self.cropped_input_size[0],
                        self.cropped_input_size[1],
                        self.cropped_input_size[2],
                    )
                    sw_batch_size = 1

                    val_outputs = sliding_window_inference(
                        input_data, roi_size, sw_batch_size, self.model
                    )
                    val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                    self.dice_metric(y_pred=val_outputs, y=label)
                    self.sensitivity_metric(y_pred=val_outputs, y=label)
                    self.precision_metric(y_pred=val_outputs, y=label)
                    self.IOU_metric(y_pred=val_outputs, y=label)

                metric[dataset] = {}
                metric[dataset]["dice"] = self.dice_metric.aggregate().item()
                metric[dataset]["sensitivity"] = self.sensitivity_metric.aggregate()[0].item()
                metric[dataset]["precision"] = self.precision_metric.aggregate()[0].item()
                metric[dataset]["IOU"] = self.IOU_metric.aggregate().item()
                self.dice_metric.reset()
                self.sensitivity_metric.reset()
                self.precision_metric.reset()
                self.IOU_metric.reset()

                if metric[dataset]["dice"] > self.best_metric[dataset]:
                    self.best_metric[dataset] = metric[dataset]["dice"]
                    self.best_metric_epoch[dataset] = epoch + 1
                    if epoch > 1 and self.model_save_path is not None:
                        model_save_best_name = (
                            self.model_save_path
                            + self.train_config.project_name
                            + "_random_drop_"
                            + str(self.randomly_drop)
                            + "_"
                            + self.date
                            + "_BEST_"
                            + dataset
                            + ".pth"
                        )
                        torch.save(self.model.state_dict(), model_save_best_name)
                        print(f"Saved new best dice model, for {dataset}:_{self.best_metric[dataset]}")
                        print(self.best_metric[dataset])

                print(
                    "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                        epoch + 1,
                        dataset,
                        metric[dataset]["dice"],
                        dataset,
                        self.best_metric[dataset],
                        self.best_metric_epoch[dataset],
                    )
                )

                total_av_dice.append(metric[dataset]["dice"])

                if self.wandb_active:
                    wandb.log(
                        {
                            "epoch_val": epoch + 1,
                            "mdice_" + dataset: metric[dataset]["dice"],
                            "sensitivity_" + dataset: metric[dataset]["sensitivity"],
                            "precision_" + dataset: metric[dataset]["precision"],
                            "mIOU_" + dataset: metric[dataset]["IOU"],
                        }
                    )

            if len(self.datasetlist) > 1:
                if np.mean(total_av_dice) > self.best_avg_dice:
                    self.best_avg_dice = np.mean(total_av_dice)
                    if self.model_save_path is not None:
                        model_save_best_name = (
                            self.model_save_path
                            + self.train_config.project_name
                            + "_random_drop_"
                            + str(self.randomly_drop)
                            + str(self.args.datasets)
                            + self.date
                            + "_BEST_AVERAGE.pth"
                        )
                        torch.save(self.model.state_dict(), model_save_best_name)
                        print(
                            f"Saved new best average dice model, for {self.datasetlist}:_{self.best_avg_dice}"
                        )
                        print(self.best_avg_dice)


def main(train_config, aug_config, database_config, k_fold, args, channels_copy):
    trainer = ModelTrainer(train_config, aug_config, database_config, k_fold, args, channels_copy)
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument(
        "--datasets", help="datasets for training, using '_' to separate", type=str)
    parser.add_argument("--k_fold", help="k_fold cross validation number fo folds", type=int, default=None)
    parser.add_argument("--agnostic_channel", help="use agnostic channel", type=bool, default=False)
    parser.add_argument("--agnostic_path", help="use agnostic path", type=bool, default=False)
    parser.add_argument("--agnostic_chan_augs", help="use agnostic channel augmentations", type=bool, default=False)


    args = parser.parse_args()
    args.device_id = 1
    args.datasets = "WMH"
    args.agnostic_channel = True 
    args.agnostic_path = True
    args.agnostic_chan_augs = True 

    train_config = config.Training_config()
    database_config = config.Database_config()
    channels_copy = copy.deepcopy(database_config.channels)
    aug_config = config.Augmentation_config()

    main(train_config, aug_config, database_config, k_fold=None, args=args, channels_copy=channels_copy)

