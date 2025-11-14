import argparse
import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.losses.dice import DiceCELoss
from nets.multi_unet import res_unet as unet_old
from nets.agnostic_unet import res_unet as agnostic_net
import numpy as np
import utils
from dataloader import get_dataloader
import wandb
import config
import datetime
import copy



class FineTuneTrainer:
    def __init__(self, args, k_fold=None,channels_copy = None):
        self.args = args
        self.k_fold = k_fold

        torch.multiprocessing.set_sharing_strategy("file_system")

        now = datetime.datetime.now()
        self.date = now.strftime("%Y-%m-%d_%H-%M")
        self.modality_remove_validation_set = self.args.modality_remove_validation_set
        self.finetune_dataset = self.args.finetune_dataset.split("_")
        self.train_config = config.Finetune_config()
        self.randomly_drop = bool(self.train_config.randomly_drop)
        self.Database_config = config.Database_config()
        self.channels_copy = channels_copy
        self.checkpoint = self.args.checkpoint
    
        self.add_agnostic_channel_to_pre_trained_model = (
            self.args.add_agnostic_channel_to_pre_trained_model
        )
        self.add_agnostic_path_to_pre_trained_model = (
            self.args.add_agnostic_path_to_pre_trained_model
        )
        self.cropped_input_size = self.train_config.cropped_input_size

        self.datasets_trained_initially = self.args.datasets_trained_initially.split("_")
        self.modality_remove = self.args.modality_remove_training_set
        self.epochs = self.train_config.epoch

        self.new_model_save_path = os.path.join(
            os.path.join(
                self.train_config.model_save_path,
                f"Finetune:_with_{self.args.finetune_dataset}_"
                + f"trained on {self.args.datasets_trained_initially}_"
                + self.date
                + "/",
            )
        )
        if not os.path.exists(self.new_model_save_path):
            os.makedirs(self.new_model_save_path)

        if self.train_config.wandb_report:
            self.run = wandb.init(
                project=self.train_config.project_name,
                name="finetune invar slot with"
                + str(self.args.new_mod_finetune)
                + ":_"
                + str(self.finetune_dataset)
                + "_"
                + "rand_drop_"
                + str(self.train_config.randomly_drop)
                + f"_trained on: {str(self.args.datasets_trained_initially)}"
                + self.date,
            )
        else:
            self.run = None

        if self.k_fold:
            print(f"Training split___: {self.k_fold}")

        self._print_training_settings()


        if self.modality_remove is not None:
            channels = self.Database_config.channels
            # remove from datasets used in pretraining
            for ds in self.datasets_trained_initially:
                if ds in channels:
                    channels[ds] = [m for m in channels[ds] if m != self.modality_remove]
            # if no new modality is being added, also remove from finetune datasets
            if self.args.new_mod_finetune is None:
                for ds in self.finetune_dataset:
                    if ds in channels:
                        channels[ds] = [m for m in channels[ds] if m != self.modality_remove]


        self.img_index = 0
        self.label_index = 1
        self.mask_index = 2

        self.channels = self.Database_config.channels
        self.train_size = self.Database_config.train_size
        self.total_size = self.Database_config.total_size
        self.datasetlist = self.args.finetune_dataset.split("_")
        self.data_size = 0

        self.combination_map = {}
        for dataset in self.datasetlist:
            self.combination_map[dataset] = utils.map_combinations(self.channels[dataset])

        self.total_modalities = []
        self.total_modalities = set(self.total_modalities)
        self.channel_map = {}

        for dataset in self.datasets_trained_initially:
            self.total_modalities = self.total_modalities.union(set(self.channels[dataset]))

        self.total_modalities = sorted(list(self.total_modalities))
    
        for dataset in self.finetune_dataset:
            if self.k_fold is not None:
                self.data_size = len(self.k_fold["train"])
            else:
                self.data_size = max(self.data_size, self.train_size[dataset])

        print("Total modalities: ", self.total_modalities)
        print("Data_size", self.data_size)

        if self.args.pre_trained_agnostic_path or self.args.pre_trained_agnostic_channel or self.add_agnostic_channel_to_pre_trained_model or self.add_agnostic_path_to_pre_trained_model:
            self.total_modalities.append(self.args.new_mod_finetune)
        
        for dataset in self.datasetlist:
            self.channel_map[dataset] = utils.map_channels(
                self.channels[dataset],
                self.total_modalities,
                rand_assign=False,
            )

        self.train_loaders = []
        self.data_loader_map = {}
        self.model_save_path = self.train_config.model_save_path
        self.val_loader = {}

        

        self.train_loaders, self.val_loader, self.data_loader_map = get_dataloader(
            self.train_config,
            self.args.modality_remove_validation_set,    
            self.Database_config,
            [self.datasetlist[-1]],
            self.cropped_input_size,
            self.data_size,
            self.channels_copy,
            k_fold=self.k_fold,
            dataset_use="Train",
        )

        self._initialize_device()
        self._initialize_metrics()

        self.epoched = 0
        self._load_model()

        self.loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)

        self.best_metric = {}
        self.best_metric_epoch = {}
        for dataset in self.datasetlist:
            self.best_metric[dataset] = -1
            self.best_metric_epoch[dataset] = -1

    def _print_training_settings(self):
        print("\n #######  Training Hyperparameters #######")
        print("lr: ", self.train_config.lr)
        print("Batch size: ", self.train_config.train_batch_size)
        print("Randomly drop modalities:", self.randomly_drop)
        print("Running for epochs:", str(self.epochs))


    def _initialize_device(self):
        print("\n #######   Training Settings  #######")
        print(f"The pre trained model trained on datasets:{self.datasets_trained_initially} to be fine tuned on {self.finetune_dataset}")
        if self.add_agnostic_channel_to_pre_trained_model or self.add_agnostic_path_to_pre_trained_model or self.args.pre_trained_agnostic_path or self.args.pre_trained_agnostic_channel:
            print(f"Add new modality {self.args.new_mod_finetune} to the agnostic input channel for dataset {self.finetune_dataset}")
        print("Running on GPU:" + str(self.args.device_id))
        self.cuda_id = "cuda:" + str(self.args.device_id)
        self.device = torch.device(self.cuda_id)
        torch.cuda.set_device(self.cuda_id)

    def _initialize_metrics(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.sensitivity_metric = ConfusionMatrixMetric(
            include_background=True, metric_name="sensitivity", reduction="mean", get_not_nans=False
        )
        self.precision_metric = ConfusionMatrixMetric(
            include_background=True, metric_name="precision", reduction="mean", get_not_nans=False
        )
        self.IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def _load_model(self):
        print('\n #######  Model Information #######')
        if self.train_config.model_type == "AGNOSTIC_NET":
            print("Model: Agnostic Net")
            print("Initial new Agnostic Channel ONLY: ", self.add_agnostic_channel_to_pre_trained_model)
            print("Initial new Agnostic Channel AND Path: ", self.add_agnostic_path_to_pre_trained_model)
            print("Finetune with pre trained Agnostic Channel ONLY: ", self.args.pre_trained_agnostic_channel)
            print("Finetune with pre trained Agnostic Channel AND Path: ", self.args.pre_trained_agnostic_path)
            in_channel = len(self.total_modalities)
            self.args.load_model_finetune_path = utils.load_test_checkpoints(self.args.load_model_finetune_path,self.checkpoint)
            print("Loading Model: ", self.args.load_model_finetune_path)

            if self.add_agnostic_channel_to_pre_trained_model:
                agnostic_path = False
                load = torch.load(
                    self.args.load_model_finetune_path,
                    map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
                )
                checkpoint = utils.add_invar_input_to_pre_trained(load)

            elif self.add_agnostic_path_to_pre_trained_model:
                agnostic_path = True
                load = torch.load(
                    self.args.load_model_finetune_path,
                    map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
                )
                checkpoint = utils.add_invar_layers_to_pre_trained(load)
            
            elif self.args.pre_trained_agnostic_path:
                # load pre-trained model with agnostic path 
                agnostic_path = True
                checkpoint = torch.load(
                    self.args.load_model_finetune_path,
                    map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
                )
            elif self.args.pre_trained_agnostic_channel:
                # load pre-trained model with agnostic channel
                agnostic_path = False
                checkpoint = torch.load(
                    self.args.load_model_finetune_path,
                    map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
                )

            else:
                agnostic_path = False
                checkpoint = torch.load(
                    self.args.load_model_finetune_path,
                    map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id},
                )

            self.model = agnostic_net(
                in_channels=in_channel, invariant_channel=agnostic_path
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr)

            self.model.load_state_dict(checkpoint)

    def train(self):
        for epoch in range(self.epoched, self.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0

            # drop learning rateFoundation models are currently at the forefront of research, and if this work isn't completed soon, other groups may develop brain lesion foundation models first.
            if self.train_config.drop_learning_rate and epoch >= self.train_config.drop_learning_rate_epoch:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.train_config.drop_learning_rate_value

            for batch_data in zip(*self.train_loaders):
                step += 1
                outputs = []
                labels = []
                for dataset in self.finetune_dataset:
                    # Only for BRATS    BRATS may use different ground truth
                    if dataset == "BRATS":
                        loader_index = self.data_loader_map["BRATS"]
                        batch = batch_data[loader_index]

                        if self.randomly_drop:
                            modalities_remaining, batch[self.img_index] = utils.rand_set_channels_to_zero(
                                self.channels["BRATS"], batch[self.img_index]
                            )
                            for i in range(batch[self.label_index].shape[0]):
                                # For BRATS because edema can only be seen on some modalities can use different ground truth for different sets of modalities in input (this need the ground truth file that have multiple channels and for each channel it contain a different gound truth)
                                if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                                    # Edema cannot be seen so change segmentation to labels without edema
                                    seg_channel = 1
                                else:
                                    seg_channel = 0
                                if self.Database_config.BRATS_two_channel_seg:
                                    label[i, :, :, :, :] = batch[self.label_index][i, [seg_channel], :, :, :].to(
                                        self.device
                                    )
                                else:
                                    # default setting of our work: not using different labels
                                    label = batch[self.label_index].to(self.device)
                        else:
                            label = batch[self.label_index].to(self.device)
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

                        input_data[:, self.channel_map["BRATS"], :, :, :] = batch[self.img_index]
                        input_data = input_data.to(self.device)
                        out = self.model(input_data)
                        outputs.append(out)
                        labels.append(label)
                    else:  # other databases are similar
                        loader_index = self.data_loader_map[dataset]
                        batch = batch_data[loader_index]



                        channels_remove_up = self.channels_copy if self.args.modality_remove_validation_set is None else self.channels
                    
                        if self.randomly_drop:
                            all_modalities_dropped, all_modalities_remaining, batch[self.img_index] = (
                                utils.rand_set_channels_to_zero_with_invar(
                                    channels_remove_up[dataset],
                                    batch[self.img_index],
                                    mask_data=batch[self.mask_index],
                                    agnostic_channel=False,
                                    combination_map=self.combination_map[dataset],
                                )
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

                        # ensure that the new modality is in the agnostic input channel for the model (final channel)
                        if self.add_agnostic_channel_to_pre_trained_model or self.add_agnostic_path_to_pre_trained_model or self.args.pre_trained_agnostic_path or self.args.pre_trained_agnostic_channel:
                            try: assert self.total_modalities[-1] == self.args.new_mod_finetune
                            except AssertionError:
                                raise ValueError(f"The new modality {self.args.new_mod_finetune} is not assigned to the last slot for model")
                        

                        input_data[:, self.channel_map[dataset], :, :, :] = batch[self.img_index]
                        input_data = input_data.to(self.device)
                        label = batch[self.label_index].to(self.device)
                        out = self.model(input_data)
                        outputs.append(out)
                        labels.append(label)

                self.optimizer.zero_grad()
                combined_outs = torch.cat(outputs, dim=0)
                combined_labels = torch.cat(labels, dim=0)
                loss = self.loss_function(combined_outs, combined_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = self.data_size // self.train_config.train_batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                if self.train_config.wandb_report:
                    wandb.log({"loss": loss.item(), "epoch": epoch + 1})
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # save model
            if (epoch + 1) % 50 == 0:
                model_save_name = (
                    self.new_model_save_path
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
                    self.new_model_save_path
                    + self.train_config.project_name
                    + "_random_drop_"
                    + str(self.randomly_drop)
                    + "_"
                    + self.date
                    + "_Epoch_"
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
            

            # validation
            if (epoch + 1) % self.train_config.val_interval == 0:
                self._validate(epoch)
        if self.train_config.wandb_report:
            wandb.finish()

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            seg_channel = 0
            val_images = None
            val_labels = None
            val_outputs = None
            metric = {}
            self.dice_metric.reset()
            self.sensitivity_metric.reset()
            self.precision_metric.reset()
            self.IOU_metric.reset()
            for dataset in self.finetune_dataset:
                metric[dataset] = {}
                loader_index = self.data_loader_map[dataset]
                for val_data in self.val_loader[dataset]:
                    # batch = val_data[loader_index]
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
                        
                    input_data[:, self.channel_map[dataset], :, :, :] = val_data[0]
                    input_data = input_data.to(self.device)
                    if dataset == "BRATS" and self.Database_config.BRATS_two_channel_seg:
                        label = val_data[1][:, [0], :, :, :].to(self.device)
                    else:
                        label = val_data[1].to(self.device)
                    roi_size = (
                        self.cropped_input_size[0],
                        self.cropped_input_size[1],
                        self.cropped_input_size[2],
                    )
                    sw_batch_size = 1
                    # using sliding window for the whole 3D image
                    val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, self.model)
                    val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    self.dice_metric(y_pred=val_outputs, y=label)
                    self.sensitivity_metric(y_pred=val_outputs, y=label)
                    self.precision_metric(y_pred=val_outputs, y=label)
                    self.IOU_metric(y_pred=val_outputs, y=label)
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
                    if epoch > 1:
                        model_save_best_name = (
                            self.new_model_save_path
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
                        print("saved new best metric model")
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
                if self.train_config.wandb_report:
                    # wandb log
                    # here only use wandb log to show other metric
                    wandb.log(
                        {
                            "epoch_val": epoch + 1,
                            "mdice_" + dataset: metric[dataset]["dice"],
                            "sensitivity_" + dataset: metric[dataset]["sensitivity"],
                            "precision_" + dataset: metric[dataset]["precision"],
                            "mIOU_" + dataset: metric[dataset]["IOU"],
                        }
                    )


def main(args, k_fold=None,channels_copy = None):
    trainer = FineTuneTrainer(args, k_fold,channels_copy)
    trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    # command line argument
    parser = argparse.ArgumentParser()
    # robust boolean parser
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("yes", "true", "t", "y", "1"):
            return True
        if v in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--load_model_finetune_path", help="The path of the pretrained model", type=str)
    parser.add_argument("--pre_trained_agnostic_path", help="use pre trained agnostic component", type=str2bool, default=False)
    parser.add_argument("--pre_trained_agnostic_channel", help="use pre trained agnostic channel", type=str2bool, default=False)
    parser.add_argument("--datasets_trained_initially",help="modalities used for training the pre-train model using '_' to separate",type=str,)
    parser.add_argument("--add_agnostic_channel_to_pre_trained_model", help="use agnostic channel", type=str2bool, default=True)
    parser.add_argument("--add_agnostic_path_to_pre_trained_model", help="use agnostic path", type=str2bool, default=False)
    parser.add_argument("--new_mod_finetune", help="new modality to be finetuned", type=str, default=None)
    parser.add_argument("--modality_remove_training_set", help="modality to be removed from the training set of pre trained model", type=str, default=None)
    parser.add_argument("--modality_remove_validation_set", help="modality to be removed from the validation set of pre trained model", type=str, default=None)
    parser.add_argument("--finetune_dataset", help="datasets for finetuning, using '_' to separate", type=str)
    parser.add_argument("--checkpoint", help="The checkpoint to test", type=str, default=None)
 
   
  

    args = parser.parse_args()

    selected = [
    args.add_agnostic_path_to_pre_trained_model,
    args.add_agnostic_channel_to_pre_trained_model,
    args.pre_trained_agnostic_path,
    args.pre_trained_agnostic_channel]

    # arguments check 
    selected_count = sum(bool(x) for x in selected)
    if selected_count > 1:
        raise ValueError(
            "Choose at most one of: add_agnostic_path, add_agnostic_channel, "
            "pre_trained_agnostic_channel, pre_trained_agnostic_path, if you want to finetune with no additional modality"
            "then set all to False i.e standard model"
        )
    # If none selected, ensure no new modality is used
    if selected_count == 0:
        args.new_mod_finetune = None
    # If one selected, require new modality name
    else:
        if args.new_mod_finetune is None:
            raise ValueError(
                "new_mod_finetune must be provided when any agnostic option is selected "
                "(add_agnostic_path_to_pre_trained_model, add_agnostic_channel_to_pre_trained_model, "
                "pre_trained_agnostic_path, or pre_trained_agnostic_channel)."
            )

    
    channels_copy = copy.deepcopy(config.Database_config.channels)

    main(args, k_fold=None,channels_copy = channels_copy)


               
