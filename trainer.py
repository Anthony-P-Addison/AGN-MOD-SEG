import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.losses import DiceCELoss
import numpy as np
import utils
import wandb
import datetime
from dataloader import get_dataloader
import copy
from tqdm import tqdm
from nets.multi_unet import res_unet as unet_old
from nets.agnostic_unet import res_unet as unet_deep
import os

class Trainer:
    def __init__(self, train_config, aug_config, database_config, args):
        self.train_config = train_config
        self.aug_config = aug_config
        self.database_config = database_config
        self.args = args
        
        # Set data indices
        self.img_index = 0
        self.label_index = 1
        self.mask_index = 2
        
        # Initialize components
        self.device = self._setup_device()
        self._load_config()
        self._initialize_metrics()
        self._setup_data_handling()
        self._setup_sliding_window()
        
    def _setup_device(self):
        """Setup CUDA device"""
        cuda_id = f"cuda:{self.args.device_id}"
        device = torch.device(cuda_id)
        torch.cuda.set_device(cuda_id)
        torch.multiprocessing.set_sharing_strategy("file_system")
        return device

    def _load_config(self):
        """Load configuration parameters"""
        self.rand_assign_channels = self.train_config.rand_assign_channels
        self.domain_invariant_slot = self.train_config.domain_invariant_slot
        self.load_model_path = self.train_config.load_model_path
        self.modality_remove = self.train_config.modality_remove
        self.randomly_drop = bool(self.train_config.random_drop)
        self.single_slot = self.train_config.single_slot
        self.wandb_active = self.train_config.wandb_active
        self.contrast_augmentation = self.train_config.contrast_augmentation
        self.lr_sched = self.train_config.lr_sched
        self.held_out_datasets = self.train_config.held_out_datasets
        self.dropped_modality = self.train_config.modality_remove
        self.cropped_input_size = self.train_config.cropped_input_size
        self.epochs = self.train_config.epoch

        if self.randomly_drop and self.single_slot:
            raise ValueError("Cannot have both random drop and single slot")

    def _setup_data_handling(self):
        """Setup dataset size and modality handling"""
        # Handle modality removal
        self.channels = self.database_config.channels
        if self.modality_remove is not None:
            for key, value in self.channels.items():
                self.channels[key] = [x for x in value if x != self.modality_remove]
            print(f"Removed {str(self.modality_remove)} from datasets")

        # Set the data size and total modalities
        self.train_size = self.database_config.train_size
        self.datasetlist = self.args.datasets.split("_")
        self.total_modalities = set()
        self.data_size = 0

        # Calculate data size and handle domain invariant slot
        for dataset in self.datasetlist:
            if self.domain_invariant_slot:
                self.channels[dataset].append("invar")

            if self.args.k_fold is not None:
                self.data_size = len(self.args.k_fold['train'])
            else:
                self.data_size = max(self.data_size, self.train_size[dataset])

            self.total_modalities = self.total_modalities.union(set(self.channels[dataset]))

        self.total_modalities = sorted(list(self.total_modalities))
        print("Data_size", self.data_size)

        # Make a copy of channels for validation
        self.channels_copy = copy.deepcopy(self.database_config.channels)

    def _setup_sliding_window(self):
        """Setup sliding window inference parameters"""
        self.roi_size = (
            self.cropped_input_size[0],
            self.cropped_input_size[1],
            self.cropped_input_size[2],
        )
        self.sw_batch_size = 1

    def _initialize_metrics(self):
        """Initialize evaluation metrics"""
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

    def _setup_model(self):
        """Setup the model architecture"""
        in_channel = 1 if self.single_slot else len(self.total_modalities)
        
        if self.train_config.model_type == "old_unet":
            print("TRAINING WITH old_unet")
            model = unet_old(in_channels=in_channel).to(self.device)
        elif self.train_config.model_type == "deep_unet":
            print("TRAINING WITH DEEP UNET")
            model = unet_deep(in_channels=in_channel, invariant_channel=self.train_config.domain_invariant_layers).to(self.device)
        
        if self.train_config.load_pre_trained_model:
            print("LOADING MODEL: ", self.load_model_path)
            checkpoint = torch.load(
                self.load_model_path, 
                map_location={"cuda:0": f"cuda:{self.args.device_id}", 
                            "cuda:1": f"cuda:{self.args.device_id}"}
            )
            model.load_state_dict(checkpoint)
            
        return model

    def _setup_wandb(self, date):
        """Setup Weights & Biases logging"""
        if self.wandb_active:
            wandb.init(
                project=self.train_config.project_name,
                name=(
                    self.train_config.project_name 
                    + self.args.datasets
                    + "_random_drop_"
                    + str(self.randomly_drop)
                    + "_"
                    + 'modality_remove:_'
                    + str(self.modality_remove)
                    + date
                )
            )

    def _create_save_paths(self):
        """Create paths for saving models"""
        if self.wandb_active:
            now = datetime.datetime.now()
            date = now.strftime("%Y-%m-%d_%H-%M")
            model_save_path = os.path.join(
                self.train_config.model_save_path, 
                f"{self.args.datasets}/{date}/"
            )
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            return model_save_path, date
        return None, None

    def train(self, k_fold=None):
        """Main training loop"""
        model_save_path, date = self._create_save_paths()
        
        # Get dataloaders
        train_loaders, val_loader, data_loader_map = get_dataloader(
            self.train_config, 
            self.database_config,
            self.args.datasets.split("_"), 
            self.cropped_input_size, 
            self.data_size,  # data_size will be determined in the function
            self.channels_copy,
            k_fold,
            dataset_use="Train"
        )

        # Setup validation data for held out datasets
        if True:  # validate_data_only - always True in original code
            original_modality_remove = self.train_config.modality_remove
            self.train_config.modality_remove = None
            WMH_loader, val_only_loader, _ = get_dataloader(
                self.train_config, 
                self.database_config,
                self.held_out_datasets,
                self.cropped_input_size, 
                self.data_size,
                self.channels_copy,
                k_fold,
                dataset_use="Val ONLY"
            )
            self.train_config.modality_remove = original_modality_remove

        # Setup channels and model
        channels = self.database_config.channels
        datasetlist = self.args.datasets.split("_")
        total_modalities = sorted(list(set().union(*[set(channels[dataset]) for dataset in datasetlist])))
        
        model = self._setup_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config.lr)
        loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
        
        if self.lr_sched:
            scheduler = utils.lr_schedule(epochs=self.epochs, optimizer=optimizer)

        # Initialize tracking metrics
        best_metric = {dataset: -1 for dataset in datasetlist + self.held_out_datasets}
        best_metric_epoch = {dataset: -1 for dataset in datasetlist + self.held_out_datasets}
        best_avg_dice = 0

        # Setup channel mappings
        channel_map = {}
        combination_map = {}
        for dataset in datasetlist:
            channel_map[dataset] = utils.map_channels(
                channels[dataset],
                total_modalities,
                rand_assign=self.rand_assign_channels,
            )
            combination_map[dataset] = utils.map_combinations(
                channels[dataset],
                invar_ratio=0.2 if self.domain_invariant_slot else 0
            )

        # Training loop
        for epoch in tqdm(range(self.epochs)):
            model.train()
            epoch_loss = self._train_epoch(
                model, 
                train_loaders, 
                optimizer, 
                loss_function, 
                datasetlist, 
                data_loader_map, 
                channel_map, 
                combination_map,
                scheduler if self.lr_sched else None
            )

            # Save model checkpoint
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(model, optimizer, epoch, epoch_loss, model_save_path, date)

            # Validation
            if (epoch + 1) % self.train_config.val_interval == 0:
                metrics = self._validate(
                    model, 
                    datasetlist, 
                    self.held_out_datasets,
                    val_loader, 
                    val_only_loader, 
                    channels, 
                    self.channels_copy,
                    total_modalities,
                    channel_map
                )
                
                # Update best metrics
                for dataset, metric_value in metrics.items():
                    if metric_value["dice"] > best_metric[dataset]:
                        best_metric[dataset] = metric_value["dice"]
                        best_metric_epoch[dataset] = epoch + 1
                        if epoch > 1:
                            self._save_best_model(model, dataset, model_save_path, date)

                # Handle average dice across datasets
                if len(datasetlist) > 1:
                    avg_dice = np.mean([metrics[dataset]["dice"] for dataset in datasetlist])
                    if avg_dice > best_avg_dice:
                        best_avg_dice = avg_dice
                        self._save_best_average_model(model, datasetlist, model_save_path, date, best_avg_dice)

        if self.wandb_active:
            wandb.finish()

    def _train_epoch(self, model, train_loaders, optimizer, loss_function, datasetlist, 
                    data_loader_map, channel_map, combination_map, scheduler=None):
        """Train for one epoch"""
        epoch_loss = 0
        step = 0
        
        for batch_data in zip(*train_loaders):
            step += 1
            loss = self._train_step(
                model, 
                batch_data, 
                datasetlist, 
                data_loader_map, 
                channel_map, 
                combination_map,
                loss_function
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if scheduler:
            scheduler.step()
            
        return epoch_loss / step

    def _train_step(self, model, batch_data, datasetlist, data_loader_map, 
                   channel_map, combination_map, loss_function):
        """Perform a single training step"""
        modality_outputs = []
        labels = []
        
        for dataset in datasetlist:
            loader_index = data_loader_map[dataset]
            batch = batch_data[loader_index]
            
            input_data, label = self._prepare_batch(
                batch, 
                dataset, 
                channel_map, 
                combination_map
            )
            
            output = model(input_data)
            modality_outputs.append(output)
            labels.append(label)
        
        combined_outputs = torch.cat(modality_outputs, dim=0)
        combined_labels = torch.cat(labels, dim=0)
        
        return loss_function(combined_outputs, combined_labels)

    def _validate(self, model, datasetlist, held_out_datasets, val_loader, val_only_loader, 
                 channels, channels_copy, total_modalities, channel_map):
        """Perform validation"""
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            for dataset in [*datasetlist, *held_out_datasets]:
                metrics[dataset] = self._validate_dataset(
                    model, 
                    dataset, 
                    val_loader if dataset in datasetlist else val_only_loader[dataset],
                    channels, 
                    channels_copy,
                    total_modalities,
                    channel_map
                )
                
        return metrics

    def _save_checkpoint(self, model, optimizer, epoch, loss, save_path, date):
        """Save model checkpoint"""
        if not self.wandb_active:
            return
            
        model_save_name = (
            f"{save_path}{self.train_config.project_name}"
            f"_random_drop_{self.randomly_drop}_{date}"
            f"_Epoch_{epoch}.pth"
        )
        
        opt_save_name = (
            f"{save_path}{self.train_config.project_name}"
            f"_random_drop_{self.randomly_drop}_{date}"
            f"_checkpoint_Epoch_{epoch}.pt"
        )
        
        torch.save(model.state_dict(), model_save_name)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, opt_save_name)
        
        print("Saved Model")

    def _save_best_model(self, model, dataset, save_path, date):
        """Save best model for a specific dataset"""
        if not self.wandb_active:
            return
            
        model_save_name = (
            f"{save_path}{self.train_config.project_name}"
            f"_random_drop_{self.randomly_drop}_{date}"
            f"_BEST_{dataset}.pth"
        )
        torch.save(model.state_dict(), model_save_name)

    def _save_best_average_model(self, model, datasetlist, save_path, date, best_avg_dice):
        """Save model with best average performance"""
        if not self.wandb_active:
            return
            
        model_save_name = (
            f"{save_path}{self.train_config.project_name}"
            f"_random_drop_{self.randomly_drop}{str(datasetlist)}{date}"
            f"_BEST_AVERAGE.pth"
        )
        torch.save(model.state_dict(), model_save_name)
        print(f"Saved new best average dice model, for {datasetlist}:{best_avg_dice}")

    def _prepare_batch(self, batch, dataset, channel_map, combination_map):
        """Prepare a batch of data for training/validation
        
        Args:
            batch: The input batch data
            dataset: Current dataset name
            channel_map: Mapping of channels
            combination_map: Mapping of modality combinations
        """
        if dataset == "BRATS":
            return self._prepare_brats_batch(batch, channel_map, combination_map)
        elif dataset == "TBI":
            return self._prepare_tbi_batch(batch, channel_map, combination_map)
        else:
            return self._prepare_standard_batch(batch, dataset, channel_map, combination_map)

    def _prepare_brats_batch(self, batch, channel_map, combination_map):
        """Prepare BRATS specific batch"""
        if self.randomly_drop:
            modalities_dropped, modalities_remaining, batch[self.img_index] = (
                utils.rand_set_channels_to_zero_with_invar(
                    self.channels["BRATS"], 
                    batch[self.img_index],
                    mask_data=batch[self.mask_index],
                    domain_invariant=self.domain_invariant_slot, 
                    batch_label_data=batch[self.label_index],
                    device_id=self.args.device_id,
                    contrast_augmentation=self.contrast_augmentation,
                    combination_map=combination_map["BRATS"],
                    augmentation_config=self.aug_config
                )
            )
            
            # Handle BRATS specific label selection
            if self.database_config.BRATS_two_channel_seg:
                label = self._select_brats_label(batch, modalities_remaining)
            else:
                label = batch[self.label_index].to(self.device)
        else:
            label = batch[self.label_index].to(self.device)

        input_data = self._prepare_input_data(batch, "BRATS", channel_map)
        return input_data, label

    def _prepare_tbi_batch(self, batch, channel_map, combination_map):
        """Prepare TBI specific batch"""
        TBI_multi_channel_seg = self.database_config.TBI_multichannel
        modalities_remaining = None

        if self.single_slot or self.randomly_drop:
            if self.single_slot:
                input_data, modalities_remaining = utils.single_slot(batch[self.img_index])

            if self.randomly_drop:
                modalities_dropped, modalities_remaining, batch[self.img_index] = (
                    utils.rand_set_channels_to_zero_with_invar(
                        self.channels["TBI"], 
                        batch[self.img_index],
                        mask_data=batch[self.mask_index],
                        domain_invariant=self.domain_invariant_slot, 
                        batch_label_data=batch[self.label_index],
                        device_id=self.args.device_id,
                        contrast_augmentation=self.contrast_augmentation,
                        combination_map=combination_map["TBI"],
                        augmentation_config=self.aug_config
                    )
                )

            # Select appropriate label based on available modalities
            label = self._select_tbi_label(batch, modalities_remaining, TBI_multi_channel_seg)
        else:
            label = batch[self.label_index][:, 0, :, :, :].to(self.device)
            label = label[:, None, :, :, :]

        if not self.single_slot:
            input_data = self._prepare_input_data(batch, "TBI", channel_map)

        input_data = input_data.to(self.device)
        return input_data, label

    def _prepare_standard_batch(self, batch, dataset, channel_map, combination_map):
        """Prepare batch for standard datasets"""
        if self.single_slot:
            input_data, _ = utils.single_slot(batch[self.img_index])
        else:
            if self.randomly_drop:
                modalities_dropped, modalities_remaining, batch[self.img_index] = utils.rand_set_channels_to_zero_with_invar(
                    self.channels[dataset], 
                    batch[self.img_index],
                    mask_data=batch[self.mask_index],
                    domain_invariant=self.domain_invariant_slot, 
                    batch_label_data=batch[self.label_index],
                    device_id=self.args.device_id,
                    contrast_augmentation=self.contrast_augmentation,
                    combination_map=combination_map[dataset],
                    augmentation_config=self.aug_config
                )

            input_data = self._prepare_input_data(batch, dataset, channel_map)

        input_data = input_data.to(self.device)
        label = batch[self.label_index].to(self.device)
        return input_data, label

    def _prepare_input_data(self, batch, dataset, channel_map):
        """Prepare input data tensor"""
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
            channel_map[dataset] = utils.rand_assign_channels(
                self.channels[dataset], 
                self.total_modalities
            )

        input_data[:, channel_map[dataset], :, :, :] = batch[self.img_index]
        return input_data

    def _select_brats_label(self, batch, modalities_remaining):
        """Select appropriate label for BRATS data"""
        label = batch[self.label_index].clone()
        for i in range(batch[self.label_index].shape[0]):
            if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                seg_channel = 1  # Edema cannot be seen
            else:
                seg_channel = 0
            if self.database_config.BRATS_two_channel_seg:
                label[i, :, :, :, :] = batch[self.label_index][i, [seg_channel], :, :, :].to(self.device)
        return label

    def _select_tbi_label(self, batch, modalities_remaining, TBI_multi_channel_seg):
        """Select appropriate label for TBI data"""
        if ((0 not in modalities_remaining) and 
            (2 not in modalities_remaining) and 
            (3 not in modalities_remaining)):
            seg_channel = 2
        elif ((0 not in modalities_remaining) and 
              (2 not in modalities_remaining) and 
              (3 in modalities_remaining)):
            seg_channel = 1
        elif 3 not in modalities_remaining:
            seg_channel = 0
        else:
            seg_channel = 2

        if TBI_multi_channel_seg:
            return batch[self.label_index][:, [seg_channel], :, :, :].to(self.device)
        else:
            return batch[self.label_index].to(self.device)


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
    trainer = Trainer(train_config, aug_config, database_config, args)
    trainer.train(k_fold=args.k_fold)