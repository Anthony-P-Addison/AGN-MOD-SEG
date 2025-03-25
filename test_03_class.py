import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from nets.unet import res_unet as Unet   
#from nets.invariant_channel import CustomUNet as Unet
import numpy as np
import utils
import config
import argparse
import tabulate
from dataloader import create_test_val_loader, get_modalities_drop
import copy 



class ModelTester:
    def __init__(self, args,checkpoint):
        self.args = args
        self.checkpoint = checkpoint
        self.initialize_variables()
        self.setup_modalities()
        self.initialize_metrics()

    def initialize_variables(self):
        self.datasets_to_test = self.args.datasets_to_test
        self.test_all_combinations = bool(self.args.test_all_combinations)
        self.datasetlist = self.args.trained_on.split("_")
        self.database_config = config.Database_config()
        self.test_config = config.Test_config()
        self.rand_assign = self.test_config.rand_assign
        self.cropped_input_size = [128, 128, 128]
        self.img_index = 0
        self.label_index = 1
        self.total_modalities = set()
        self.data_size = 0
        self.channel_map = {}
        self.val_loaders = []
        self.val_loader = {}
        self.cuda_id = "cuda:" + str(self.args.device_id)
        self.device = torch.device(self.cuda_id)

        self.test_config.model_file_path = self.checkpoint

    def setup_modalities(self):
        self.channels_copy = copy.deepcopy(self.database_config.channels)
        for data in self.datasetlist:
            if self.test_config.domain_invariant_slot:
                if self.test_config.modality_rem_train is None:
                    self.database_config.channels[data].append("invar")
                elif self.test_config.modality_rem_train is not None:
                    self.database_config.channels[data] = ["invar" if modality == self.test_config.modality_rem_train else modality for modality in self.database_config.channels[data]]
                    
            # if removing modality completely from the dataset during testing
            if  self.test_config.modality_remove is not None:
                self.database_config.channels[data] = [modality for modality in self.database_config.channels[data] if modality != self.test_config.modality_remove]

            if self.test_config.domain_invariant_slot:
                self.total_modalities.add("invar")
            
            self.total_modalities = self.total_modalities.union(set(self.database_config.channels[data]))
            self.data_size = max(self.data_size, self.database_config.train_size[data])

        self.total_modalities = sorted(list(self.total_modalities))

        # dataset to test (the above was for dataset trained on
        if self.test_config.modality_rem_train is not None:
            self.database_config.channels[self.datasets_to_test] = ["invar" if modality == self.test_config.modality_rem_train else modality for modality in self.database_config.channels[self.datasets_to_test]]     
        if  self.test_config.modality_remove is not None:
                self.database_config.channels[self.datasets_to_test] = [modality for modality in self.database_config.channels[self.datasets_to_test] if modality != self.test_config.modality_remove]           


    def initialize_metrics(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name="sensitivity", reduction="mean", get_not_nans=False)
        self.precision_metric = ConfusionMatrixMetric(include_background=True, metric_name="precision", reduction="mean", get_not_nans=False)
        self.IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.best_metric = {}
        self.best_metric_epoch = {}
        self.metric_values = []
        self.mean_dice_comb = []

        for dataset in self.datasetlist:
            self.best_metric[dataset] = -1
            self.best_metric_epoch[dataset] = -1

    def create_val_loader(self, dataset):
        val_size = self.database_config.total_size[dataset] - self.database_config.train_size[dataset]
        images = sorted(glob(os.path.join(self.database_config.img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(self.database_config.seg_path[dataset], "*.*")))
        self.val_loader[dataset] = create_test_val_loader(
            val_size=val_size,
            images=images,
            segs=segs,
            workers=2,
            dataset= dataset,
            modality_remove=self.test_config.modality_remove,
            channels = self.channels_copy[dataset],
            image_only=False,
           
        )
        self.val_loaders.append(self.val_loader[dataset])

    def load_model(self):
        if self.test_config.model_net_type == "UNET":
            

            if self.test_config.single_slot:
                model = Unet(in_channels=1, out_channels=1).to(self.device)
            else:
                model = Unet(in_channels=len(self.total_modalities), out_channels=1).to(self.device)
            print("LOADING CHECKPOINT: ", self.test_config.model_file_path)
            checkpoint = torch.load(self.test_config.model_file_path, map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id})
            model.load_state_dict(checkpoint)
            print(f'Sum of model parameters: {sum(p.numel() for p in model.parameters())}')
            return model
        return None

    def evaluate_model(self, model, combination, modality_list):
        model.eval()
        with torch.no_grad():
            steps = 0
            dice_metrics = []
            segment_pixel_vol = []
            gt_pixel_vol = []
            metric = {}
            self.dice_metric.reset()
            self.sensitivity_metric.reset()
            self.precision_metric.reset()
            self.IOU_metric.reset()
            dataset = self.args.datasets_to_test
            metric[dataset] = {}

            for val_data in self.val_loader[dataset]:
                if self.test_config.model_net_type == "UNET":
                    if self.test_config.single_slot:
                        val_data[0] = utils.create_single_channel_UNET_input(
                            val_data,
                            combination,
                            self.args.datasets_to_test,
                            self.test_config.num_modalities_trained_on,
                            self.channel_map,
                        )
                    else:
                        val_data[0] = utils.create_UNET_input(
                            val_data,
                            combination,
                            self.args.datasets_to_test,
                            self.test_config.num_modalities_trained_on,
                            self.channel_map,
                        )

                input_data = val_data[0].to(self.device)

                self.database_config.TBI_multichannel = False
                if dataset == "BRATS" and self.database_config.BRATS_two_channel_seg:
                    label = val_data[1][:, [0], :, :, :].to(self.device)
                elif dataset == "TBI" and self.database_config.TBI_multichannel:
                    label = val_data[1][:, [2], :, :, :].to(self.device)
                else:
                    label = val_data[1].to(self.device)

                roi_size = (self.cropped_input_size[0], self.cropped_input_size[1], self.cropped_input_size[2])
                sw_batch_size = 1

                val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]

                current_dice = self.dice_metric(y_pred=val_outputs, y=label)
                self.sensitivity_metric(y_pred=val_outputs, y=label)
                self.precision_metric(y_pred=val_outputs, y=label)
                self.IOU_metric(y_pred=val_outputs, y=label)

                #print("File:", images[-val_size:][steps], "Dice: ", np.round(current_dice, 4))

                if self.test_config.save_segs:
                    file_save_path = self.test_config.save_path + str(steps) + "_" + str(current_dice) + ".nii.gz"
                    utils.save_nifti(val_outputs[0], file_save_path, val_data[3]["affine"])

                pixels_segmented = np.count_nonzero(val_outputs[0])
                gt_segmented = np.count_nonzero(label[0])
                segment_pixel_vol.append(pixels_segmented)
                gt_pixel_vol.append(gt_segmented)
                steps += 1

            metric[dataset]["dice"] = self.dice_metric.aggregate().item()
            metric[dataset]["sensitivity"] = self.sensitivity_metric.aggregate()[0].item()
            metric[dataset]["precision"] = self.precision_metric.aggregate()[0].item()
            metric[dataset]["IOU"] = self.IOU_metric.aggregate().item()
            self.dice_metric.reset()
            self.sensitivity_metric.reset()
            self.precision_metric.reset()
            self.IOU_metric.reset()

            print(f'\n  mdice: {np.round(metric[dataset]["dice"], 4)}\n ')

            self.mean_dice_comb.append([modality_list, (np.round(metric[dataset]["dice"], 4))])

    def run(self):
        print(f"Random assign: {self.rand_assign}\n Domain invariant slot: {self.test_config.domain_invariant_slot}\n modality removed during training: {self.test_config.modality_remove}")
        print("Total modalities: ", self.total_modalities)

        dataset = self.args.datasets_to_test
        self.channel_map[dataset] = utils.map_channels(self.database_config.channels[dataset], self.total_modalities, rand_assign=self.rand_assign)
        print("channel map:", dataset, self.channel_map[dataset])

        print("Testing: ", dataset)
        self.create_val_loader(dataset)
        torch.cuda.set_device(self.cuda_id)
        model = self.load_model()

        if self.test_all_combinations:
            modalities = utils.create_modality_combinations([int(x) for x in self.args.modalities_to_test.split("_")])
        else:
            modalities = [[int(x) for x in self.args.modalities_to_test.split("_")]]

        for combination in modalities:
            modality_list = [self.database_config.channels[self.datasets_to_test][seg_channel] for seg_channel in combination]
            print(f"Testing on: {'_'.join(modality_list)} {combination}")
            self.evaluate_model(model, combination, modality_list)

        print(tabulate.tabulate(self.mean_dice_comb, headers=["Combination", "Mean Dice"]))


 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets_to_test", help="dataset for testing", type=str)
    parser.add_argument(
        "--modalities_to_test",
        help="The modalities for testing (the index of the modalities for that input),using '_' to separate if 0_1_2 for BRATS it would mean test on FLAIR, T1, T1c",
        type=str,
    )
    parser.add_argument(
        "--test_all_combinations",
        help="0 or 1 1 if testing on all possible modality_comb combinations",
        type=int,
        default="0",
    )
    parser.add_argument(
        "--trained_on", help="The datasets the model was trained on", type=str
    )

    args = parser.parse_args()

    ####################

    args.datasets_to_test = 'WMH' #'TBI' # dataset for testing
    args.modalities_to_test ="0_1"       # numeric order of modalities
    args.test_all_combinations = 1
    args.device_id = 0
    args.trained_on = 'WMH_MSSEG_BRATS_ATLAS_TBI'    # The datasets the model was trained on
    #########################

    checkpoint1 = ['models/all_in_one/WMH_MSSEG_BRATS_ATLAS_TBI/all_in_one_random_drop_1_2024-12-19_17-40_Epoch_599.pth']
    #['models/all_in_one/_model_remove:_None/TBI/2025-02-13_21-31/all_in_one_random_drop_False_2025-02-13_21-31_Epoch_599.pth']



    for file in checkpoint1:

        tester = ModelTester(args,file)
        x=tester.run()
    


    
# msse - 0.5438 after 240 epochs 


