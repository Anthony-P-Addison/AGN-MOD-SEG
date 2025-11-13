import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from nets.multi_unet import res_unet as Unet   
from nets.agnostic_unet import res_unet as agnostic_net
import numpy as np
import utils
import config
import argparse
import tabulate
from dataloader import create_test_val_loader, get_modalities_drop
import copy 
import json 



class ModelTester:
    def __init__(self, args,checkpoint):
        self.args = args
        self.checkpoint = checkpoint
        self.initialize_variables()
        self.setup_modalities()
        self.num_modalities_trained_on = len(self.total_modalities)
        self.initialize_metrics()

    def initialize_variables(self):
        self.datasets_to_test = self.args.datasets_to_test
        self.test_all_combinations = bool(self.args.test_all_combinations)
        self.datasetlist = self.args.trained_on.split("_")
        self.database_config = config.Database_config()
        self.test_config = config.Test_config()
        self.rand_assign = self.test_config.rand_assign
        self.img_index = 0
        self.label_index = 1
        self.total_modalities = set()
        self.data_size = 0
        self.channel_map = {}
        self.val_loaders = []
        self.val_loader = {}
        self.cuda_id = "cuda:" + str(self.args.device_id)
        self.device = torch.device(self.cuda_id)
        self.validate_outputs = {} 
        self.test_config.model_file_path = self.checkpoint
        self.agnostic_path = self.args.agnostic_path
        self.modality_remove = self.args.modality_remove
        self.modality_for_agnostic_channel = self.args.modality_for_agnostic_channel
        self.agnostic_channel = self.args.agnostic_channel
        self.model_name = self.args.model_name
  


    def setup_modalities(self):
        self.channels_copy = copy.deepcopy(self.database_config.channels)
        for data in self.datasetlist:
            if self.agnostic_path or self.agnostic_channel:
                if self.modality_for_agnostic_channel is None:
                    self.database_config.channels[data].append("invar")
                elif self.modality_for_agnostic_channel is not None:
                    self.database_config.channels[data] = ["invar" if modality == self.modality_for_agnostic_channel else modality for modality in self.database_config.channels[data]]
                    
            # if removing modality completely from the dataset during testing
            if  self.modality_remove is not None:
                self.database_config.channels[data] = [modality for modality in self.database_config.channels[data] if modality not in self.modality_remove]

            if self.agnostic_path or self.agnostic_channel:
                self.total_modalities.add("invar")
            
            self.total_modalities = self.total_modalities.union(set(self.database_config.channels[data]))
            self.data_size = max(self.data_size, self.database_config.train_size[data])

        self.total_modalities = sorted(list(self.total_modalities))

        # dataset to test (the above was for dataset trained on
        if self.modality_for_agnostic_channel is not None:
            self.database_config.channels[self.datasets_to_test] = ["invar" if modality == self.modality_for_agnostic_channel else modality for modality in self.database_config.channels[self.datasets_to_test]]     
        if  self.modality_remove is not None:
                self.database_config.channels[self.datasets_to_test] = [modality for modality in self.database_config.channels[self.datasets_to_test] if modality not in self.modality_remove]           


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
        val_size = self.database_config.val_size[dataset]
        images = sorted(glob(os.path.join(self.database_config.img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(self.database_config.seg_path[dataset], "*.*")))
        self.val_loader[dataset] = create_test_val_loader(
            val_size=val_size,
            images=images,
            segs=segs,
            workers=2,
            dataset= dataset,
            modality_remove=self.modality_remove,
            channels = self.channels_copy[dataset],
            image_only=False,
           
        )
        self.val_loaders.append(self.val_loader[dataset])

    def load_model(self):
        if self.test_config.model_net_type == "agnostic_net":
            if self.test_config.single_slot:
                model = agnostic_net(in_channels=1,out_channels=1,invariant_channel=False).to(self.device)
            else:
                model = agnostic_net(in_channels=len(self.total_modalities), out_channels=1,invariant_channel=self.agnostic_path).to(self.device)

        elif self.test_config.model_net_type == "unet_old":
            if self.test_config.single_slot:
                model = Unet(in_channels=1, out_channels=1).to(self.device)
            else:
                model = Unet(in_channels=len(self.total_modalities), out_channels=1).to(self.device)
        print("LOADING CHECKPOINT: ", self.test_config.model_file_path)
        checkpoint = torch.load(self.test_config.model_file_path, map_location={"cuda:0": self.cuda_id, "cuda:1": self.cuda_id})
        model.load_state_dict(checkpoint)
        print(f'Sum of model parameters: {sum(p.numel() for p in model.parameters())}')
        return model
       

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
            self.current_sample_logit_predictions = []
            self.current_sample_label = []

            for val_data in self.val_loader[dataset]:
                if self.test_config.model_net_type == "agnostic_net" or "unet_old":
                    if self.test_config.single_slot:
                        val_data[0] = utils.create_single_channel_UNET_input(
                            val_data,
                            combination,
                            self.args.datasets_to_test,
                            self.num_modalities_trained_on,
                            self.channel_map,
                        )
                    else:
                        val_data[0] = utils.create_UNET_input(
                            val_data,
                            combination,
                            self.args.datasets_to_test,
                            self.num_modalities_trained_on,
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

                roi_size = (self.test_config.croppped_input_size[0], self.test_config.croppped_input_size[1], self.test_config.croppped_input_size[2])
                sw_batch_size = 1

               
                val_output = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                val_outputs = [self.post_trans(i) for i in decollate_batch(val_output)]
                self.current_sample_logit_predictions.append(val_output[0])
                
               

                # add the val output to a dictionary with the combination as the key and the val output as the value.
                
                self.current_sample_label.append(label)
                current_dice = self.dice_metric(y_pred=val_outputs, y=label)
                print(f"Current dice: {current_dice}")
                self.sensitivity_metric(y_pred=val_outputs, y=label)
                self.precision_metric(y_pred=val_outputs, y=label)
                self.IOU_metric(y_pred=val_outputs, y=label)

                

                if self.test_config.save_segs:
                    file_save_path = self.test_config.save_path + str(steps) + "_" + str(current_dice) + ".nii.gz"
                    utils.save_nifti(val_outputs[0], file_save_path, val_data[3]["affine"])

                # Clear intermediate variables to free memory
                del input_data, label, val_outputs
                torch.cuda.empty_cache()
                
                steps += 1

            self.validate_outputs[f'{combination}'] = self.current_sample_logit_predictions

            metric[dataset]["dice"] = self.dice_metric.aggregate().item()
            metric[dataset]["sensitivity"] = self.sensitivity_metric.aggregate()[0].item()
            metric[dataset]["precision"] = self.precision_metric.aggregate()[0].item()
            metric[dataset]["IOU"] = self.IOU_metric.aggregate().item()
            self.dice_metric.reset()
            self.sensitivity_metric.reset()
            self.precision_metric.reset()
            self.IOU_metric.reset()

            print(f'\n  mdice: {np.round(metric[dataset]["dice"], 4)}\n ')
            print(f'\n  sensitivity: {np.round(metric[dataset]["sensitivity"], 4)}\n ')
            print(f'\n  precision: {np.round(metric[dataset]["precision"], 4)}\n ')
            print(f'\n  IOU: {np.round(metric[dataset]["IOU"], 4)}\n ')


            self.mean_dice_comb.append([modality_list, (np.round(metric[dataset]["dice"], 4))])

    def run(self):
        print(f"Random assign: {self.rand_assign}\n Domain agnostic path: {self.agnostic_path}\n modality to be used in agnostic channel: {self.modality_for_agnostic_channel} \n Domain agnostic channel: {self.agnostic_channel}")
        print("Total modalities: ", self.total_modalities)

        dataset = self.args.datasets_to_test
        if self.rand_assign:
            self.channel_map[dataset] = utils.rand_assign_channels(self.database_config.channels[dataset], self.total_modalities)
        else:
            self.channel_map[dataset] = utils.map_channels(self.database_config.channels[dataset], self.total_modalities, rand_assign=self.rand_assign)
        print("channel map:", dataset, self.channel_map[dataset])

        print(f"Testing: {dataset} \n\n")
        self.create_val_loader(dataset)
        torch.cuda.set_device(self.cuda_id)
        model = self.load_model()

        if self.test_all_combinations:
            modalities = utils.create_modality_combinations([int(x) for x in self.args.modalities_to_test.split("_")])
        else:
            modalities = [[int(x) for x in self.args.modalities_to_test.split("_")]]

        for combination in modalities:
            modality_list = [self.database_config.channels[self.datasets_to_test][seg_channel] for seg_channel in combination]
            print(f"Testing on: {'_'.join(modality_list)} {combination}\n\n")
            self.evaluate_model(model, combination, modality_list)


        print(tabulate.tabulate(self.mean_dice_comb, headers=["Combination", "Mean Dice"]))
        print("\n ---------------------------------------------------------------\n")
        print(f" Completed test for {dataset} with modalities: {modality_list}. Using model: {self.model_name}\n\n")
    

    def single_slot_ensemble(self):

        print("Ensembling predictions for single slot model")
        all_predictions = {}
        all_labels = None  # Only store labels once
        
        # Get the number of modalities to test
        modalities = self.args.modalities_to_test.split("_")
        original_modalities = self.args.modalities_to_test  # Store original value

  
        
        for i in [int(x) for x in modalities]:
            # print(f"\nProcessing modality {i+1}/{len(modalities)}")
            self.args.modalities_to_test = str(i)
            
            
            # Create a new tester instance with current args
            tester = ModelTester(self.args, self.checkpoint)
            tester.run()
            x, y = tester.return_dictionary_and_label()
            
            # Initialize the list if this is the first time for this modality
            key = f'[{i}]'
            if key not in all_predictions:
                all_predictions[key] = []
            
            # Move predictions to CPU and store
            cpu_predictions = [pred.cpu() for pred in list(x.values())[0]]
            all_predictions[key].extend(cpu_predictions)
            
            # Only store labels once (they're the same for all modalities)
            if all_labels is None:
                all_labels = [label.cpu() for label in y]
            
            # Clear GPU memory
            del tester, x, y, cpu_predictions
            torch.cuda.empty_cache()
            
            print(f"GPU memory after modality {i}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
        # Restore original modalities value
        self.args.modalities_to_test = original_modalities
        
        print("\nEnsembling predictions...")
        
        # Move predictions back to GPU for ensembling (one at a time)
        device = torch.device(f"cuda:{self.args.device_id}")
        
        # Convert all_predictions to GPU tensors for ensembling
        for key in all_predictions:
            all_predictions[key] = [pred.to(device) for pred in all_predictions[key]]
        
        final_predictions = utils.ensemble_across_modalities(all_predictions)
        
        # Calculate Dice scores
        dice_scores = utils.calculate_dice_scores(final_predictions, all_labels, device)
        
        print(f"\n=== Ensemble Results ===")
        for i, score in enumerate(dice_scores):
            print(f"Patient {i}: Dice = {score:.4f}")
        print(f"Mean Ensemble Dice: {np.mean(dice_scores):.4f}")
        print(f"Std Ensemble Dice: {np.std(dice_scores):.4f}")

        
        # Final cleanup
        del all_predictions, all_labels, final_predictions
        torch.cuda.empty_cache()


    def return_dictionary_and_label(self):
        return self.validate_outputs,self.current_sample_label


#####################################################################




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
    parser.add_argument(
        "--checkpoint", help="The checkpoint to test", type=str, default=None
    )
    parser.add_argument(
        "--model_name", help="The name of the model to test", type=str, default=None
    )
    parser.add_argument(
        "--agnostic_path", help="True if using agnostic path", action='store_true'
    )
    parser.add_argument(
        "--modality_for_agnostic_channel", help="The modality for the agnostic channel", type=str, default=None
    )
    parser.add_argument(
        "--agnostic_channel", help="True if using agnostic channel", action='store_true'
    )
    parser.add_argument("--modality_remove", help="The modality to be removed", type=str, default=None)

    args = parser.parse_args()




    # args.datasets_to_test = 'ISLES'
    # args.modalities_to_test = "0_1_2_3" #ic order of modalities 
    # args.device_id = 1
    # args.trained_on = "TBI_WMH_BRATS_MSSEG_ATLAS" #DATASETS the model was trained on
    # args.checkpoint =  None #'models/WMH_PRELIM_TEST/_model_remove:_None/TBI_WMH_BRATS_MSSEG_ATLAS/2025-06-13_19-14/WMH_PRELIM_TEST_random_drop_True_2025-06-13_19-14_Epoch_599.pth'   # None if using the pre defined checkpoints in checkpoint_paths.json and model name for own models.
    # args.model_name = 'setting_1_agnostic_channel'   # only applicable is checkpoint is None 
    # args.agnostic_path = False
    # args.agnostic_channel = True
    # args.modality_for_agnostic_channel ='DWI'

    

    # Load test checkpoints
    checkpoint = utils.load_test_checkpoints(args.model_name,args.checkpoint)

    
    #Run model: Differs for single slot model as need to ensemble predictions across modalities.
    test_config = config.Test_config()
    if not test_config.single_slot:
        tester = ModelTester(args,checkpoint)
        x=tester.run()
        x,y = tester.return_dictionary_and_label()
    
    if test_config.single_slot:
        tester = ModelTester(args,checkpoint)
        tester.single_slot_ensemble()
            



