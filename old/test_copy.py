import torch
from monai.utils import set_determinism
import utils
from monai.transforms import Compose, EnsureChannelFirst, Activations, AsDiscrete
from glob import glob
import os
from monai.data import ImageDataset, DataLoader, decollate_batch
import config
import argparse
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from old.plots_graphs import bar_plot_1
from nets.unet import Unet


def create_dataset_for_test(dataset: str):
    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])
    Database_config = config.Database_config()
    img_path = Database_config.img_path
    seg_path = Database_config.seg_path
    val_size = Database_config.val_size
    images = sorted(glob(os.path.join(img_path[dataset], "*.*")))
    segs = sorted(glob(os.path.join(seg_path[dataset], "*.*")))
    val_ds = ImageDataset(
        images[-val_size[dataset] :],
        segs[-val_size[dataset] :],
        transform=val_imtrans,
        seg_transform=val_segtrans,
        image_only=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=0)
    return val_loader


def test(
    model,
    val_loader,
    dataset_name,
    modalities,
    model_net_type,
    model_modalities_trained_on,
    model_channel_map,
    device,
    save_outputs,
    save_path,
):
    cropped_input_size = [128, 128, 128]

    # can add other metrics (here only show dice)
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    with torch.no_grad():

        # initialize
        val_images = None
        val_labels = None
        val_outputs = None
        steps = 0
        dice_metric.reset()
        dice_metrics = []
        segment_pixel_vol = []
        gt_pixel_vol = []
        file_dice_dict = {}
        i = 0

        for val_data in val_loader:
            roi_size = (
                cropped_input_size[0],
                cropped_input_size[1],
                cropped_input_size[2],
            )
            sw_batch_size = 1

            # Loop for allocating channel
            channel_map = {}
            
            dataset = args.datasets_to_test

            
        

            # test using sliding window
            if model_net_type == "UNET":
                val_data[0] = utils.create_UNET_input(
                    val_data,
                    modalities,
                    dataset_name,
                    model_modalities_trained_on,
                    model_channel_map,                                 # model_channel_map
                )

            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model
            )
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            # compute metric for the current iteration

            current_dice = dice_metric(y_pred=val_outputs, y=val_labels)

            if save_outputs:
                # save output with the original affine
                file_save_path = (
                    save_path + str(steps) + "_" + str(current_dice) + ".nii.gz"
                )
                utils.save_nifti(val_outputs[0], file_save_path, val_data[3]["affine"])
            pixels_segmented = np.count_nonzero(val_outputs[0])
            gt_segmented = np.count_nonzero(val_labels[0])
            segment_pixel_vol.append(pixels_segmented)
            gt_pixel_vol.append(gt_segmented)
            steps += 1
            
            
            if dataset_name == "VOETS2":
                names = sorted(os.listdir('data/VOETS2/Images'))
                # add combination to list
                key = str(modalities)
                file_dice_dict.setdefault(key,[]).append((names[i],np.round(current_dice.item(),3)))
                dice_metrics.append((names[i],np.round(current_dice.item(),3)))
              
            i += 1

            print("Dice score: ",np.round(current_dice.item(),3))

        metric = dice_metric.aggregate().item()
        print("DICE Metric:")
        print(np.round(metric,3))
        dice_metric.reset()
        #print(dice_metrics)  # dict of filename and dice score
        

        return metric, segment_pixel_vol, gt_pixel_vol,file_dice_dict


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
    args = parser.parse_args()

    dice_combination = []

    for combinations in [0]:
        for dataset in ["VOETS2"]:
            for modality_comb in ["0_1"]:

                args.test_all_combinations = combinations
                args.datasets_to_test = dataset
                args.modalities_to_test = modality_comb

                database_config = config.Database_config()
                channels = database_config.channels[args.datasets_to_test]


                test_all_combinations = bool(combinations)
                cuda_id = "cuda:" + str(args.device_id)
                device = torch.device(cuda_id)
                torch.cuda.set_device(cuda_id)
                results = {}
                Test_config = config.Test_config()

                ####

                Test_config.model_file_path = 'models/new_test_BRATS_ATLAS_WMH_MSSEG_TBI_random_drop_0_Epoch_599.pth' # 'models/Train_BRATS_TBI_ATLAS_MSSEG_WMH.pth'   #models/new_test_MSSEG_random_drop_0_Epoch_599.pth'
                Test_config.model_channel_map = {"VOETS2":[1,5],"BRATS":[1,3,4,5], "ATLAS":[3], "MSSEG":[1,3,4,5,0], "ISLES":[1,3,5,0], "TBI":[1,3,5,2], "WMH":[1,3]}   #{dataset: [0,1,2,3]} 
                Test_config.model_modalities_trained_on = 6

                #####

                print(
                    "*************** TESTING NET "
                    + str(Test_config.model_file_path)
                    + " **************"
                )
                

                model = utils.create_net(Test_config.model_file_path,Test_config.model_net_type,Test_config.model_modalities_trained_on, device, cuda_id)

                print(
                    "************** TESTING DATASET "
                    + args.datasets_to_test
                    + " ***************"
                )


                dataloader = create_dataset_for_test(args.datasets_to_test)
                dice_list = []
                if test_all_combinations:
                    modalities = utils.create_modality_combinations(
                        [int(x) for x in args.modalities_to_test.split("_")]
                    )
                else:
                    modalities = [[int(x) for x in args.modalities_to_test.split("_")]]

                for combination in modalities:
                    print(combination)
                    dsc_scores, seg_pix_vols, gt_pix_vols,file_dice_dict = test(
                        model,
                        dataloader,
                        args.datasets_to_test,
                        combination,
                        Test_config.model_net_type,
                        Test_config.model_modalities_trained_on,
                        Test_config.model_channel_map,
                        device,
                        save_outputs=Test_config.save_segs,
                        save_path=Test_config.save_path,
                    )
                    dice_list.append(np.round(dsc_scores, 3))

                    if dataset == "VOETS2":
                      dice_combination.append(file_dice_dict)

                if dice_combination:
                  # Flatten the list of dictionaries into a single dictionary
                  combined_dict = {}
                  for d in dice_combination:
                      for key, value in d.items():
                          if key not in combined_dict:
                              combined_dict[key] = value
                          else:
                              combined_dict[key].extend(value)

                  # Convert the dictionary to a DataFrame
                  rows = []
                  for key, values in combined_dict.items():
                      for value in values:
                          rows.append([value[0], key, value[1]])

                  df = pd.DataFrame(rows, columns=["File Name", "Modality Combination", "Dice Score"])

                  # Pivot the DataFrame to have file names as rows and modality_comb combinations as columns
                  df_pivot = df.pivot(index="File Name", columns="Modality Combination", values="Dice Score")

                  # Sort the columns to ensure consistent ordering
                  df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

                  print(df_pivot)

                  # Save the DataFrame to a CSV file with table lines
                  df_pivot.to_csv("dice_combination_results.csv")

                  # Create a neatly formatted table with evenly spaced columns using tabulate
                  table = tabulate(df_pivot, headers="keys", tablefmt="grid")



                  # Save the table to a text file
                  with open("dice_combination_results.txt", "w") as f:
                      f.write(table)
            


                # plot of modality_comb combinations against dice 
                if test_all_combinations:
                    name = "Trained_on_all"
                    bar_plot_1(modalities, dice_list,args.datasets_to_test,channels, args.test_all_combinations,name,indiv_mod=modalities,save=True)

                

                 

