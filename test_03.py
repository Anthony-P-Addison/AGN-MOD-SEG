import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from nets.unet import Unet
import numpy as np
import utils
import config
import argparse
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst




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
    parser.add_argument("--trained_on", help="The datasets the model was trained on", type=str)

    args = parser.parse_args()

    args.datasets_to_test = "MSSEG"
    args.modalities_to_test = "0_1_2_3_4"
    args.test_all_combinations = 0
    args.device_id = 0
    args.trained_on = "MSSEG_BRATS_ATLAS_WMH_TBI"


    test_all_combinations = bool(args.test_all_combinations)


    #load config
    train_config = config.Training_config()
    database_config = config.Database_config()
    test_config = config.Test_config()

    cropped_input_size = [128,128,128]   

    save_outputs = test_config.save_segs
    

    #set index
    img_index = 0
    label_index = 1    

    # Set the data size and total modalities
    channels= database_config.channels     # dictionary of databases and channels as string values
    train_size=database_config.train_size
    total_size=database_config.total_size
    datasetlist=args.trained_on.split("_")    # list of datatbases used to train model
    total_modalities=[]
    total_modalities=set(total_modalities)
    data_size=0
    for dataset in datasetlist:
        total_modalities=total_modalities.union(set(channels[dataset]))
        data_size=max(data_size,train_size[dataset])
    total_modalities = sorted(list(total_modalities))                                             # all the modalities used in training the model as a  set of strings
    
    print("Total modalities: ", total_modalities)
    
    # Loop for allocating channel
    channel_map={}
    for dataset in datasetlist:            
         channel_map[dataset]=utils.map_channels(channels[dataset], total_modalities)
         print("channel map:", dataset,channel_map[dataset]) 


    # path initialization
    train_loaders = []
    val_loaders = []
    val_loader={}
    data_loader_map = {}    
    img_path=database_config.img_path
    seg_path=database_config.seg_path
    load_model_path = test_config.model_file_path       

    
    # # get dataloader
    for dataset in datasetlist:
        print("Testing: ",dataset)
        val_size = total_size[dataset]-train_size[dataset]
        print("Val size: ",val_size)
        images= sorted(glob(os.path.join(img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(seg_path[dataset],"*.*"))) 
        train_loader_one, val_loader[dataset] = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=train_config.workers,train_batch_size=train_config.train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size[dataset],cropped_input_size=cropped_input_size,image_only = False)
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)
        val_loaders.append(val_loader[dataset])


     
    # initialize GPU
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)
    
    # initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean", get_not_nans=False)    
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean", get_not_nans=False)   
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # initialize the model (only show multiunet here)

    if test_config.model_net_type == "UNET":

        model = Unet(in_channels=len(total_modalities)).to(device)

        print("LOADING CHECKPOINT: ", load_model_path)
        checkpoint = torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})
        model.load_state_dict(checkpoint['model_state_dict'])
          
    

    # initialize the best metric
    best_metric={}
    best_metric_epoch={}
    for dataset in datasetlist:
        best_metric[dataset] = -1
        best_metric_epoch[dataset] = -1    

    metric_values = list()



    if test_all_combinations:
                    modalities = utils.create_modality_combinations(
                        [int(x) for x in args.modalities_to_test.split("_")]
                    )
    else:
          modalities = [[int(x) for x in args.modalities_to_test.split("_")]]

    for combination in modalities:
        print("Testing on: ", combination)


      
    
        model.eval()
        with torch.no_grad():
            steps = 0 
            seg_channel = 0
            val_images = None
            val_labels = None
            val_outputs = None
        
            dice_metrics = []
            segment_pixel_vol = []
            gt_pixel_vol = []
            file_dice_dict = {}
            i = 0
            metric={}
            dice_metric.reset()
            sensitivity_metric.reset()
            precision_metric.reset()                
            IOU_metric.reset() 
            dataset = args.datasets_to_test               
            
            metric[dataset]={}
            loader_index = data_loader_map[dataset]
            for val_data in val_loader[dataset]:



                if test_config.model_net_type == "UNET":
                    val_data[0] = utils.create_UNET_input(
                    val_data,
                    combination,
                    args.datasets_to_test,
                    test_config.model_modalities_trained_on,
                    channel_map,
                )

                input_data = val_data[0]
                #batch = val_data[loader_index]         
                
                input_data = input_data.to(device)

                if dataset == "BRATS" and train_config.BRATS_two_channel_seg:
                    label = val_data[1][:,[0],:,:,:].to(device)                      
                else:                        
                    label = val_data[1].to(device)                        
                roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                sw_batch_size = 1

                #using sliding window for the whole 3D image
                val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                # compute metric for current iteration
                current_dice = dice_metric(y_pred=val_outputs, y=label)
                sensitivity_metric(y_pred=val_outputs, y=label)
                precision_metric(y_pred=val_outputs, y=label)                        
                IOU_metric(y_pred=val_outputs, y=label)   

                print('Dice: ',np.round(current_dice,3))

            
                if save_outputs:
                        #save output with the original affine
                    file_save_path = test_config.save_path + str(steps)+'_'+str(current_dice) + ".nii.gz"
                    utils.save_nifti(val_outputs[0], file_save_path, val_data[3]["affine"])
                pixels_segmented = np.count_nonzero(val_outputs[0])
                gt_segmented = np.count_nonzero(label[0])
                segment_pixel_vol.append(pixels_segmented)
                gt_pixel_vol.append(gt_segmented)
                steps+=1


            metric[dataset]["dice"] = dice_metric.aggregate().item()
            metric[dataset]["sensitivity"] = sensitivity_metric.aggregate()[0].item()
            metric[dataset]["precision"] = precision_metric.aggregate()[0].item()                    
            metric[dataset]["IOU"] = IOU_metric.aggregate().item()            
            dice_metric.reset()
            sensitivity_metric.reset()
            precision_metric.reset()                    
            IOU_metric.reset()


            print(f'\n mdice: {np.round(metric[dataset]["dice"],3)}\n ')
            