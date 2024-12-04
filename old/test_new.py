import torch
import utils
import config
from glob import glob
import os
from monai.inferers import sliding_window_inference
from monai.data import ImageDataset, DataLoader, decollate_batch
from monai.transforms import Compose, EnsureChannelFirst, Activations, AsDiscrete
from monai.metrics import DiceMetric
from nets.unet import Unet as res_unet
import numpy as np
import argparse
from nets.unet import Unet

# checkpoint = torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})
#             model.load_state_dict(checkpoint['model_state_dict'])



def create_dataset_for_test(datasetlist, Database_config, Training_config,Test_config):

    val_loaders = []
    train_loaders = []
    val_loader ={}
    train_loaders = []
    data_loader_map = {}
    data_size = 0
    total_modalities = []
    total_modalities=set(total_modalities)
    img_path = Database_config.img_path
    seg_path = Database_config.seg_path
    val_size = Database_config.val_size
    total_size = Database_config.total_size
    train_size = Database_config.train_size
    cropped_input_size = Training_config.cropped_input_size
    
    for dataset in datasetlist:
      
        total_modalities=total_modalities.union(set(Test_config.model_channel_map[dataset])) # multipe dataset only want to return modalities present
        data_size=max(data_size,train_size[dataset])  
      
        print("training: ",dataset)
        val_size = total_size[dataset]-train_size[dataset]
        images= sorted(glob(os.path.join(img_path[dataset], "*.*")))    # mathcing any file with any extension at specified path 
        segs = sorted(glob(os.path.join(seg_path[dataset],"*.*"))) 
        train_loader_one, val_loader[dataset] = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=train_config.workers,train_batch_size=train_config.train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size[dataset],cropped_input_size=cropped_input_size)
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)
        val_loaders.append(val_loader)

    return val_loader   # returns a dict of dataloaders




def test_model(model, val_loader, datasetlist, device, post_trans, dice_metric, total_modalities, channel_map, cropped_input_size):
    model.eval()
    with torch.no_grad():

        dice_metric.reset()
        for dataset in datasetlist:

            print(f"Testing on dataset: {dataset}")
            for val_data in val_loader[dataset]:   # all good
                input_data = torch.from_numpy(np.zeros((1, (total_modalities), val_data[0].shape[2], val_data[0].shape[3], val_data[0].shape[4]), dtype=np.float32))
                input_data[:, channel_map[dataset], :, :, :] = val_data[0]
                input_data = input_data.to(device)
                label = val_data[1].to(device)
                roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                sw_batch_size = 1
                val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=label)
            dice_score = dice_metric.aggregate().item()
            print(f"Dice score for {dataset}: {dice_score}")
            dice_metric.reset()







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets", help="datasets for testing, using '_' to separate", type=str)
    parser.add_argument("--load_model_path", help="Path to the pre-trained model checkpoint", type=str)
    parser.add_argument("--test_all_combinations", help="Test all combinations of modalities", type=int)

    args = parser.parse_args()

    args.device_id = 0
    args.datasets= ["MSSEG"]
    args.load_model_path ="models/new_test_MSSEG_random_drop_0_BEST_MSSEG.pth"
    args.modalities_to_test = "0_1_2_3_4"


    # Load config
    database_config = config.Database_config()
    train_config = config.Training_config()
    test_config = config.Test_config()

    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)

    test_config.model_file_path = args.load_model_path

    val_loader = create_dataset_for_test(args.datasets, database_config, train_config,test_config)
    cropped_input_size = [128, 128, 128]

    test_config.model_channel_map = {'MSSEG': [0,1,2,3,4]}
    test_config.model_modalities_trained_on = 5
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans= Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

   




   

   
    #model = Unet(in_channels=5, out_channels=1).to(device)
    #model = torch.load("models/tesT_MSSEG_NO_DROP_checkpoint_Epoch_449.pt",map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})
  
    #model.load_state_dict(model_state_dict['optimiser_state_dict'])
    # checkpoint  = torch.load("models/new_test_MSSEG_random_drop_0_checkpoint_Epoch_599.pt",map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})

   
    # model.load_state_dict(checkpoint['model_state_dict'])



 

    




    model = utils.create_net(test_config.model_file_path,test_config.model_net_type,test_config.model_modalities_trained_on, device, cuda_id)


    test_model(model, val_loader, args.datasets, device, post_trans, dice_metric, test_config.model_modalities_trained_on,test_config.model_channel_map, cropped_input_size)
   

  

  