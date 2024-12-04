import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from nets.unet import Unet
import numpy as np
import utils
import wandb
import config
import argparse
import datetime



def train(train_config,database_config,args):
    """train function"""

    
    
    torch.multiprocessing.set_sharing_strategy('file_system') 

    cropped_input_size = train_config.cropped_input_size
    epochs = train_config.epoch

    # set the project in wand b.
    #TODO: add project name into config
    save_name = train_config.save_name
    run = wandb.init(
    project="all_in_one",
    name=save_name,
    )  

    #print training settings
    print("lr: ",train_config.lr)
    print("Workers: ", train_config.workers)
    print("Batch size: ",train_config.train_batch_size)
    print("RANDOMLY DROP? ",rand_drop)

    #set index
    img_index = 0
    label_index = 1    

    # Set the data size and total modalities
    channels= database_config.channels    # modality channels dicitionary with datasset as key and list of modalities as values
    train_size=database_config.train_size
    total_size=database_config.total_size   
    datasetlist=args.datasets.split("_")     # dataset list input as command line argument  e.g voets_ATLAS_ etc split into list 
    total_modalities=[]
    total_modalities=set(total_modalities)
    data_size=0

    for dataset in datasetlist:    # iterate through each dataset 
        total_modalities=total_modalities.union(set(channels[dataset])) # multipe dataset only want to return modalities present
        data_size=max(data_size,train_size[dataset])                # whcih dataset has the max size
    total_modalities = sorted(list(total_modalities))    # sort into numerical order
    print("data_size",data_size)
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

    # save model path / create directory
    model_save_path = os.path.join(train_config.model_save_path,args.datasets)
    if not os.path.exists(os.path.join(train_config.model_save_path,args.datasets)):
        os.makedirs(model_save_path)
    
    
    load_model_path = train_config.load_model_path         

    # get dataloader
    for dataset in datasetlist:
        print("training: ",dataset)
        val_size = total_size[dataset]-train_size[dataset]
        images= sorted(glob(os.path.join(img_path[dataset], "*.*")))    # mathcing any file with any extension at speciofifed path 
        segs = sorted(glob(os.path.join(seg_path[dataset],"*.*"))) 
        train_loader_one, val_loader[dataset] = utils.create_dataloader(val_size=val_size, images=images,segs=segs, workers=train_config.workers,train_batch_size=train_config.train_batch_size,total_train_data_size=data_size,current_train_data_size=train_size[dataset],cropped_input_size=cropped_input_size)
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)
        val_loaders.append(val_loader[dataset])
        
    # initialize GPU
    print("Running on GPU:" + str(args.device_id))
    print("Running for epochs:" + str(epochs))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    # initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    sensitivity_metric = ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean", get_not_nans=False)    
    precision_metric = ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean", get_not_nans=False)   
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  

    # initialize the model (only show multiunet here)
    print("in_channels= ",len(total_modalities))
    print("batch size = ",train_config.train_batch_size)
    if train_config.model_type == "UNET":
        print("TRAINING WITH UNET")
        model = Unet(in_channels=len(total_modalities)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
        epoched=0   # initial epoch, zero if starting from scrath
        # load pre-trained weights
        if train_config.load_pre_trained_model:
            print("LOADING MODEL: ", load_model_path)
            checkpoint = torch.load(load_model_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id})
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoched=checkpoint['epoch']+1     # initial epoch, value if loading pre trained model


    # defined loss function
    loss_function = DiceLoss(sigmoid=True)    

    # initialize the best metric
    best_metric={}
    best_metric_epoch={}
    for dataset in datasetlist:
        best_metric[dataset] = -1
        best_metric_epoch[dataset] = -1    

    metric_values = list()  # what is this for ?

    


    #training
    for epoch in range(epoched,epochs):     
       
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # drop learning rate 
        if train_config.drop_learning_rate and epoch >= train_config.drop_learning_rate_epoch:       # train_config.drop_learning_rate - bool
            for g in optimizer.param_groups:
                g['lr'] = train_config.drop_learning_rate_value

        for batch_data in zip(*train_loaders):         # train loaders of multiple dataset beiing zipped together and get batch. TODO: read a little more into this.  
            step += 1
            outputs = []
            labels = []
            for dataset in datasetlist:
                #Only for BRATS    BRATS may use different ground truth

                # TODO: want to put this into a different script - does not need to be in training script
                if dataset == "BRATS":
                    loader_index = data_loader_map["BRATS"]
                    batch = batch_data[loader_index]                
                    if rand_drop:
                        modalities_remaining, batch[img_index] = utils.rand_set_channels_to_zero(channels["BRATS"], batch[img_index])
                        for i in range (batch[label_index].shape[0]):
                        # For BRATS because edema can only be seen on some modalities can use different ground truth for different sets of modalities in input (this need the ground truth file that have multiple channels and for each channel it contain a different gound truth)
                            if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                                # Edema cannot be seen so change segmentation to labels without edema
                                seg_channel = 1
                            else:
                                seg_channel = 0
                            if train_config.BRATS_two_channel_seg:                            
                                label[i,:,:,:,:] = batch[label_index][i,[seg_channel],:,:,:].to(device)                                
                            else:
                                #default setting of our work: not using different labels
                                label = batch[label_index].to(device)
                    else:                        
                        label = batch[label_index].to(device)                         
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,channel_map["BRATS"],:,:,:] = batch[img_index]
                    input_data = input_data.to(device)        
                    out = model(input_data)
                    outputs.append(out)
                    labels.append(label)

                elif dataset == "TBI":
                    loader_index = data_loader_map["TBI"]
                    batch = batch_data[loader_index]
                    TBI_multi_channel_seg = False

                    if rand_drop:
                        modalities_remaining, batch[img_index] = (
                            utils.rand_set_channels_to_zero(
                                channels["TBI"], batch[img_index]
                            )
                        )
                        # this part is only relevant for TBI when doing multi channel segmentation with modality drop 
                        # if (have FLAIR or T2, no SWI: label on Flair) (have SWI, no FLAIR and T2 : label on SWI) (Other: merged)
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
                            label = batch[label_index][:, [seg_channel], :, :, :].to(
                                device
                            )
                        else:
                            label = batch[label_index].to(device)
                    else:
                        label = batch[label_index][:, 0, :, :, :].to(            #TODO: need to check index
                            device
                        )  # for dropout
                        label = label[:, None, :, :, :]

                    input_data = torch.from_numpy(
                        np.zeros(
                            (
                                batch[img_index].shape[0],
                                len(total_modalities),
                                cropped_input_size[0],
                                cropped_input_size[1],
                                cropped_input_size[2],
                            ),
                            dtype=np.float32,
                        )
                    )
                    input_data[:, channel_map["TBI"], :, :, :] = batch[img_index]
                    input_data = input_data.to(device)
                    
                    out = model(input_data)  # run model

                    outputs.append(out)
                    labels.append(label)

                else: #other databases are similar
                    loader_index = data_loader_map[dataset]
                    batch = batch_data[loader_index]

                
                    if rand_drop:
                        _, batch[img_index] = utils.rand_set_channels_to_zero(channels[dataset], batch[img_index])          #ATLAS WILL ALWAYS BE ONE    
                    
                
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    input_data[:,channel_map[dataset],:,:,:] = batch[img_index]
                    input_data = input_data.to(device)
                    
                    label = batch[label_index].to(device)
                    out = model(input_data)
                    outputs.append(out)
                    labels.append(label)

            optimizer.zero_grad()
            combined_outs = torch.cat(outputs, dim=0)
            combined_labels = torch.cat(labels,dim=0)
            loss = loss_function(combined_outs, combined_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_size  // train_config.train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            wandb.log({"loss":loss.item(),"epoch":epoch+1})
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # save model
        if (epoch+1) % 50 == 0:            
            model_save_name = os.path.join(model_save_path,args.save_name + "_Epoch_" + str(epoch) + ".pth")
            opt_save_name= os.path.join(model_save_path,args.save_name + "_checkpoint_Epoch_" + str(epoch) + ".pt")
            torch.save(model.state_dict(), model_save_name)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss ,
            }, opt_save_name)
            print("Saved Model")
                
        # validation
        if (epoch + 1) % train_config.val_interval == 0:
            
            model.eval()
            with torch.no_grad():
                seg_channel = 0
                val_images = None    # TODO CAN REMOVE ?
                val_labels = None    # TODO CAN REMOVE ?
                val_outputs = None
                metric={}
                dice_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()                
                IOU_metric.reset()                
                for dataset in datasetlist:
                    metric[dataset]={}
                    loader_index = data_loader_map[dataset]
                    for val_data in val_loader[dataset]:
                        # batch = val_data[loader_index]                       
                        input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                        input_data[:,channel_map[dataset],:,:,:] = val_data[0]
                        input_data = input_data.to(device)
                        if dataset == "BRATS" and train_config.BRATS_two_channel_seg:
                            label = val_data[1][:,[0],:,:,:].to(device)
                        elif dataset == "TBI" and TBI_multi_channel_seg:
                            label = val_data[1][:,[2],:,:,:].to(device)                     
                        else:                        
                            label = val_data[1].to(device)  
                    
                                            
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        #using sliding window for the whole 3D image
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=label)
                        sensitivity_metric(y_pred=val_outputs, y=label)
                        precision_metric(y_pred=val_outputs, y=label)                        
                        IOU_metric(y_pred=val_outputs, y=label)           
                    metric[dataset]["dice"] = dice_metric.aggregate().item()
                    metric[dataset]["sensitivity"] = sensitivity_metric.aggregate()[0].item()
                    metric[dataset]["precision"] = precision_metric.aggregate()[0].item()                    
                    metric[dataset]["IOU"] = IOU_metric.aggregate().item()            
                    dice_metric.reset()
                    sensitivity_metric.reset()
                    precision_metric.reset()                    
                    IOU_metric.reset()
                    if metric[dataset]["dice"] > best_metric[dataset]:
                        best_metric[dataset] = metric[dataset]["dice"]
                        best_metric_epoch[dataset] = epoch + 1
                        if epoch>1:
                            model_save_name = os.path.join(model_save_path,  args.save_name + "_BEST_"+dataset+".pth")
                            torch.save(model.state_dict(), model_save_name)                   
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                            epoch + 1,dataset,metric[dataset]["dice"],dataset, best_metric[dataset], best_metric_epoch[dataset]
                        )
                    )
                    #wandb log  
                    #here only use wandb log to show other metric
                    wandb.log({"epoch_val":epoch+1,"mdice_"+dataset:metric[dataset]["dice"], "sensitivity_"+dataset:metric[dataset]["sensitivity"],"precision_"+dataset:metric[dataset]["precision"],"mIOU_"+dataset:metric[dataset]["IOU"]})

        # model = Unet(in_channels=len(total_modalities)).to(device)
        # model = utils.create_net("models/tesT_MSSEG_NO_DROP_checkpoint_Epoch_349.pt",'UNET',5, device, cuda_id)
        # model.eval()
        # print(model)
        
        
        test_model(model,val_loader, datasetlist, device, post_transforms, dice_metric, channel_map, total_modalities, cropped_input_size,cuda_id)





def test_model(model,val_loader, datasetlist, device, post_trans, dice_metric, channel_map, total_modalities, cropped_input_size,cuda_id):
    
    with torch.no_grad():

    
        for dataset in datasetlist:
            print(f"Testing on dataset: {dataset}")
            for val_data in val_loader[dataset]:
                input_data = torch.from_numpy(np.zeros((1, len(total_modalities), val_data[0].shape[2], val_data[0].shape[3], val_data[0].shape[4]), dtype=np.float32))
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

    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets", help="datasets for training, using '_' to separate", type=str)
    parser.add_argument("--save_name", help="File name for saving model weights and checkpoints", type=str, default='save')
    parser.add_argument("--rand_drop", help="0 or 1, 1 if random dropping modalities when training", type=int, default=0)

    args = parser.parse_args()
    rand_drop = bool(args.rand_drop)

    #load config
    train_config = config.Training_config()
    database_config = config.Database_config()
    for dataset in ["MSSEG"]: #["BRATS", "ATLAS", "WMH", "MSSEG"]:      #"TBI" 
        for rand_drop in [0]:  # [0,1]

            args.device_id = 1
            args.datasets = dataset
            args.save_name = f"new_test_{dataset}_rand_drop_{rand_drop}_02_12"
            args.rand_drop = rand_drop
            train_config.save_name = args.save_name
            train_config.load_model_path = "models/tesT_MSSEG_NO_DROP_checkpoint_Epoch_349.pt"

         
            try:
                train(train_config,database_config,args)
            except Exception as e:
                print(e)
                continue

    

    print("training Finished")


    



