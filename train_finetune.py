import argparse
import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,MeanIoU
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


def main(args,k_fold: None):

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M")

    torch.multiprocessing.set_sharing_strategy('file_system') 

    finetune_dataset = args.datasets.split("_")

    # load config
    train_config=config.Finetune_config()
    randomly_drop = bool(args.randomly_drop)
    Database_config=config.Database_config()
    add_invar_channel_to_pre_trained_model= train_config.add_invar_channel_to_pre_trained_model
    add_invar_layers_to_pre_trained_model= train_config.add_invar_layers_to_pre_trained_model
    cropped_input_size = train_config.cropped_input_size
   
    datasets_trained_initially=args.datasets_trained_initially.split("_")
    modality_remove = train_config.modality_remove
    epochs=train_config.epoch

    new_model_save_path = os.path.join(os.path.join(train_config.model_save_path, f"Finetune:_with_{args.datasets}_" + f"trained on {args.datasets_trained_initially}_" + date + "/"))
    if not os.path.exists(
        new_model_save_path):
        os.makedirs(new_model_save_path)

    if train_config.wandb_report:
        # Use wandb for recording
        run = wandb.init(
        #Set the project where this run will be logged
        project=train_config.project_name,
        name= "finetune invar slot with" + str(train_config.new_mod_finetune) +":_" + str(finetune_dataset) + "_"+ "rand_drop_" + str(args.randomly_drop) + f"_trained on: {str(args.datasets_trained_initially)}" + date
                )   
    
    if k_fold:
        print(f"Training split___: {k_fold}")

    # print setting for training
    print("lr: ",train_config.lr)
    print("Workers: ", train_config.workers)
    print("Batch size: ",train_config.train_batch_size)
    print("RANDOMLY DROP? ",randomly_drop)
    print(f"THe model trained on {datasets_trained_initially} to be fine tuned on {finetune_dataset}")

    if modality_remove !=  None:  
        # from channels remove one modality for all datasets in question
        channels = Database_config.channels
        for key,value in channels.items():
            channels[key] = [x for x in value if x != modality_remove]
        print(f"Removed {str(modality_remove)} from datasets")

    # set index
    img_index = 0
    label_index = 1
    mask_index = 2

    # Set the data size and total modalities
    channels= Database_config.channels
    train_size=Database_config.train_size
    total_size=Database_config.total_size
    datasetlist=args.datasets.split("_")
    data_size=0

    #####

    combination_map = {}
    #loop for allocating combination of mnodalities for each dataset
    for dataset in datasetlist:
        combination_map[dataset] = utils.map_combinations(
            channels[dataset])

    total_modalities = []
    total_modalities = set(total_modalities)
    # Loop for allocating channel
    channel_map = {}

    ## Previous datasets ######
    for dataset in datasets_trained_initially:
        total_modalities = total_modalities.union(set(channels[dataset]))
    
    # if "DWI"in total_modalities:
    #     total_modalities.remove("DWI")
   
    total_modalities = sorted(list(total_modalities))
    

    # get max dataset size. Over sample smaller datasets.
    for dataset in finetune_dataset:
        if k_fold is not None:
            data_size = len(k_fold['train'])
        else:
            data_size = max(data_size, train_size[dataset])
        
    print("Total modalities: ", total_modalities)
    print("Data_size", data_size)

    
    if add_invar_channel_to_pre_trained_model or train_config.new_mod_finetune:
        total_modalities.append(train_config.new_mod_finetune)                  


    for dataset in datasetlist:
        channel_map[dataset] = utils.map_channels(
            channels[dataset],
            total_modalities,
            rand_assign=False,
        )

    # path initialization
    train_loaders = []
    data_loader_map = {}    
    model_save_path = train_config.model_save_path    
    val_loader={}
 
    # get dataloader
    train_config.modality_remove = None
    train_loaders, val_loader,data_loader_map = get_dataloader(train_config, Database_config, [dataset], cropped_input_size, data_size,channels,k_fold=k_fold,dataset_use="Train")



    # initialize GPU
    print("Running on GPU:" + str(args.device_id))
    print("Running for epochs:" + str(epochs))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    # initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    sensitivity_metric=ConfusionMatrixMetric(include_background=True, metric_name='sensitivity', reduction="mean", get_not_nans=False)    
    precision_metric=ConfusionMatrixMetric(include_background=True, metric_name='precision', reduction="mean", get_not_nans=False)   
    IOU_metric=MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # load pre-trained weights
    if train_config.model_type == "UNET":
        print("TRAINING WITH UNET")
        in_channel = len(total_modalities) 
        epoched = 0
        print("LOADING MODEL: ", args.load_model_finetune_path)
        # add slot to the pre-trained model to finetune unseen modality.
        if add_invar_channel_to_pre_trained_model:

            invariant_channel  = False
            load = torch.load(
                args.load_model_finetune_path,
                map_location={"cuda:0": cuda_id, "cuda:1": cuda_id},
            )
            checkpoint = utils.add_invar_input_to_pre_trained(load)

        elif add_invar_layers_to_pre_trained_model:
            invariant_channel = True
            load = torch.load(
                args.load_model_finetune_path,
                map_location={"cuda:0": cuda_id, "cuda:1": cuda_id},
            )
            checkpoint = utils.add_invar_layers_to_pre_trained(load)
      
        else:
            invariant_channel = False
            checkpoint = torch.load(
                args.load_model_finetune_path,
                map_location={"cuda:0": cuda_id, "cuda:1": cuda_id},
            )
        

        model = agnostic_net(in_channels=in_channel,invariant_channel=invariant_channel).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
        
        
        model.load_state_dict(checkpoint)

    # defined loss function
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5) 

    # initialise best metric
    best_metric={}
    best_metric_epoch={}
    for dataset in datasetlist:
        best_metric[dataset] = -1
        best_metric_epoch[dataset] = -1   
    metric_values = list()

    # training
    for epoch in range(epoched,epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # drop learning rate
        if train_config.drop_learning_rate and epoch >= train_config.drop_learning_rate_epoch:
            for g in optimizer.param_groups:
                g['lr'] = train_config.drop_learning_rate_value

        for batch_data in zip(*train_loaders): 
            step += 1
            outputs = []
            labels = []
            for dataset in finetune_dataset:
                # Only for BRATS    BRATS may use different ground truth
                if dataset == "BRATS":
                    loader_index = data_loader_map["BRATS"]
                    batch = batch_data[loader_index]

                     

                    if randomly_drop:

                        modalities_remaining, batch[img_index] = utils.rand_set_channels_to_zero(channels["BRATS"], batch[img_index])
                        for i in range (batch[label_index].shape[0]):
                            # For BRATS because edema can only be seen on some modalities can use different ground truth for different sets of modalities in input (this need the ground truth file that have multiple channels and for each channel it contain a different gound truth)
                            if (0 not in modalities_remaining[i]) and (3 not in modalities_remaining[i]):
                                # Edema cannot be seen so change segmentation to labels without edema
                                seg_channel = 1
                            else:
                                seg_channel = 0
                            if Database_config.BRATS_two_channel_seg:                            
                                label[i,:,:,:,:] = batch[label_index][i,[seg_channel],:,:,:].to(device)                                
                            else:
                                # default setting of our work: not using different labels
                                label = batch[label_index].to(device)
                    else:
                        label = batch[label_index].to(device)                         
                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    
                    # When using invariant layers, handle FLAIR separately
                    modality = None
                    if add_invar_layers_to_pre_trained_model and modality in channels[dataset]:
                        # Find FLAIR index in the dataset channels
                        flair_idx = channels[dataset].index(modality)
                        
                        # Map non-FLAIR modalities to regular channels
                        non_flair_channels = [ch for ch in channels[dataset] if ch != modality]
                        non_flair_channel_map = utils.map_channels(non_flair_channels, total_modalities[:-1], rand_assign=False)
                        
                        # Place non-FLAIR data in regular modality channels
                        non_flair_data = torch.cat([batch[img_index][:, :flair_idx], batch[img_index][:, flair_idx+1:]], dim=1)
                        input_data[:, non_flair_channel_map, :, :, :] = non_flair_data
                        
                        # Place FLAIR data in the invariant channel (last position)
                        input_data[:, -1, :, :, :] = batch[img_index][:, flair_idx, :, :, :]
                        
                        # print(f"✓ FLAIR placed in invariant channel (position {len(total_modalities)-1})")
                        # print(f"✓ Other modalities {non_flair_channels} placed in positions {non_flair_channel_map}")
                    else:
                        # Original behavior for non-invariant models
                        input_data[:,channel_map["BRATS"],:,:,:] = batch[img_index]
                    input_data = input_data.to(device)           
                    out = model(input_data)
                    outputs.append(out)
                    labels.append(label)                  
                else: #other databases are similar
                    loader_index = data_loader_map[dataset]
                    batch = batch_data[loader_index]               
                    # if randomly_drop:
                    #     _, batch[img_index] = utils.rand_set_channels_to_zero(channels[dataset], batch[img_index])     #ATLAS WILL ALWAYS BE ONE
                    
                    
                    if randomly_drop:
                         #_, batch[img_index]
                        
                        all_modalities_dropped, all_modalities_remaining, batch[img_index] = utils.rand_set_channels_to_zero_with_invar(
                            channels[dataset], batch[img_index],mask_data=batch[mask_index],domain_invariant=False,combination_map = combination_map[dataset]
                            )  # ATLAS WILL ALWAYS BE ONE CHANNEL (no drop) 

                    input_data = torch.from_numpy(np.zeros((batch[img_index].shape[0],len(total_modalities),cropped_input_size[0],cropped_input_size[1],cropped_input_size[2]),dtype=np.float32))
                    
                    # When using invariant layers, handle FLAIR separately
                    modality = None
                    if add_invar_layers_to_pre_trained_model and modality in channels[dataset]:
                        # Find FLAIR index in the dataset channels
                        flair_idx = channels[dataset].index(modality)
                        
                        # Map non-FLAIR modalities to regular channels
                        non_flair_channels = [ch for ch in channels[dataset] if ch != modality]
                        non_flair_channel_map = utils.map_channels(non_flair_channels, total_modalities[:-1], rand_assign=False)
                        
                        # Place non-FLAIR data in regular modality channels
                        non_flair_data = torch.cat([batch[img_index][:, :flair_idx], batch[img_index][:, flair_idx+1:]], dim=1)
                        input_data[:, non_flair_channel_map, :, :, :] = non_flair_data
                        
                        # Place FLAIR data in the invariant channel (last position)
                        input_data[:, -1, :, :, :] = batch[img_index][:, flair_idx, :, :, :]
                        
                        # print(f"✓ FLAIR placed in invariant channel (position {len(total_modalities)-1})")
                        # print(f"✓ Other modalities {non_flair_channels} placed in positions {non_flair_channel_map}")
                    else:
                        # Original behavior for non-invariant models
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
            if train_config.wandb_report:
                wandb.log({"loss":loss.item(),"epoch":epoch+1})
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # save model
        if (epoch+1) % 50 == 0:   
            model_save_name = (
            new_model_save_path
            +  train_config.project_name
            + "_random_drop_"
            + str(randomly_drop)
            + "_"
            + date
            + "_Epoch_"
            + str(epoch)
            + ".pth"
            )

            opt_save_name = (
            new_model_save_path
            +  train_config.project_name
            + "_random_drop_"
            + str(randomly_drop)
            + "_"
            + date
            + "_Epoch_"
            + str(epoch)
            + ".pt"
            )

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
                val_images = None
                val_labels = None
                val_outputs = None
                metric={}
                dice_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()                
                IOU_metric.reset()                
                for dataset in finetune_dataset:
                    metric[dataset]={}
                    loader_index = data_loader_map[dataset]
                    for val_data in val_loader[dataset]:
                        # batch = val_data[loader_index]
                        input_data = torch.from_numpy(np.zeros((1,len(total_modalities),val_data[0].shape[2],val_data[0].shape[3],val_data[0].shape[4]),dtype=np.float32))
                        
                        # When using invariant layers, handle FLAIR separately in validation too
                        modality = None
                        if add_invar_layers_to_pre_trained_model and modality in channels[dataset]:
                            # Find FLAIR index in the dataset channels
                            flair_idx = channels[dataset].index(modality)
                            
                            # Map non-FLAIR modalities to regular channels
                            non_flair_channels = [ch for ch in channels[dataset] if ch != modality]
                            non_flair_channel_map = utils.map_channels(non_flair_channels, total_modalities[:-1], rand_assign=False)
                            
                            # Place non-FLAIR data in regular modality channels
                            if flair_idx == 0:
                                non_flair_data = val_data[0][:, 1:, :, :, :]
                            elif flair_idx == val_data[0].shape[1] - 1:
                                non_flair_data = val_data[0][:, :-1, :, :, :]
                            else:
                                non_flair_data = torch.cat([val_data[0][:, :flair_idx, :, :, :], val_data[0][:, flair_idx+1:, :, :, :]], dim=1)
                            
                            input_data[:, non_flair_channel_map, :, :, :] = non_flair_data
                            
                            # Place FLAIR data in the invariant channel (last position)
                            input_data[:, -1, :, :, :] = val_data[0][:, flair_idx, :, :, :]
                            
                            # print(f"✓ VAL: FLAIR placed in invariant channel (position {len(total_modalities)-1})")
                            # print(f"✓ VAL: Other modalities {non_flair_channels} placed in positions {non_flair_channel_map}")
                        else:
                            # Original behavior for non-invariant models
                            input_data[:,channel_map[dataset],:,:,:] = val_data[0]
                        input_data = input_data.to(device)
                        if dataset == "BRATS" and Database_config.BRATS_two_channel_seg:
                            label = val_data[1][:,[0],:,:,:].to(device)                      
                        else:                        
                            label = val_data[1].to(device)                        
                        roi_size = (cropped_input_size[0], cropped_input_size[1], cropped_input_size[2])
                        sw_batch_size = 1
                        # using sliding window for the whole 3D image
                        val_outputs = sliding_window_inference(input_data, roi_size, sw_batch_size, model)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
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
                            model_save_best_name = (
                                    new_model_save_path
                                    +    train_config.project_name
                                    + "_random_drop_"
                                    + str(randomly_drop)
                                    + "_"
                                    + date
                                    + "_BEST_"
                                    + dataset
                                    + ".pth"
                                )
                            torch.save(model.state_dict(), model_save_best_name)                                                  
                            print("saved new best metric model")
                    print(
                        "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                            epoch + 1,dataset,metric[dataset]["dice"],dataset, best_metric[dataset], best_metric_epoch[dataset]
                        )
                    )
                    if train_config.wandb_report:
                        # wandb log
                        # here only use wandb log to show other metric
                        wandb.log({"epoch_val":epoch+1,"mdice_"+dataset:metric[dataset]["dice"], "sensitivity_"+dataset:metric[dataset]["sensitivity"],"precision_"+dataset:metric[dataset]["precision"],"mIOU_"+dataset:metric[dataset]["IOU"]})


if __name__ == "__main__":


    torch.multiprocessing.set_sharing_strategy('file_system') 

    #command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument("--datasets", help="datasets for training, using '_' to separate", type=str)
    #parser.add_argument("--save_name", help="File name for saving model weights and checkpoints", type=str, default='save')
    parser.add_argument("--randomly_drop", help="0 or 1, 1 if random dropping modalities when training", type=int, default='1')
    parser.add_argument("--load_model_finetune_path", help="The path of the pretrained model", type=str)
    parser.add_argument("--manual_channel_map", help="The allocated channel index of the modalities(each channel) in the finetuning input (start from 0)     Using '_' to separate.  For example, 1_3 means the first modality in the finetuning input goes to the second channel of the model, and the second modality goes to the fourth channel of the model.", type=str)
    parser.add_argument("--datasets_trained_initially", help="modalities used for training the pre-train model using '_' to separate", type=str)
    args = parser.parse_args()

    args.device_id = 0
    args.datasets = "WMH"
    args.randomly_drop = 1
    args.load_model_finetune_path = 'models/UPPER_BOUND_FINETUNE/2025-07-03_15-19/WMH_PRELIM_TEST_random_drop_True_2025-07-03_15-19_Epoch_599.pth'
    args.datasets_trained_initially = 'TBI_ISLES2022_BRATS_MSSEG_ATLAS'  

    
    main(args,k_fold=None)
