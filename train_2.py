import torch
from glob import glob
import os
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from monai.losses import DiceCELoss
from nets.unet import res_unet as unet_old
from nets.unet_deep import res_unet as unet_deep
import numpy as np
import utils
import wandb
import config
import argparse
import datetime
from dataloader import get_dataloader
import copy
from tqdm import tqdm




def main(train_config,aug_config,database_config,k_fold,args,channels_copy):


    torch.multiprocessing.set_sharing_strategy("file_system")


    print(f"prob_tumor_scale_shift: {aug_config.prob_tumor_scale_shift}")
    print(f"prob_brain_scale_shift: {aug_config.prob_brain_scale_shift}")    

    # load config

    rand_assign_channels = train_config.rand_assign_channels
    domain_invariant_slot = train_config.domain_invariant_slot
    load_model_path = train_config.load_model_path
    modality_remove =  train_config.modality_remove
    randomly_drop = bool(train_config.random_drop)
    single_slot = train_config.single_slot
    wandb_active = train_config.wandb_active
    contrast_augmentation = train_config.contrast_augmentation
    lr_sched = train_config.lr_sched
    held_out_datasets = train_config.held_out_datasets
    dropped_modality = train_config.modality_remove
    cropped_input_size = train_config.cropped_input_size
    epochs = train_config.epoch
    
    
    if randomly_drop and single_slot:
        raise ValueError("Cannot have both random drop and single slot")
    

    # create save model path. If the path does not exist, create it
    if wandb_active:
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d_%H-%M")
        model_save_path = os.path.join(train_config.model_save_path, args.datasets + "/" + date + "/")
        if not os.path.exists(
        model_save_path
        ):
            os.makedirs(model_save_path)
        config.save_config_file(model_save_path)

       
    if wandb_active:
        # Use wandb for recording
        wandb.init(
            # Set the project where this run will be logged
            project=train_config.project_name,
            name=(
                train_config.project_name 
                + args.datasets
                + "_random_drop_"
                + str(randomly_drop)
                + "_"
                + 'modality_remove:_'
                + str(modality_remove)
                + date
            )  
        )

    # print setting for training
    print("lr: ", train_config.lr)
    print("Workers: ", train_config.workers)
    print("Batch Size: ", train_config.train_batch_size)
    print("RANDOM DROP: ", randomly_drop)

    print("\n #######  Training_methods #######")
    print("Domain Invariant Slot: ", domain_invariant_slot)
    print("Training with single input channel/slot: ", single_slot)
    print("Randomly assign channels: ", rand_assign_channels)
    print("Modality to remove: ", modality_remove, "\n")

    if k_fold:
        print(f"Training split___: {k_fold}")


    # set index
    img_index = 0
    label_index = 1
    mask_index = 2

    if modality_remove !=  None:  
        # from channels remove one modality for all datasets in question
        channels = database_config.channels
        for key,value in channels.items():
            channels[key] = [x for x in value if x != modality_remove]
        print(f"Removed {str(modality_remove)} from datasets")


    # Set the data size and total modalities
    channels = database_config.channels
    train_size = database_config.train_size
    datasetlist = args.datasets.split("_")
    total_modalities = []
    total_modalities = set(total_modalities)
    data_size = 0
    for dataset in datasetlist:

        if domain_invariant_slot == True:
            channels[dataset].append("invar")

        if k_fold is not None:
            data_size = len(k_fold['train'])
        else:
            data_size = max(data_size, train_size[dataset])

        total_modalities = total_modalities.union(set(channels[dataset]))
        
    total_modalities = sorted(list(total_modalities))
    print("Data_size", data_size)


    # load data for training   
    train_loaders,val_loader,data_loader_map = get_dataloader(train_config, database_config,datasetlist, cropped_input_size , data_size,channels_copy,k_fold,dataset_use = "Train")
    # print('load WMH only for validation:')

    # Load data for validation only. Temporarily modify train_config to not drop FLAIR for WMH loading
    validate_data_only = True

    if validate_data_only:
        # Temporarily store the original value
        original_modality_remove = train_config.modality_remove
        # Set to None for validation data loading
        train_config.modality_remove = None
        WMH_loader,val_only_loader,data_laod = get_dataloader(train_config, database_config,held_out_datasets,cropped_input_size , data_size,channels_copy,k_fold,dataset_use = "Val ONLY")
        # Restore the original value
        train_config.modality_remove = original_modality_remove
    

    # initialize GPU
    print("Running on GPU:" + str(args.device_id))
    print("Running for epochs:" + str(epochs))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id)
    torch.cuda.set_device(cuda_id)

    # initialize metrics
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
 
    sensitivity_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name="sensitivity",
        reduction="mean",
        get_not_nans=False,
    )
    precision_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name="precision",
        reduction="mean",
        get_not_nans=False,
    )
    IOU_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # initialize the model (only show multiunet here)

    if train_config.single_slot:
        in_channel = 1  
    else:
        in_channel = len(total_modalities)


    if train_config.model_type == "old_unet":
        print("TRAINING WITH old_unet")
        model = unet_old(in_channels=in_channel).to(device)

    elif train_config.model_type == "deep_unet":
        print("TRAINING WITH DEEP UNET")
        model = unet_deep(in_channels=in_channel,invariant_channel= train_config.domain_invariant_layers).to(device)



    
    # ################################
        
    print("In Channels= ", len(total_modalities))
    print("Batch size = ", train_config.train_batch_size)


    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    epoched = 0

    # load pre-trained weights
    if train_config.load_pre_trained_model:

        print("LOADING MODEL: ", load_model_path)

        checkpoint = torch.load(
                load_model_path, map_location={"cuda:0": cuda_id, "cuda:1": cuda_id}
            )
        
        model.load_state_dict(checkpoint)   

        

    # Initialize loss function 
    loss_function = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)

    # initialize the best metric
    best_metric = {}
    best_metric_epoch = {}
    best_avg_dice = 0
    for dataset in datasetlist + held_out_datasets:
        best_metric[dataset] = -1
        best_metric_epoch[dataset] = -1


    # Loop for allocating channel
    channel_map = {}

    for dataset in datasetlist:
        channel_map[dataset] = utils.map_channels(
            channels[dataset],
            total_modalities,
            rand_assign=rand_assign_channels,
        )

    combination_map = {}
    if domain_invariant_slot == True:
        #loop for allocating combination of modalities for each dataset
        for dataset in datasetlist:
            combination_map[dataset] = utils.map_combinations(
                channels[dataset],invar_ratio= 0.2)
    else:
        for dataset in datasetlist:
            combination_map[dataset] = utils.map_combinations(
                channels[dataset],invar_ratio= 0)


    if lr_sched:
        scheduler = utils.lr_schedule(epochs=epochs,optimizer=optimizer)

    ##training##

    for epoch in tqdm(range(epoched, epochs)):

        print("-" * 10)
        #print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0


        # #drop learning rate
        if not lr_sched:
            if (
                train_config.drop_learning_rate
                and epoch >= train_config.drop_learning_rate_epoch
            ):
                for g in optimizer.param_groups:
                    g["lr"] = train_config.drop_learning_rate_value

        for batch_data in zip(*train_loaders):
            
            step += 1
            modality_outputs = []
            aux_outputs = []
            labels = []
            total_modalities_dropped = []

            for dataset in datasetlist:

                # Only for BRATS BRATS may use different ground truth
                if dataset == "BRATS":
                    loader_index = data_loader_map["BRATS"]
                    batch = batch_data[loader_index]

                    if randomly_drop:
                        modalities_dropped,modalities_remaining, batch[img_index] = (
                            utils.rand_set_channels_to_zero_with_invar(
                                channels["BRATS"], batch[img_index],mask_data = batch[mask_index],domain_invariant=domain_invariant_slot, batch_label_data=batch[label_index],device_id = args.device_id,contrast_augmentation=contrast_augmentation,combination_map=combination_map["BRATS"],augmentation_config=aug_config
                            )
                        )
                        for i in range(batch[label_index].shape[0]):

                            
                            if database_config.BRATS_two_channel_seg:
                                # For BRATS because edema can only be seen on some modalities can use different ground truth for different sets of modalities in input (this need the ground truth file that have multiple channels and for each channel it contain a different gound truth)
                                if (0 not in modalities_remaining[i]) and (
                                    3 not in modalities_remaining[i]
                                ):
                                    # Edema cannot be seen so change segmentation to labels without edema
                                    seg_channel = 1
                                else:
                                    seg_channel = 0
                                if database_config.BRATS_two_channel_seg:
                                    label[i, :, :, :, :] = batch[label_index][
                                        i, [seg_channel], :, :, :
                                    ].to(device)
                            else:
                                # default setting of our work: not using different labels
                                label = batch[label_index].to(device)
                    else:
                        label = batch[label_index].to(device)
                    
                    
                    if single_slot:
                        input_data,_ = utils.single_slot(batch[img_index])
                    
                    else:
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


                        if rand_assign_channels:

                            channel_map["BRATS"] = utils.rand_assign_channels(channels["BRATS"], total_modalities)

                            #random.shuffle(channel_map["BRATS"])                        
                    
                        input_data[:, channel_map["BRATS"], :, :, :] = batch[img_index]
                    

                    input_data = input_data.to(device)
                    modality_output = model(input_data)
                    modality_outputs.append(modality_output)
                    labels.append(label)
                    if randomly_drop:
                        total_modalities_dropped.append(modalities_dropped)
                    #FIXME: need to set  Brats up so for single slot train depending on which modality is randomly selected use a different label 
                    # currently only works for merged labels - which is okay as only going to use this when training a single slot. 

                # TBI may use a different ground truth. 
                elif dataset == "TBI":
                    loader_index = data_loader_map["TBI"]
                    batch = batch_data[loader_index]
                    TBI_multi_channel_seg = database_config.TBI_multichannel

                    if single_slot or randomly_drop:

                        if single_slot:
                            input_data,modalities_remaining = utils.single_slot(batch[img_index])

                        if randomly_drop:
                            modalities_dropped,modalities_remaining, batch[img_index] = (
                                utils.rand_set_channels_to_zero_with_invar(
                                    channels["TBI"], batch[img_index],mask_data = batch[mask_index],domain_invariant=domain_invariant_slot, batch_label_data=batch[label_index], device_id= args.device_id,contrast_augmentation=contrast_augmentation,combination_map=combination_map["TBI"],augmentation_config=aug_config
                                )
                        )
                        # this part is only relevant for TBI when doing multi channel segmentation with modality drop 
                        # if (have FLAIR or T2, no SWI: label on Flair) (have SWI, no FLAIR and T2 : label on SWI) (Other: merged)

                        # TODO: Remove Channels before  training will need to remove the corresponding labels. For invariant slot. 
                        #  

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
                        label = batch[label_index][:, 0, :, :, :].to(            
                            device
                        )  # for dropout
                        label = label[:, None, :, :, :]

                    if single_slot:
                        input_data = input_data

                    else:  
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

                        if rand_assign_channels:
                            channel_map["TBI"] = utils.rand_assign_channels(channel_map["TBI"], total_modalities)
                            #random.shuffle(channel_map["TBI"]) 

                        input_data[:, channel_map["TBI"], :, :, :] = batch[img_index]
                    
                    input_data = input_data.to(device)
                    modality_output = model(input_data)  # run model
                    modality_outputs.append(modality_output)      
                    labels.append(label)
                    if randomly_drop:
                        total_modalities_dropped.append(modalities_dropped)


                else:  # other databases are similar
                    loader_index = data_loader_map[dataset]
                    batch = batch_data[loader_index]
                  
                    # Perform cropping or other transformations here

                    if single_slot:
                        input_data,_ = utils.single_slot(batch[img_index])

                    
                    else:
                        if randomly_drop:
                            modalities_dropped, modalities_remaining, batch[img_index] = utils.rand_set_channels_to_zero_with_invar(
                                channels[dataset], batch[img_index],mask_data = batch[mask_index],domain_invariant=domain_invariant_slot, batch_label_data = batch[label_index],device_id =args.device_id,contrast_augmentation=contrast_augmentation,combination_map=combination_map[dataset],augmentation_config=aug_config
                            )  # ATLAS WILL ALWAYS BE ONE CHANNEL (no drop)
                        
                        input_data = torch.from_numpy(
                        np.zeros(
                            (
                                batch[img_index].shape[0],
                                len(total_modalities),
                                cropped_input_size[0],
                                cropped_input_size[1],
                                cropped_input_size[2],
                            ),
                            dtype=np.float32,))
                        
                        
                        if rand_assign_channels:
                            channel_map[dataset] = utils.rand_assign_channels(channel_map[dataset], total_modalities)
                            #random.shuffle(channel_map[dataset]) 

                        input_data[:, channel_map[dataset], :, :, :] = batch[img_index]

                    input_data = input_data.to(device)

                    label = batch[label_index].to(device)
                    combined_outs= model(input_data)
                    modality_outputs.append(combined_outs)
                    

                    labels.append(label)
                    if randomly_drop:
                        total_modalities_dropped.append(modalities_dropped)
            
            optimizer.zero_grad()
            modality_outs = torch.cat(modality_outputs, dim=0)
    
            
            combined_labels = torch.cat(labels, dim=0)
            modality_loss = loss_function(modality_outs, combined_labels)
            loss = modality_loss  

            if torch.isnan(loss):
                raise ValueError("Loss produced NaN value")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_size // train_config.train_batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            if wandb_active:
                wandb.log({"loss": loss.item(), "epoch": epoch + 1,"lr": optimizer.param_groups[0]["lr"]})
          
        if lr_sched:
            scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print("\n------------------------\n")
        # save model
        if (epoch + 1) % 50 == 0:
            model_save_name = (
                model_save_path
                +    train_config.project_name
                + "_random_drop_"
                + str(randomly_drop)
                + "_"
                + date
                + "_Epoch_"
                + str(epoch)
                + ".pth"
            )

            opt_save_name = (
                model_save_path
                +    train_config.project_name
                + "_random_drop_"
                + str(randomly_drop)
                + "_"
                + date
                + "_checkpoint_Epoch_"
                + str(epoch)
                + ".pt"
            )
            torch.save(model.state_dict(), model_save_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                opt_save_name,
            )
            print("Saved Model")


            # if wandb_active:
            #     wandb.log({"model_save_name":model_save_name})

        # validation

        if (epoch + 1) % train_config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                seg_channel = 0
                # val_images = None
                # val_labels = None
                val_outputs = None
                metric = {}
                dice_metric.reset()
                sensitivity_metric.reset()
                precision_metric.reset()
                IOU_metric.reset()
                total_av_dice = list()

                ################ I want to test WMH as I go along to see how it does #######################
                
                # Test with all modalities
        
                # if domain_invariant_slot:
                    
                #     for dataset in ["ATLAS","TUMOUR2"]:
                        
                #         for val_data in val_only_loader[dataset]:

                #             channels[dataset] = channels_copy[dataset]

                #             channels[dataset] = [x if x != 'T1' else 'invar' for x in channels[dataset]]
                            
                #             channel_map[dataset] = utils.map_channels(
                #                 channels[dataset],
                #                 total_modalities,
                #                 rand_assign=rand_assign_channels,
                #             )

                #             if single_slot:
                #                 input_data, _ = utils.single_slot(val_data[0])
                #             else:
                #                 input_data = torch.from_numpy(
                #                     np.zeros(
                #                         (
                #                             1,
                #                             len(total_modalities),
                #                             val_data[0].shape[2],
                #                             val_data[0].shape[3],
                #                             val_data[0].shape[4],
                #                         ),
                #                         dtype=np.float32,
                #                     )
                #                 )
                #                 if domain_invariant_slot:
                #                     input_data[:, channel_map[dataset], :, :, :] = val_data[0]
                #                 else:
                #                     input_data[:, channel_map[dataset], :, :, :] = val_data[0]

                #             input_data = input_data.to(device)
                #             label = val_data[1].to(device)
                #             roi_size = (
                #                 cropped_input_size[0],
                #                 cropped_input_size[1],
                #                 cropped_input_size[2],
                #             )
                #             sw_batch_size = 1

                #             val_outputs = sliding_window_inference(
                #             input_data, roi_size, sw_batch_size,model
                #             )
                #             val_outputs = [
                #                 post_trans(i) for i in decollate_batch(val_outputs)
                #             ]
                #             dice_metric(y_pred=val_outputs, y=label)
                #             sensitivity_metric(y_pred=val_outputs, y=label)
                #             precision_metric(y_pred=val_outputs, y=label)
                #             IOU_metric(y_pred=val_outputs, y=label)
                #         metric[dataset] = {
                #             "dice": dice_metric.aggregate().item(),
                #             "sensitivity": sensitivity_metric.aggregate()[0].item(),
                #             "precision": precision_metric.aggregate()[0].item(),
                #             "IOU": IOU_metric.aggregate().item(),
                #         }
                    
                    
                #         dice_metric.reset()
                    
                #         sensitivity_metric.reset()
                #         precision_metric.reset()
                #         IOU_metric.reset()
                #         if metric[dataset]["dice"] > best_metric.get(dataset, -1):
                #             best_metric[dataset] = metric[dataset]["dice"]
                #             best_metric_epoch[dataset] = epoch + 1
                #         print("current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                #                 epoch + 1,
                #                 dataset,
                #                 metric[dataset]["dice"],
                #                 dataset,
                #                 best_metric[dataset],
                #                 best_metric_epoch[dataset],
                #             )
                #         )
        

                    
                

                #         total_av_dice.append(metric[dataset]["dice"])

                
                    
                    ##### Test with only the invariant channel   #####
                    # dice_metric_invar = DiceMetric(include_background=True, reduction="mean")
                    # sensitivity_metric_invar = ConfusionMatrixMetric(metric_name="sensitivity", include_background=True)
                    # precision_metric_invar = ConfusionMatrixMetric(metric_name="precision", include_background=True)
                    # IOU_metric_invar = MeanIoU(include_background=True)
                
                    
                    # for val_data in val_only_loader["WMH"]:
                    #     # Create input with only the invariant channel
                    #     input_data_invar = torch.from_numpy(
                    #         np.zeros(
                    #             (
                    #                 1,
                    #                 len(total_modalities),
                    #                 val_data[0].shape[2],
                    #                 val_data[0].shape[3],
                    #                 val_data[0].shape[4],
                    #             ),
                    #             dtype=np.float32,
                    #         )
                    #     )
                        
                    #     # Find the invariant channel index
                    #     invar_channel_idx = None
                    #     for i, modality in enumerate(channels['WMH']):
                    #         if modality == 'invar':
                    #             invar_channel_idx = channel_map['WMH'][i]
                    #             invar_id = invar_channel_idx
                    #             break
                            
                        
                    #     if invar_id is not None:
                    #         # Copy only the invariant channel
                    #         input_data_invar[:, invar_id, :, :, :] = val_data[0][:, i, :, :, :]
                    #     else:
                    #         print("Warning: No invariant channel found in WMH data")
                        
                    #     input_data_invar = input_data_invar.to(device)
                    #     label = val_data[1].to(device)
                    #     roi_size = (
                    #         cropped_input_size[0],
                    #         cropped_input_size[1],
                    #         cropped_input_size[2],
                    #     )
                    #     sw_batch_size = 1

                    
                        
                    #     val_outputs_invar = sliding_window_inference(
                    #         input_data_invar, roi_size, sw_batch_size,model
                    #     )
                    #     val_outputs_invar = [
                    #         post_trans(i) for i in decollate_batch(val_outputs_invar)
                    #     ]
                    #     dice_metric_invar(y_pred=val_outputs_invar, y=label)
                    #     sensitivity_metric_invar(y_pred=val_outputs_invar, y=label)
                    #     precision_metric_invar(y_pred=val_outputs_invar, y=label)
                    #     IOU_metric_invar(y_pred=val_outputs_invar, y=label)
                    
                    # metric["WMH_invar"] = {
                    #     "dice": dice_metric_invar.aggregate().item(),
                    #     "sensitivity": sensitivity_metric_invar.aggregate()[0].item(),
                    #     "precision": precision_metric_invar.aggregate()[0].item(),
                    #     "IOU": IOU_metric_invar.aggregate().item(),
                    # }
                    
                    # print(
                    #     "current epoch: {} current mean dice WMH (invar only): {:.4f}".format(
                    #         epoch + 1,
                    #         metric["WMH_invar"]["dice"],
                    #     )
                    # )
                    
                    # if wandb_active:
                    #     wandb.log(
                    #         {
                    #             "epoch_val": epoch + 1,
                    #             "mdice_ATLAS": metric["ATLAS"]["dice"],          # was WMH before
                    #             #"mdice_WMH_invar": metric["WMH_invar"]["dice"],
                    #             "mdice_TUMOUR2": metric["TUMOUR2"]["dice"], }
                    #     )


                #########################################
             
            
                
                for dataset in [*datasetlist, *held_out_datasets]:
                    metric[dataset] = {}
                    
                    if dataset in held_out_datasets:
                        channels[dataset] = channels_copy[dataset]
                        channels[dataset] = [x if x != train_config.modality_remove else 'invar' for x in channels[dataset]]
                        
                    
                    # Choose appropriate loader based on dataset type eitehr held out training or completely held out dataset.
                    if dataset in held_out_datasets:
                        current_loader = val_only_loader[dataset]
                    elif dataset in datasetlist:
                        current_loader = val_loader[dataset]

                    for val_data in current_loader:
                        
                        channel_map[dataset] = utils.map_channels(
                            channels[dataset],
                            total_modalities,
                            rand_assign=rand_assign_channels,
                        )

                        if single_slot:
                            input_data,_ = utils.single_slot(val_data[0])
                        
                        else:
                            input_data = torch.from_numpy(
                                np.zeros(
                                    (
                                        1,
                                        len(total_modalities),
                                        val_data[0].shape[2],
                                        val_data[0].shape[3],
                                        val_data[0].shape[4],
                                    ),
                                    dtype=np.float32,
                                )
                            )
                            if domain_invariant_slot == True:
                                if dataset in held_out_datasets:
                                    input_data[:, channel_map[dataset], :, :, :] = val_data[0]
                                elif dataset in datasetlist:
                                    input_data[:, channel_map[dataset][:-1], :, :, :] = val_data[0]
                            else:
                                input_data[:, channel_map[dataset], :, :, :] = val_data[0]

                        
                        input_data = input_data.to(device)

                        if dataset == "BRATS" and database_config.BRATS_two_channel_seg:
                            label = val_data[1][:, [0], :, :, :].to(device)
                        elif dataset == "TBI" and TBI_multi_channel_seg:
                            label = val_data[1][:,[2],:,:,:].to(device)
                        
                        else:
                            label = val_data[1].to(device)
                        roi_size = (
                            cropped_input_size[0],
                            cropped_input_size[1],
                            cropped_input_size[2],
                        )
                        sw_batch_size = 1

                        # using sliding window for the whole 3D image
                        val_outputs = sliding_window_inference(
                            input_data, roi_size, sw_batch_size,model
                        )
                        val_outputs = [
                            post_trans(i) for i in decollate_batch(val_outputs)
                        ]
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
                        if epoch > 1:
                            model_save_best_name =  (   
                                model_save_path
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
                            print(f"Saved new best dice model, for {dataset}:_{best_metric[dataset]}")
                            print(best_metric[dataset])

                    print(
                        "current epoch: {} current mean dice {}: {:.4f} best mean dice {}: {:.4f} at epoch {}".format(
                            epoch + 1,
                            dataset,
                            metric[dataset]["dice"],
                            dataset,
                            best_metric[dataset],
                            best_metric_epoch[dataset],
                        )
                    )
                
                    
                    # calculate the average dice across all dataset used
                    total_av_dice.append(metric[dataset]["dice"])
                    
                    if wandb_active:

                        # wandb log
                        wandb.log(
                            {
                                "epoch_val": epoch + 1,
                                "mdice_" + dataset: metric[dataset]["dice"],
                                "sensitivity_" + dataset: metric[dataset]["sensitivity"],
                                "precision_" + dataset: metric[dataset]["precision"],
                                "mIOU_" + dataset: metric[dataset]["IOU"],

                            }
                        )
                # average dice across datasets
                if len(datasetlist)>1:
                    if np.mean(total_av_dice) > best_avg_dice:
                        best_avg_dice = np.mean(total_av_dice)
                        #save model
                        model_save_best_name = (
                            model_save_path
                            +    train_config.project_name
                            + "_random_drop_"
                            + str(randomly_drop)
                            + str(args.datasets)
                            + date
                            + "_BEST_AVERAGE.pth"
                        )
                        torch.save(model.state_dict(), model_save_best_name)
                        print(f"Saved new best average dice model, for {datasetlist}:_{best_avg_dice}")

                        print(best_avg_dice)

    if wandb_active:
        wandb.finish()
    # # save best checkpoint to 
    # if wandb_active:
    #     wandb.log({"Best_Checkpint_Path":model_save_best_name})

   

            
if __name__ == "__main__":


    # command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=int, default=0)
    parser.add_argument(
        "--datasets", help="datasets for training, using '_' to separate", type=str
    )
    parser.add_argument(
        "--k_fold", help="k_fold cross validation number fo folds", type=int, default=None
    )

 

    parser.add_argument(
        "--augmentation", help="augmentation", type=str, default=None
    )


    # Augmentation parameters
    parser.add_argument("--prob_brain_invert", type=float, default=None)
    parser.add_argument("--prob_brain_mixup", type=float, default=None)
    parser.add_argument("--prob_pathology_switch", type=float, default=None)
    parser.add_argument("--prob_pathology_invert", type=float, default=None)
    parser.add_argument("--prob_pathology_mixup", type=float, default=None)
    parser.add_argument("--prob_tumor_scale_shift", type=bool, default=False)
    parser.add_argument("--prob_brain_scale_shift", type=bool, default=False)
    parser.add_argument("--uniform_augs", type=bool, default=False)
    parser.add_argument("--uniform_scale_shift", type=int, default=None)
    parser.add_argument("--uniform_mix_up", type=float, default=None)
    parser.add_argument("--uniform_invert", type=float, default=None)



    #########################
    args = parser.parse_args()
    args.device_id = 1
    args.datasets =  'MSSEG'   #'ISLES2022'
    
    

  
    ######################################

    train_config = config.Training_config()
    database_config = config.Database_config()
    channels_copy = copy.deepcopy(database_config.channels)
    aug_config = config.Augmentation_config()


     # Override augmentation config with command line arguments
    if args.prob_brain_invert is not None:
        aug_config.prob_brain_invert = args.prob_brain_invert
        print(f"prob_brain_invert: {args.prob_brain_invert}")
    if args.prob_brain_mixup is not None:
        aug_config.prob_brain_mixup = args.prob_brain_mixup
        print(f"prob_brain_mixup: {args.prob_brain_mixup}")
    if args.prob_pathology_switch is not None:
        aug_config.prob_pathology_switch = args.prob_pathology_switch
        print(f"prob_pathology_switch: {args.prob_pathology_switch}")
    if args.prob_pathology_invert is not None:
        aug_config.prob_pathology_invert = args.prob_pathology_invert
        print(f"prob_pathology_invert: {args.prob_pathology_invert}")
    if args.prob_pathology_mixup is not None:
        aug_config.prob_pathology_mixup = args.prob_pathology_mixup
        print(f"prob_pathology_mixup: {args.prob_pathology_mixup}")
    if args.prob_tumor_scale_shift is not False:
        aug_config.prob_tumor_scale_shift = args.prob_tumor_scale_shift
        print(f"prob_tumor_scale_shift: {args.prob_tumor_scale_shift}")
    if args.prob_brain_scale_shift is not False:
        aug_config.prob_brain_scale_shift = args.prob_brain_scale_shift
        print(f"prob_brain_scale_shift: {args.prob_brain_scale_shift}")
    if args.uniform_augs is not False:
        aug_config.uniform_augs = args.uniform_augs
        print(f"uniform_augs: {args.uniform_augs}")
    if args.uniform_scale_shift is not None:
        aug_config.uniform_scale_shift = args.uniform_scale_shift
        print(f"uniform_scale_shift: {args.uniform_scale_shift}")
    if args.uniform_mix_up is not None:
        aug_config.uniform_mix_up = args.uniform_mix_up
        print(f"uniform_mix_up: {args.uniform_mix_up}")
    if args.uniform_invert is not None:
        aug_config.uniform_invert = args.uniform_invert
        print(f"uniform_invert: {args.uniform_invert}")

    #########

    print(aug_config)

   
    main(train_config,aug_config,database_config,k_fold=None,args = args,channels_copy = channels_copy)



  