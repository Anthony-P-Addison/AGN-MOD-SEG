import random
import numpy as np
import math 
import torch
from nets.multi_unet import res_unet as Unet
from itertools import combinations
import nibabel as nib
from monai.transforms import  Compose,EnsureChannelFirst
from monai.data import ImageDataset, DataLoader
from augment_utils import spatial_contrast_aug
from nets.agnostic_residual_block_identity import ResidualUnit_changed as ResidualUnit
from nets.agnostic_unet import Convolution
import torch.nn as nn
import os


def rand_assign_channels(
    dataset_modalities: list[int], total_modalities: list[str]
) -> list[int]:
    """Randomly assign channels to the batch"""
    num_to_assign = len(dataset_modalities)
    all_indices = list(range(len(total_modalities)))
    assigned_indices = random.sample(all_indices, num_to_assign)
    return assigned_indices


def map_channels(
    dataset_channels: list[str],
    total_modalities: list[str],
    rand_assign: bool,
) -> list[int]:
    """map specific dataset channels to total modalities
    dataset_channesl: list of modalities in dataset in question
    total_modalities: list of all possible modalities across all datasets used"""
    channel_map = []
    for channel in dataset_channels:
        if channel not in total_modalities:
            continue
            # raise ValueError(f"Modality for testing: '{channel}' is not in the list of trained modalities.")
        for index, modality in enumerate(total_modalities):
            if channel == modality:
                channel_map.append(index)

    if rand_assign:
        random.shuffle(channel_map)

    return channel_map


def rand_set_channels_to_zero(dataset_modalities: list, batch_img_data: torch.Tensor):
    """Randomly set a subset of channels to zero"""
    modalities_remaining=[]
    for i in range (batch_img_data.shape[0]):
        number_of_dropped_modalities = np.random.randint(0,len(dataset_modalities))
        modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)
        modalities_dropped.sort()
        batch_img_data[i,modalities_dropped,:,:,:] = 0.
        modalities_remaining.append(list(set(np.arange(len(dataset_modalities))) - set(modalities_dropped)))        
    return modalities_remaining, batch_img_data


def map_combinations(dataset_modalities: list[str], invar_ratio: float = 0.2) -> list[list[int]]:
    """Map all possible modality combinations of each dataset as a dict.
    optionaly boost the number of time invariatn channel trained on own (NOTE:modalities to keep)
    """

    all_combinations = []
    for r in range(1, len(dataset_modalities) + 1):
        combos = list(combinations(range(len(dataset_modalities)), r))
        # Convert each combination tuple to a list with no commas
        for combo in combos:
            all_combinations.append(list(combo))

    if len(all_combinations) > 1:
        num_combs = len(all_combinations)
        num_to_add = math.ceil((num_combs * invar_ratio))
        invar_added = len(dataset_modalities) - 1
        
        for _ in range(num_to_add-1):
            all_combinations.append([invar_added])

    return all_combinations


def rand_set_channels_to_zero_with_invar(
    dataset_modalities: list,
    batch_img_data: torch.Tensor,
    mask_data: torch.Tensor,
    domain_invariant: bool,
    contrast_augmentation: bool = False,
    batch_label_data: torch.Tensor = None,
    device_id: str = None,
    combination_map: list = None,
    augmentation_config = None,
) -> tuple[list[int], torch.Tensor]:
    """Randomly set channels to zero and handle agnostic channel with optional augmentations"""
    all_modalities_remaining = []
    all_modalities_dropped = []
    
    # Keep original data for augmentations
    original_batch = batch_img_data.clone()
    # Working copy for modifications
    working_batch = batch_img_data.clone()

    if domain_invariant:
        # append invariant channel
        working_batch = torch.cat((working_batch, torch.zeros((working_batch.shape[0], 1, working_batch.shape[2], working_batch.shape[3], working_batch.shape[4]))),dim=1)
        original_batch = torch.cat((original_batch, torch.zeros((original_batch.shape[0], 1, original_batch.shape[2], original_batch.shape[3], original_batch.shape[4]))),dim=1)
        
    for i in range(working_batch.shape[0]):   
    
        modalities_remaining = random.choice(combination_map)
        modalities_dropped = list(set(np.arange(len(dataset_modalities))) - set(modalities_remaining))

        # Apply dropout and verify
        working_batch[i,modalities_dropped,:,:,:] = 0
        
        # Verify dropped modalities are actually zero
        for mod in modalities_dropped:
            if not torch.all(working_batch[i,mod,:,:,:] == 0):
                print(f"Warning: Modality {mod} was not properly zeroed")
                working_batch[i,mod,:,:,:] = 0  # Force zero if not already zero
        
        # Handle aggnostic channel with augmentations
        if domain_invariant and len(dataset_modalities) > 2:
            invar = None
          
            if contrast_augmentation:
                random_number = random.random()
                
                # number below is probability of not using augmentations
                if random_number < 0.15:
                    invar = None   
                else:
                    pathology_label = batch_label_data[i,0,:,:,:]
                    brain_mask = mask_data[i,0,:,:,:] 

                    if len(modalities_dropped) > 0 and modalities_dropped[-1] == (len(dataset_modalities)-1):
                        invar = original_batch[i,modalities_dropped[-1],:,:,:]
                    elif len(modalities_dropped) > 0:
                        channel_add = random.sample(modalities_dropped, 1)
                        if original_batch[i,channel_add,:,:,:].any():
                            invar = spatial_contrast_aug(augmentation_config,channel_add,pathology_label,brain_mask,original_batch[i,:,:,:])
                
                    else:
                        # all modalities remaining. so need to place something in invar channel. 

                        channel_add = random.sample(modalities_remaining[:-1], 1)
                        invar = spatial_contrast_aug(augmentation_config,channel_add,pathology_label,brain_mask,original_batch[i,:,:,:])

        
            ######## TODO:ADD MORE AUGMENTATTIONS HERE WITH A NEW SWITCH  - FOR THE INVAR CHANNEL AS APPROPTIATE##########
        
            # just simple add of dropped modalities no augs. 
            if invar is None: 
               
                if len(modalities_dropped) > 0 and modalities_dropped[-1] == (len(dataset_modalities)-1):
                    invar = original_batch[i,modalities_dropped[-1],:,:,:]
                elif len(modalities_dropped) > 0:
                    channel_add = random.sample(modalities_dropped, 1)
                    invar = original_batch[i,channel_add,:,:,:]
                else:

                    channel_add = random.sample(modalities_remaining, 1)
                    if contrast_augmentation:
                        pathology_label = batch_label_data[i,0,:,:,:]
                        brain_mask = mask_data[i,0,:,:,:] 
                        invar = spatial_contrast_aug(augmentation_config,channel_add,pathology_label,brain_mask,original_batch[i,:,:,:])
                    
                    else:
                        invar = original_batch[i,channel_add,:,:,:]

            # Set invariant channel and verify dropped modalities are still zero
            working_batch[i,[len(dataset_modalities)-1],:,:,:] = invar
            
            # Final verification of dropped modalities
            for mod in modalities_dropped:
                if not torch.all(working_batch[i,mod,:,:,:] == 0):
                    print(f"Warning: Modality {mod} was not zero after invariant channel processing")
                    working_batch[i,mod,:,:,:] = 0  # Force zero if not already zero
            
        all_modalities_dropped.append(modalities_dropped)
        all_modalities_remaining.append(modalities_remaining)
    
    
    return all_modalities_dropped, all_modalities_remaining, working_batch


def single_slot( batch_img_data: torch.Tensor):
    """ randomly select one channel from input and place into single invariant slot"""
    batch_size, num_channels, height, width, depth = batch_img_data.shape
    # Initialize the output tensor with zeros
    batch_img_da = torch.zeros((batch_size, 1, height, width, depth), dtype=batch_img_data.dtype, device=batch_img_data.device)
    for i in range(batch_size):
        # Randomly select one channel to keep
        selected_channel = random.randint(0, num_channels - 1)
        # Copy the selected channel to the output tensor
        batch_img_da[i, 0, :, :, :] = batch_img_data[i, selected_channel, :, :, :]
    return batch_img_da,[selected_channel]


def create_net(model_file_path,model_net_type,model_modalities_trained_on, device,cuda_id):
    if model_net_type == "UNET":    
        model = Unet(in_channels=model_modalities_trained_on,out_channels=1).to(device)
        model.load_state_dict(torch.load(model_file_path, map_location={"cuda:0":cuda_id,"cuda:1":cuda_id}))
        model.eval()
    return model


def create_modality_combinations(modalities: list):

    modality_combinations = []
    for i in range(1, len(modalities) + 1):
        modality_combinations = modality_combinations + list(
            combinations(modalities, i)
        )
    return modality_combinations


def create_UNET_input(
    batch, modalities, dataset_name, model_modalities_trained_on, model_channel_map
):
    """Create input data for UNET model"""
    zeros_arr = np.zeros_like(batch[0])
    zeros_arr[:, modalities, :, :, :] = np.array(batch[0][:, modalities, :, :, :])
    batch[0] = torch.from_numpy(zeros_arr)
    input_data = torch.from_numpy(
        np.zeros(
            (
                1,
                model_modalities_trained_on,
                batch[0].shape[2],
                batch[0].shape[3],
                batch[0].shape[4],
            ),
            dtype=np.float32,
        )
    )
    input_data[:, model_channel_map[dataset_name], :, :, :] = batch[0][
        :, range(0, batch[0].shape[1]), :, :, :
    ]
    # batch[0][:,range(0,batch[0].shape[1]),:,:,:]
    return input_data


def create_single_channel_UNET_input(
    batch, modalities, dataset_name, model_modalities_trained_on, model_channel_map
):
    """Create single channel input data for UNET model"""
    if len(modalities) != 1:
        raise ValueError(
            "Single slot is selected, but more than one modality is provided"
        )
    input_data = batch[0][:, modalities, :, :, :]
    return input_data


def save_nifti(tensor: torch.Tensor, file_path: str, affine):
    vars_numpy = tensor[0].cpu().detach().numpy()
    vars_numpy = np.squeeze(vars_numpy)
    # vars_numpy = np.transpose(vars_numpy,(1,2,3,0))
    new_image = nib.Nifti1Image(vars_numpy, affine=affine.squeeze())
    # ensure file path exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    nib.save(new_image, file_path)


####### Randomly initiating Agnostic channel and Agnostic channel + Agnostic pathway #######

def add_invar_input_to_pre_trained(model: dict) -> dict:
    """Add additional input channel (randomly initialised weights) to the first layer of a pre trained model"""

    # input size of first layer
    input_size = model["conv_1.conv.unit0.conv.weight"].shape[1]

    print(f"Input channels of model: {input_size}")
    # additinal input channel
    num_input_channels = input_size + 1
    # create a new Conv3d layer with an additional input channel
    # new_conv = torch.nn.Conv3d(num_input_channels, 7, kernel_size=3, stride=1, padding=1, bias=True)
    new_conv1 = ResidualUnit(
        spatial_dims=3,
        in_channels=num_input_channels,
        out_channels=8,
        strides=1,
        kernel_size=3,
        subunits=1,
        dropout=0.2,
    )
    new_down_conv1 = Convolution(
        spatial_dims=3,
        in_channels=8,
        out_channels=32,
        strides=1,
        kernel_size=3,
        dropout=0.2,
    )

    # copy the weights from the old conv layer to the new conv layer
    with torch.no_grad():
        new_conv1.conv[0].conv.weight[:input_size, :input_size, :, :, :] = model[
            "conv_1.conv.unit0.conv.weight"
        ]
        new_conv1.conv[0].conv.bias[:input_size] = torch.nn.Parameter(
            model["conv_1.conv.unit0.conv.bias"]
        )  # Convert to torch.nn.Parameter
        new_down_conv1.conv.weight[:, :input_size, :, :, :] = model[
            "down_conv_1.conv.weight"
        ]
        new_down_conv1.conv.bias[:] = torch.nn.Parameter(
            model["down_conv_1.conv.bias"]
        )  # Convert to torch.nn.Parameter
    # replace the old conv layer with the new conv layer in the model
    model["conv_1.conv.unit0.conv.weight"] = new_conv1.conv[0].conv.weight
    model["conv_1.conv.unit0.conv.bias"] = new_conv1.conv[0].conv.bias
    model["down_conv_1.conv.weight"] = new_down_conv1.conv.weight
    model["down_conv_1.conv.bias"] = new_down_conv1.conv.bias

    print(
        f'Input channels of model after update: {model["conv_1.conv.unit0.conv.weight"].shape[1]}'
    )
    print(
        f'Input channels of model after update: {model["conv_1.conv.unit0.conv.weight"][1][4]}'
    )

    return model


def add_invar_layers_to_pre_trained(
    model: dict, invariant_out_channels: int = 8, dropout: float = 0.2
) -> dict:
    """Modify model dict to match unet_deep architecture with invariant channels enabled: used for finetuning the model with designated agnostic channel"""

    # Get original layer dimensions (includes invariant channel)
    original_input_size = model["conv_1.conv.unit0.conv.weight"].shape[1]
    # conv1_out_channels = model["conv_1.conv.unit0.conv.weight"].shape[0]
    # down_conv1_out_channels = model["down_conv_1.conv.weight"].shape[0]

    # In unet_deep: modality_channels = in_channels - 1 (excluding invariant channel)

    modality_channels = original_input_size

    print(f"Original input channels: {original_input_size}")
    print(f"Modality channels : {modality_channels}")

    print(f"Converting to unet_deep architecture...")

    # Create invariant stream exactly like unet_deep
    invariant_stream = nn.Sequential(
        ResidualUnit(
            spatial_dims=3,
            in_channels=1,
            out_channels=8,
            strides=1,
            kernel_size=3,
            subunits=1,
            dropout=dropout,
        ),
        Convolution(
            spatial_dims=3,
            in_channels=8,
            out_channels=16,
            strides=1,
            kernel_size=3,
            dropout=0.2,
        ),
        Convolution(
            spatial_dims=3,
            in_channels=16,
            out_channels=invariant_out_channels,
            strides=1,
            kernel_size=3,
            dropout=0.2,
        ),
    )

    # Create new down_conv_1 for concatenated features (modality_features + invariant_features)
    downstream_in_channels = modality_channels + invariant_out_channels
    new_down_conv1 = Convolution(
        spatial_dims=3,
        in_channels=downstream_in_channels,
        out_channels=32,
        strides=2,
        kernel_size=3,
        dropout=dropout,
    )

    with torch.no_grad():
        # Add invariant stream layers to model state dict

        new_down_conv1.conv.weight[:, :modality_channels, :, :, :] = model[
            "down_conv_1.conv.weight"
        ]
        new_down_conv1.conv.bias[:] = torch.nn.Parameter(model["down_conv_1.conv.bias"])

        model["invariant_stream.0.conv.unit0.conv.weight"] = (
            invariant_stream[0].conv[0].conv.weight
        )
        model["invariant_stream.0.conv.unit0.conv.bias"] = (
            invariant_stream[0].conv[0].conv.bias
        )
        # Add the specific ADN component mentioned in error

        model["invariant_stream.0.conv.unit0.adn.A.weight"] = (
            invariant_stream[0].conv[0].adn.A.weight
        )

        # Layer 1: Convolution (8 -> 16) - add only required parameters
        model["invariant_stream.1.conv.weight"] = invariant_stream[1].conv.weight
        model["invariant_stream.1.conv.bias"] = invariant_stream[1].conv.bias
        # Add the specific ADN component mentioned in error

        model["invariant_stream.1.adn.A.weight"] = invariant_stream[1].adn.A.weight

        # Layer 2: Convolution (16 -> invariant_out_channels) - add only required parameters
        model["invariant_stream.2.conv.weight"] = invariant_stream[2].conv.weight
        model["invariant_stream.2.conv.bias"] = invariant_stream[2].conv.bias
        # Add the specific ADN component mentioned in error

        model["invariant_stream.2.adn.A.weight"] = invariant_stream[2].adn.A.weight

        # Update down_conv_1 to accept concatenated features (modality_channels + invariant_out_channels -> 32)
        # Copy existing weights for the modality channels part
        new_down_conv1.conv.weight[:, :modality_channels, :, :, :] = model[
            "down_conv_1.conv.weight"
        ][:, :modality_channels, :, :, :]
        # The additional invariant channels (last 8 channels) will be randomly initialized

        model["down_conv_1.conv.weight"] = new_down_conv1.conv.weight
        model["down_conv_1.conv.bias"] = new_down_conv1.conv.bias

    print(
        f"✓ Kept conv_1 unchanged: {modality_channels} -> {modality_channels} (modality channels)"
    )
    print(f"✓ Added invariant_stream: 1 -> 8 -> 16 -> {invariant_out_channels}")
    print(
        f"✓ Updated down_conv_1: {downstream_in_channels} -> 32 (concatenated features)"
    )
    print(
        f"✓ Architecture flow: modalities({modality_channels}) -> conv_1({modality_channels}) \\"
    )
    print(
        f"                                                                        concat({downstream_in_channels}) -> down_conv_1(32)"
    )
    print(
        f"                      invariant(1) -> invariant_stream({invariant_out_channels}) /"
    )

    return model




def lr_schedule(epochs, optimizer):
    """Learning rate scheduler function that implements warmup and decay.
    
    Args:
        current_epoch (int): Current training epoch
        epochs (int): Total number of epochs
        optimizer (torch.optim.Optimizer): The optimizer to create scheduler for
        
    Returns:
        scheduler: LambdaLR scheduler
    """
    def lr_lambda(current_epoch):
        # Warm up phase (first 50 epochs)
        if current_epoch < 50:
            return (float(current_epoch) + 1) / float(max(1, 50))
        # Constant learning rate phase
        elif 50 <= current_epoch <= 350:
            return 1.0
        # First decay phase
        elif 350 < current_epoch <= 400:
            return 0.4
        # Second decay phase
        elif 400 < current_epoch <= 450:
            return 0.1
        # Final decay phase
        else:
            return max(0.0, 0.1 - (0.1 * (current_epoch - 450) / float(max(1, epochs - 450))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler





def ensemble_across_modalities(prediction_dict, threshold=0.5):
    """
    Ensembles predictions across modalities for each sample index when training single slot model.

    Args:
        prediction_dict (dict): keys are modalities, values are lists of predictions (length N).
        threshold (float): threshold for binarization.

    Returns:
        list: ensembled predictions for each sample index.
    """
    num_samples = len(next(iter(prediction_dict.values())))
    modalities = list(prediction_dict.keys())
    ensembled_predictions = []

    for idx in range(num_samples):
        # Collect predictions for this sample from all modalities
        preds = [prediction_dict[mod][idx] for mod in modalities]
        # Stack: shape [num_modalities, ...]
        stacked = torch.stack(preds, dim=0)
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probabilities = torch.sigmoid(stacked)
        # Average probabilities across modalities
        mean_prob = torch.mean(probabilities, dim=0)
        # Final threshold to get binary prediction
        final_pred = (mean_prob > threshold).float()
        ensembled_predictions.append(final_pred)
        
        # Clear intermediate tensors
        del preds, stacked, probabilities, mean_prob, final_pred
    
    return ensembled_predictions


def calculate_dice_scores(predictions, labels, device):
    """
    Calculates Dice scores for each patient for SINGLE CHANNEL MODELS
    """
    from monai.metrics import DiceMetric
    
    # Create DiceMetric once outside the loop for efficiency
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_scores = []
    
    # Loop through each prediction and label pair
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # Move to GPU for calculation
        pred = pred.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)
        
        # Calculate Dice for this sample
        dice_score = dice_metric(pred.unsqueeze(0), label.unsqueeze(0))
        dice_scores.append(dice_score.item())
        
        # Reset metric for next calculation
        dice_metric.reset()
        
        # Clear from GPU
        del pred, label
        
    return np.array(dice_scores)



def load_test_checkpoints(model_name:str,checkpoint_own:str):
    """load checkpoints either  models trained by author or own checkpoints"""
    import json 

    if model_name != 'own_checkpoint':
        with open('checkpoint_paths.json', 'r') as f:
            checkpoint_paths = json.load(f)
        
        if model_name in checkpoint_paths:
            checkpoint = checkpoint_paths[model_name]['path']
            print(f"Testing {model_name}: {checkpoint_paths[model_name]['description']}")

        else:
            print(f"Model '{model_name}' not found in checkpoint_paths.json")
            print("Available models:")
            for name in checkpoint_paths.keys():
                print(f"- {name}")

    ## load own checkpoint ##
    elif model_name =='own_checkpoint':
        checkpoint = checkpoint_own
        if checkpoint is None:
            raise ValueError("No checkpoint path provided. Please provide a valid path to a model checkpoint file when using 'own_checkpoint' mode.")
    return checkpoint


