import random
import numpy as np
import math 
import torch
from nets.unet import res_unet as Unet
from itertools import combinations
import nibabel as nib
from monai.transforms import  Compose,EnsureChannelFirst
from monai.data import ImageDataset, DataLoader
from augment_utils import mixup1_augmentation, mixup_data_causality, spatial_contrast_aug


def map_channels(dataset_channels: list[str], total_modalities: list[str],rand_assign: bool,) -> list[int]:
    """map specific dataset channels to total modalities
    dataset_channesl: list of modalities in dataset in question
    total_modalities: list of all possible modalities across all datasets used """
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
    mixup: bool = False,
    gin_mix: bool = False,
    gin_ipa: str = None,
    contrast_augmentation: bool = False,
    batch_label_data: torch.Tensor = None,
    device_id: str = None,
    combination_map: list = None,
    augmentation_config = None,
) -> tuple[list[int], torch.Tensor]:
    """Randomly set channels to zero and handle invariant channel with optional augmentations"""
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
        # Uniform dropout probability for all cases
        # number_of_dropped_modalities = np.random.randint(0, len(dataset_modalities))
        # modalities_dropped = random.sample(
        #     list(np.arange(len(dataset_modalities))),
        #     number_of_dropped_modalities,
        # )
        # modalities_dropped.sort()

        # provide evey combination of modalities and then randomly select when training.
        # want to ensure invariatn channel is trained on more times for each combination. 
        modalities_remaining = random.choice(combination_map)
        modalities_dropped = list(set(np.arange(len(dataset_modalities))) - set(modalities_remaining))

        # Apply dropout and verify
        working_batch[i,modalities_dropped,:,:,:] = 0
        
        # Verify dropped modalities are actually zero
        for mod in modalities_dropped:
            if not torch.all(working_batch[i,mod,:,:,:] == 0):
                print(f"Warning: Modality {mod} was not properly zeroed")
                working_batch[i,mod,:,:,:] = 0  # Force zero if not already zero
        
        # modalities_remaining = sorted(
        #     set(np.arange(len(dataset_modalities))) - set(modalities_dropped))
     
        # Handle invariant channel with augmentations
        if domain_invariant and len(dataset_modalities) > 2:
            invar = None
          
            if mixup:
                # Mixup augmentation logic
                if len(modalities_dropped) == 0 and len(modalities_remaining) >= 2:
                    # Mix two remaining channels
                    channels = random.sample(modalities_remaining, 2)
                    invar = mixup1_augmentation(
                        original_batch[i, channels, :, :],
                        all_mod_dropped=False,
                        one_mod_dropped=False,
                        two_not_dropped=True,
                        mod_3=False
                    )
                elif len(modalities_dropped) == 1 and len(modalities_remaining) >= 1:
                    # Mix one dropped with one remaining
                    channels = random.sample(modalities_remaining, 1) + random.sample(modalities_dropped, 1)
                    invar = mixup1_augmentation(
                        original_batch[i, channels, :, :],
                        all_mod_dropped=False,
                        one_mod_dropped=True,
                        two_not_dropped=False,
                        mod_3=False
                    )
                elif len(modalities_dropped) == 2:
                    # Mix two dropped channels
                    channels = random.sample(modalities_dropped, 2)
                    invar = mixup1_augmentation(
                        original_batch[i, channels, :, :],
                        all_mod_dropped=True,
                        one_mod_dropped=False,
                        two_not_dropped=False,
                        mod_3=False
                    )
                elif len(modalities_dropped) > 2:
                    # Mix three dropped channels
                    channels = random.sample(modalities_dropped, 3)
                    invar = mixup1_augmentation(
                        original_batch[i, channels, :, :],
                        all_mod_dropped=False,
                        one_mod_dropped=False,
                        two_not_dropped=False,
                        mod_3=True
                    )
                    
            elif gin_mix:
                # GIN augmentation logic
                if len(modalities_dropped) == 0 and len(modalities_remaining) >= 1:
                    # Augment one remaining channel
                    channel = random.sample(modalities_remaining, 1)
                    invar = mixup_data_causality(
                        original_batch[i, channel, :, :],
                        device_id=device_id,
                        aug_type=gin_ipa,
                        all_mod_dropped=False,
                        one_mod_dropped=False,
                        two_not_dropped=True,
                        mod_3=False
                    )
                elif len(modalities_dropped) >= 1:
                    # Augment one dropped channel
                    channel = random.sample(modalities_dropped, 1)
                    invar = mixup_data_causality(
                        original_batch[i, channel, :, :],
                        device_id=device_id,
                        aug_type=gin_ipa,
                        all_mod_dropped=True,
                        one_mod_dropped=False,
                        two_not_dropped=False,
                        mod_3=False
                    )
                    
            elif contrast_augmentation:
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
    ##### --- Track modalities used for training (not dropped) ---##########

    # # Initialize counters if they don't exist
    # if not hasattr(rand_set_channels_to_zero_with_invar, 'modality_used_counter'):
    #     rand_set_channels_to_zero_with_invar.modality_used_counter = {}
    #     rand_set_channels_to_zero_with_invar.total_iterations = 0

    # if not hasattr(rand_set_channels_to_zero_with_invar, 'combination_used_counter'):
    #     rand_set_channels_to_zero_with_invar.combination_used_counter = {}

    # # Increment total iterations
    # rand_set_channels_to_zero_with_invar.total_iterations += 1

    # # Count used modalities and combinations
    # for i, modalities in enumerate(modalities_remain):
    #     # Individual modality usage
    #     for mod_idx in modalities:
    #         mod_name = dataset_modalities[mod_idx]
    #         if mod_name not in rand_set_channels_to_zero_with_invar.modality_used_counter:
    #             rand_set_channels_to_zero_with_invar.modality_used_counter[mod_name] = 0
    #         rand_set_channels_to_zero_with_invar.modality_used_counter[mod_name] += 1

    #     # Combination usage
    #     used_mods = [dataset_modalities[j] for j in sorted(modalities)]
    #     combo_str = '+'.join(used_mods) if used_mods else 'None'
    #     if combo_str not in rand_set_channels_to_zero_with_invar.combination_used_counter:
    #         rand_set_channels_to_zero_with_invar.combination_used_counter[combo_str] = 0
    #     rand_set_channels_to_zero_with_invar.combination_used_counter[combo_str] += 1

    # # Print statistics every 10 iterations
    # if rand_set_channels_to_zero_with_invar.total_iterations % 30 == 0:
    #     total_samples = rand_set_channels_to_zero_with_invar.total_iterations * batch_img_data.shape[0]

    #     print(f"\n--- Modality Used Statistics (after {rand_set_channels_to_zero_with_invar.total_iterations} iterations) ---")
    #     for mod_name, count in rand_set_channels_to_zero_with_invar.modality_used_counter.items():
    #         use_percentage = (count / total_samples) * 100
    #         print(f"  {mod_name}: used {count} times ({use_percentage:.2f}%)")
    #     print("---------------------------------------------------\n")

    #     print(f"--- Used Modality Combinations (after {rand_set_channels_to_zero_with_invar.total_iterations} iterations) ---")
    #     print(f"{'Combination':<30} | {'Count':<10}")
    #     print("-" * 45)
    #     for combo, count in sorted(rand_set_channels_to_zero_with_invar.combination_used_counter.items(), key=lambda x: -x[1]):
    #         print(f"{combo:<30} | {count:<10}")
    #     print("---------------------------------------------------\n")

    #############################################################################
    
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
    for i in range(1,len(modalities)+1):   
      modality_combinations = modality_combinations + list(combinations(modalities,i))
    return modality_combinations


def create_UNET_input(batch, modalities, dataset_name,model_modalities_trained_on,model_channel_map):
    """Create input data for UNET model"""   
    zeros_arr = np.zeros_like(batch[0])
    zeros_arr[:,modalities,:,:,:] = np.array(batch[0][:,modalities,:,:,:])
    batch[0] = torch.from_numpy(zeros_arr)
    input_data = torch.from_numpy(np.zeros((1,model_modalities_trained_on,batch[0].shape[2],batch[0].shape[3],batch[0].shape[4]),dtype=np.float32))
    input_data[:,model_channel_map[dataset_name],:,:,:] = batch[0][:,range(0,batch[0].shape[1]),:,:,:]
    #batch[0][:,range(0,batch[0].shape[1]),:,:,:]
    return input_data

def create_single_channel_UNET_input(batch, modalities, dataset_name,model_modalities_trained_on,model_channel_map):
    """Create single channel input data for UNET model"""
    if len(modalities) != 1:
        raise ValueError("Single slot is selected, but more than one modality is provided")
    input_data = batch[0][:,modalities,:,:,:]
    return input_data


# def create_UNET_input(batch, modalities, dataset_name, model_modalities_trained_on, model_channel_map):
#     """Create input data for UNET model"""
#     # Initialize input_data tensor with zeros
#     input_data = torch.from_numpy(np.zeros((1, model_modalities_trained_on, batch[0].shape[2], batch[0].shape[3], batch[0].shape[4]), dtype=np.float32))

#     # Copy data into input_data based on model_channel_map
#     input_data[:, model_channel_map[dataset_name], :, :, :] = batch[0][:, modalities, :, :, :]


#     return input_data


def create_UNET_input_quicktest(batch, modalities, channel_map, model_modalities_trained_on):
    zeros_arr = np.zeros_like(batch[0])
    zeros_arr[:,modalities,:,:,:] = np.array(batch[0][:,modalities,:,:,:])
    batch[0] = torch.from_numpy(zeros_arr)
    input_data = torch.from_numpy(np.zeros((1,model_modalities_trained_on,batch[0].shape[2],batch[0].shape[3],batch[0].shape[4]),dtype=np.float32))
    input_data[:,channel_map,:,:,:] = batch[0][:,range(0,batch[0].shape[1]),:,:,:]
    return input_data

def save_nifti(tensor: torch.Tensor, file_path: str,affine):
    vars_numpy = tensor[0].cpu().detach().numpy()
    vars_numpy = np.squeeze(vars_numpy)
    # vars_numpy = np.transpose(vars_numpy,(1,2,3,0))    
    new_image = nib.Nifti1Image(vars_numpy,affine=affine.squeeze())       
    nib.save(new_image, file_path)

#######################

def modality_select_invar(dataset: list, modality: str) -> list:
    """Select a modality to be inserted into the invar channel at test time"""

    updated_modalities = ["invar" if m == modality else m for m in dataset]
        
    return updated_modalities


def add_input_to_pre_trained(model:dict)-> dict:

    """Add additional input channel to the first layer of a pre trained model"""

    #input size of first layer
    input_size = model['conv_1.conv.unit0.conv.weight'].shape[1]
    print(f'Input channels before update: {model["conv_1.conv.unit0.conv.weight"][1][3]}')
    print(f'Input channels of model: {input_size}')
    # additinal input channel 
    num_input_channels = input_size + 1
    # create a new Conv3d layer with an additional input channel
    new_conv = torch.nn.Conv3d(num_input_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
    # copy the weights from the old conv layer to the new conv layer
    with torch.no_grad():
        new_conv.weight[:, :input_size, :, :, :] = model['conv_1.conv.unit0.conv.weight']
        new_conv.bias = torch.nn.Parameter(model['conv_1.conv.unit0.conv.bias'])  # Convert to torch.nn.Parameter
    # replace the old conv layer with the new conv layer in the model
    model['conv_1.conv.unit0.conv.weight'] = new_conv.weight
    model['conv_1.conv.unit0.conv.bias'] = new_conv.bias

    print(f'Input channels of model after update: {model["conv_1.conv.unit0.conv.weight"].shape[1]}')
    print(f'Input channels of model after update: {model["conv_1.conv.unit0.conv.weight"][1][4]}')

    return model


def create_test_val_loader(
    val_size: int,
    images,
    segs,
    workers,
    image_only: bool = False,
):
    """Create monai wrapped dataloaders for training and validation data"""

    # image augmentation through spatial cropping to size and by randomly rotating

    val_imtrans = Compose([EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])
    # create a training data loader

    # create a validation data loader
    val_ds = ImageDataset(
        images[-val_size:],
        segs[-val_size:],
        transform=val_imtrans,
        seg_transform=val_segtrans,
        image_only=image_only,
    )
    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    return val_loader


if __name__ == "__main__":
    #test functions here 

    # single slot test 
    
    single_slot(torch.rand((2, 4, 128, 128, 128)))
