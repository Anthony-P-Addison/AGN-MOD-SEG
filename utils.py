import random
import numpy as np
import torch
from nets.unet import res_unet as Unet
from itertools import combinations
import nibabel as nib
from monai.transforms import  Compose,EnsureChannelFirst
from monai.data import ImageDataset, DataLoader
from augment_utils import mixup1_augmentation, mixup_data_causality


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


def rand_set_channels_to_zero_with_invar(
    dataset_modalities: list,
    batch_img_data: torch.Tensor,
    domain_invariant: bool,
    mixup: bool = False,
    gin_mix: bool = False,
    gin_ipa: str = None,
    batch_label_data: torch.Tensor = None,
    device_id: str = None,
) -> tuple[list[int], torch.Tensor]:
    """Randomly set channels to zero and handle invariant channel with optional augmentations
    Args:
        dataset_modalities: List of modalities in the dataset
        batch_img_data: Input image batch [B, C, H, W, D]
        domain_invariant: Whether to use invariant channel
        mixup: Whether to use mixup augmentation
        gin_mix: Whether to use GIN augmentation
        gin_ipa: Type of GIN/IPA augmentation
        batch_label_data: Labels for supervised augmentation
        device_id: Device for GIN augmentation
    Returns:
        tuple: (list of remaining modalities, augmented batch)
    """
    modalities_remain = []
    
   
    batch_img_da = batch_img_data

    for i in range(batch_img_da.shape[0]):   
        # Uniform dropout probability for all cases
        number_of_dropped_modalities = np.random.randint(0, len(dataset_modalities)-1) # okay as invar appended already
        
        # Random sampling for all dataset sizes
        
        modalities_dropped = random.sample(
            list(np.arange(len(dataset_modalities)-1)),
            number_of_dropped_modalities,
        )
        modalities_dropped.sort()
        
        # Apply dropout
        batch_img = batch_img_da.clone()
        batch_img_da[i,modalities_dropped,:,:,:] = 0
        
        modalities_remaining = sorted(
            set(np.arange(len(dataset_modalities)-1)) - set(modalities_dropped)
        )

        if domain_invariant:
            # append invariant channel
            batch_img_da = torch.cat((batch_img_data, torch.zeros((batch_img_data.shape[0], 1, 128,128,128))),dim=1)
    

        # Handle invariant channel with augmentations
        if domain_invariant and random.random() < 0.375 and len(dataset_modalities) > 2:  # Match average modality presence can change to how often want to add info to invariant channel
            invar = None

          
            if mixup:
                # Mixup augmentation logic
                if len(modalities_dropped) == 0 and len(modalities_remaining) >= 2:
                    # Mix two remaining channels
                    channels = random.sample(modalities_remaining, 2)
                    invar = mixup1_augmentation(
                        batch_img[i, channels, :, :],
                        all_mod_dropped=False,
                        one_mod_dropped=False,
                        two_not_dropped=True,
                        mod_3=False
                    )
                elif len(modalities_dropped) == 1 and len(modalities_remaining) >= 1:
                    # Mix one dropped with one remaining
                    channels = random.sample(modalities_remaining, 1) + random.sample(modalities_dropped, 1)
                    invar = mixup1_augmentation(
                        batch_img[i, channels, :, :],
                        all_mod_dropped=False,
                        one_mod_dropped=True,
                        two_not_dropped=False,
                        mod_3=False
                    )
                elif len(modalities_dropped) == 2:
                    # Mix two dropped channels
                    channels = random.sample(modalities_dropped, 2)
                    invar = mixup1_augmentation(
                        batch_img[i, channels, :, :],
                        all_mod_dropped=True,
                        one_mod_dropped=False,
                        two_not_dropped=False,
                        mod_3=False
                    )
                elif len(modalities_dropped) > 2:
                    # Mix three dropped channels
                    channels = random.sample(modalities_dropped, 3)
                    invar = mixup1_augmentation(
                        batch_img[i, channels, :, :],
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
                        batch_img[i, channel, :, :],
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
                        batch_img[i, channel, :, :],
                        device_id=device_id,
                        aug_type=gin_ipa,
                        all_mod_dropped=True,
                        one_mod_dropped=False,
                        two_not_dropped=False,
                        mod_3=False
                )
                    
            ######## TODO:ADD MORE AUGMENTATTIONS HERE WITH A NEW SWITCH FOR THE INVAR CHANNEL##########
        
            # If no augmentation or augmentation failed, use simple channel copy (of droppepd_channels)
            elif invar is None: 
                if len(modalities_dropped) > 0:
                    channel_add = random.sample(modalities_dropped, 1)
                    invar = batch_img[i,channel_add,:,:,:]
                else:
                    channel_add = random.sample(modalities_remaining, 1)
                    invar = batch_img[i,channel_add,:,:,:]
            batch_img_da[i,[len(dataset_modalities)-1],:,:,:] = invar.to(torch.device("cpu"))
        
        modalities_remain.append(modalities_remaining)

    
    return modalities_remain, batch_img_da


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
    vars_numpy = tensor.cpu().detach().numpy()
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
