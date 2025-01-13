import random
import numpy as np
import torch
from nets.unet import Unet
from itertools import combinations
import nibabel as nib


def map_channels(dataset_channels: list[str], total_modalities: list[str],rand_assign: bool,) -> list[int]:
    """map specific dataset channels to total modalities
    dataset_channesl: list of modalities in dataset in question
    total_modalities: list of all possible modalities across all datasets used """
    channel_map = []
    for channel in dataset_channels:
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


def rand_set_channels_to_zero_with_invar(dataset_modalities: list, batch_img_data: torch.Tensor, domain_invariant:bool) -> tuple[list[int], torch.Tensor]:
    """Randomly set a subset of channels to zero
    return a list of modalities remaining(after drop) and an image tensors from remaining channels """
    modalities_remaining=[]
    
    if domain_invariant:
        # append a new channel to dimension 1
        batch_img_da = torch.cat((batch_img_data, torch.zeros((batch_img_data.shape[0], 1, 128, 128, 128))), dim=1)
        print(f'Batch size after additional slot added: {batch_img_da.size()}')
    else:
        batch_img_da = batch_img_data

    for i in range (batch_img_da.shape[0]):   

        if domain_invariant:
            # start from 1 dropped as added modality is zeros so is effectively dropped. 
            number_of_dropped_modalities = np.random.randint(1,len(dataset_modalities))
        else:
            number_of_dropped_modalities = np.random.randint(0,len(dataset_modalities))    
        
        modalities_dropped = random.sample(list(np.arange(len(dataset_modalities))), number_of_dropped_modalities)    
        modalities_dropped.sort()
        
        # copy of batch image data
        batch_img = batch_img_da.clone()

        # multiplied dropped channels by zero. 
        batch_img_da[i,modalities_dropped,:,:,:] = 0
        modalities_remaining.append(list(set(np.arange(len(dataset_modalities))) - set(modalities_dropped)))   
        print(f'modalities_remaining: {modalities_remaining}')

        if domain_invariant:
            
            # if invar channel in modalities dropped, then always drop this channel so has same oribability of other channels being dropped. 
            if modalities_dropped[-1] == len(dataset_modalities)-1:
                batch_img_da = batch_img_da
            
            else:
                channel_add = random.sample(modalities_dropped, 1)
                
                #select channel to be added to invariant slot
                invar = batch_img[i,channel_add,:,:,:]  
                #print(f' channel being added {batch_img[i,channel_add,:,:,:]} ' )
                invar = torch.unsqueeze(invar,1)
                batch_img_da[i,[len(dataset_modalities)-1],:,:,:]=invar

                # double check that the invariant slot is the same as the selected dropped channel
                # result = torch.allclose(batch_img_da[i][-1],batch_img[i][channel_add])
                #print(f"Assert selected dropped channel is coancatenated: {result}")
                          
    return modalities_remaining, batch_img_da


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



def one_slot():
    """all data through the same slot"""
    #TODO: implemeent this 
    return 0 




# def swap_slot_for_invar(slot_dropped:str,channels:):
#     """swap removed modality  train time for invariant slot at test time"""
    



if __name__ == "__main__":
    #test functions here 

    # single slot test 
    
    single_slot(torch.rand((2, 4, 128, 128, 128)))