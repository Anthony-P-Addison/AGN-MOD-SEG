import torch
from config import Training_config, Database_config
#from augment_data import get_dataloader
import random
import copy




#### just put all the training data from every modaility into a single list ###
def collect_train_data(train_loaders,datasetlist,data_loader_map):
    train_data = []
    for dataset in datasetlist:
        for batch_data in zip(*train_loaders):
            loader_index = data_loader_map[dataset]
            batch_data = batch_data[loader_index]
            img, label = batch_data
            train_data.append((img, label))
    
    shuffle_data = random.sample(train_data, len(train_data))
    print("total_train_data_len:", len(train_data))
    return shuffle_data


## seperate all data into a single modality ##
def separate_into_single_modality(train_data):
    separated_data = []
    
    for img, label in train_data:
        for idx in range (img.shape[0]):
           for i in range (img.shape[1]):
                if i < img.shape[1]:  # Ensure the index is within bounds
                    
                    separated_data.append((torch.unsqueeze(img[idx, i, :, :, :],0), label[idx, :, :, :, :]))
                    
               
    return separated_data


def batch_data(separated_data, batch_size):
    batched_data = []
    separated_data = random.sample(separated_data, len(separated_data))
    for i in range(0, len(separated_data), batch_size):
        batch = separated_data[i:i+batch_size]
        data_batch = torch.cat([item[0] for item in batch], dim=0)
        label_batch = torch.cat([item[1] for item in batch], dim=0)
        batched_data.append((data_batch, label_batch))
        
    return batched_data






if __name__ == "__main__":

    train_config = Training_config
    database_config = Database_config
    datasetlist = ["BRATS","WMH"]
    cropped_input_size = [128, 128, 128]
    k_fold = None
    channels_copy = copy.deepcopy(database_config.channels)

    # data_size
    train_size = database_config.train_size
    data_size = 10
    # data_size = min(train_size[dataset] for dataset in datasetlist)

    # load data

    # train_loaders, val_loader, data_loader_map = get_dataloader(
    #     train_config,
    #     database_config,
    #     datasetlist,
    #     cropped_input_size,
    #     data_size,
    #     channels_copy,
    #     k_fold,)

    train_data = collect_train_data(train_loaders)

    separated_data = separate_into_single_modality(train_data)

    print("total_separated_data_len:", len(separated_data))