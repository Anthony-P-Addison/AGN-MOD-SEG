from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from config import Database_config, Training_config
from dataloader import get_dataloader
import copy

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

####
from VAE.VAE import VAE, VAE_config, ConvVAE
from VAE.dataloader import collect_train_data, separate_into_single_modality, batch_data


#############################################
cuda_id = 0
device = torch.device(f"cuda:{cuda_id}")
torch.cuda.set_device(cuda_id)

# Define the loss function
def vae_loss(recon_x, x, z_mean, z_log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_divergence


# Training function
def train_vae(data, num_epochs):

    model = "yo"
    # Model
    if model == "VAE":
        config = VAE_config()
        vae = VAE(config).to(device)
    
    else:
        config = VAE_config()
        vae = ConvVAE(config).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    #Define a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    


    for epoch in range(num_epochs):


        epoch_loss = 0

        for batch in tqdm(data, desc=f"Training VAE for  epoch {epoch}"):

            # batch data in following format tuple([data,label],[data,label]) (for batch size of 2)
            img_index = 0
            label_index = 1
            # flatten tensor here so dont need to do twice
            if model == "VAE":
                input_data = batch[img_index].view(batch[img_index].size(0), -1)
            else:
                input_data = batch[img_index]
            input_data1 = input_data.to(device)
            input_data1 = input_data1.unsqueeze(1)
            recon_x, z_mean, z_log_var = vae(input_data1)

            optimizer.zero_grad()
            loss = vae_loss(recon_x, input_data1, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            scheduler.step()
    

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data)}")

        if epoch%20  == 0:
            # save_model
            folder_path = 'VAE/checkpoints/'
            torch.save(vae.state_dict(), folder_path +"vaeee.pth")



if __name__ == "__main__":


    # batch_size = 2
    # instantiate the variables and dataloader


    # load up el data
    train_config = Training_config()
    database_config = Database_config()
    datasetlist = ["WMH",'ATLAS']
    cropped_input_size = [128, 128, 128]
    k_fold = None
    channels_copy = copy.deepcopy(database_config.channels)
    batch_size = 2

    # data_size
    train_size = database_config.train_size
    data_size = 10
    #data_size = max(train_size[dataset] for dataset in datasetlist)

    # load data

    train_loaders, val_loader, data_loader_map = get_dataloader(
        train_config,
        database_config,
        datasetlist,
        cropped_input_size,
        data_size,
        channels_copy,
        k_fold,
    )


    train_data = collect_train_data(train_loaders, datasetlist, data_loader_map)

    separated_data = separate_into_single_modality(train_data)

    data = batch_data(separated_data, batch_size)

    print("total_separated_data_len:", len(data))

    train_vae(data, num_epochs=300)


    ## inference #########################################




    #put some data thorugh the trained model and try and recreate it 

    # load the model 
    config = VAE_config()
    vae = ConvVAE(config).to(device)
    vae.load_state_dict(torch.load('VAE/checkpoints/vaeee.pth', map_location=device))




    def reconstruct_image(model, test_image):
        model.eval()
        with torch.no_grad():
            if model == "VAE":
                test_image = test_image.view(1, -1).to(device)
            else:
                test_image = test_image.unsqueeze(0).to(device)
            recon_image, _, _ = model(test_image)
            recon_image = recon_image.view(test_image.size()).cpu().numpy()
        return recon_image

    # Example usage
    test_image = separated_data[0][0]  # Assuming separated_data is a list of (image, label) tuples
    test_image  = test_image.unsqueeze(0)
    reconstructed_image = reconstruct_image(vae, test_image)

    # Plot original and reconstructed images
    slice_idx = 64  # Choose a slice index to visualize

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image Slice")
    plt.imshow(test_image.view(128, 128, 128)[slice_idx, :, :].cpu().numpy(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image Slice")
    plt.imshow(reconstructed_image[0].reshape(128, 128, 128)[slice_idx, :, :], cmap='gray')
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    

    


