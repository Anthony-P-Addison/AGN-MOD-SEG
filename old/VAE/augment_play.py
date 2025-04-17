from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from dataloader import get_dataloader
from tqdm import tqdm
import random







def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#### load up the data in same format as when training for validation and training ####

from config import Database_config, Training_config
from dataloader import get_dataloader
import copy
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns




train_config = Training_config
database_config = Database_config
datasetlist = ["BRATS","WMH"]
cropped_input_size = [128, 128, 128]
k_fold = None
channels_copy = copy.deepcopy(database_config.channels)


# data_size
train_size = database_config.train_size
data_size = 5
# data_size = min(train_size[dataset] for dataset in datasetlist)


print("Data size:", data_size)

# load up the training data and the data loader map

train_loaders, val_loader, data_loader_map = get_dataloader(
    train_config,
    database_config,
    datasetlist,
    cropped_input_size,
    data_size,
    channels_copy,
    k_fold,
)


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3_mean = nn.Linear(hidden_dim2, latent_dim)
        self.fc3_log_var = nn.Linear(hidden_dim2, latent_dim)

    def forward(self, x) -> torch.tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        z_mean = self.fc3_mean(h)           # calculate mean 
        z_log_var = self.fc3_log_var(h)     # calculate log variance
        return z_mean, z_log_var


# Sampling function
def sampling(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return z_mean + eps * std


# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim2, hidden_dim1, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        x_recon = torch.sigmoid(self.fc3(h))
        return x_recon


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim2, hidden_dim1, input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var


# Instantiate the VAE model

hidden_dim1 = 64
hidden_dim2 = 32
latent_dim = 2
input_dim = 1
vae = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)


# ##############################################################################

# img_index = 0


# Function to get latent vectors for a dataset
# def get_latent_vectors(batch_size):
#     latent_vectors_dict = {}
#     for data in datasetlist:
#         latent_vectors = []
#         for i, batch_data in enumerate(
#             tqdm(
#                 train_loaders[data_loader_map[data]],
#                 desc=f"Extracting latent vectors for {data}",
#             )
#         ):
#             if i > 20:
#                 break
#             img = batch_data[0]
#             img = img.view(-1, img.size(0)) 
#             # Flatten the image tensor
#             input_dim = img.shape[1]
#             vae = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
#             z_mean, z_log_var = vae.encoder(img)
#             z = sampling(z_mean, z_log_var)
#             latent_vectors.append(z.detach().view(-1).cpu().numpy())
#         latent_vectors_dict[data] = latent_vectors
#     return latent_vectors_dict


# #############################################################################


# # Get latent vectors for each dataset

# latent_vector = get_latent_vectors(1)

# # Calculate pairwise distances and plot for each dataset
# #plt.figure(figsize=(10, 6))
# colors = ["blue", "green", "red", "purple", "orange", "brown"]

# for idx, dataset in enumerate(datasetlist):

#     # Calculate pairwise distances
#     distances = pdist(latent_vector[dataset], metric="euclidean")
#     distance_matrix = squareform(distances)

#     # Reduce dimensions using PCA for visualization
#     pca = PCA(n_components=2)
#     latent_vectors_reduced = pca.fit_transform(latent_vector[dataset])

#     # Normalize the values to be between 0 and 100
#     latent_vectors_reduced = (
#         100
#         * (latent_vectors_reduced - np.min(latent_vectors_reduced))
#         / (np.max(latent_vectors_reduced) - np.min(latent_vectors_reduced))
#     )

#     # Plot the latent vectors using seaborn
#     sns.scatterplot(
#         x=latent_vectors_reduced[:, 0],
#         y=latent_vectors_reduced[:, 1],
#         label=dataset,
#         alpha=0.6,
#         color=colors[idx % len(colors)],
#     )

# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title(f"Latent Space Distribution (PCA Reduced) for {datasetlist}")
# plt.legend()
# plt.show()

  
# ##### Calculate the average location for each point #####
# average_locations = {}
# for dataset in datasetlist:
#     latent_vectors = np.array(latent_vector[dataset])
#     average_location = np.mean(latent_vectors, axis=0)
#     average_locations[dataset] = average_location

# # Plot the heat map of the average locations
# plt.figure(figsize=(10, 6))
# for idx, dataset in enumerate(datasetlist):
#     avg_loc = average_locations[dataset]
#     plt.scatter(avg_loc[0], avg_loc[1], label=dataset, color=colors[idx % len(colors)], s=100)

# plt.xlabel("Latent Dimension 1")
# plt.ylabel("Latent Dimension 2")
# plt.title("Average Location in Latent Space")
# plt.legend()
# plt.show()



### randomly sample and decode the information to see what the images look like

# Sample from the latent space and decode the images




def get_latent_vectors(batch_size):
    latent_vectors_dict = {}
    for data in datasetlist:
        latent_vectors = []
        for i, batch_data in enumerate(
            tqdm(
                train_loaders[data_loader_map[data]],
                desc=f"Extracting latent vectors for {data}",
            )
        ):
            if i > 5:               # start with 5 samples from traininig set
                break
            img = batch_data[0]
            img = img.view(-1, img.size(0)) 
            # Flatten the image tensor
            input_dim = img.shape[1]
            vae = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
            z_mean, z_log_var = vae.encoder(img)
            z = sampling(z_mean, z_log_var)
            latent_vectors.append(z.detach().view(-1).cpu().numpy())
        latent_vectors_dict[data] = latent_vectors
    return latent_vectors_dict






# Define the loss function
def vae_loss(recon_x, x, z_mean, z_log_var):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_divergence

# Optimizer

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Training function
def train_vae(train_loaders, num_epochs=1):

   
    for epoch in range(num_epochs):
       
        train_loss = 0
        optimizer.zero_grad()
        for data in datasetlist:
            for batch_data in tqdm(train_loaders[data_loader_map[data]], desc=f"Training VAE for {data} epoch {epoch}"):
                img = batch_data[0]
                img = img.view(-1, img.size(0))  # Flatten the image tensor
                input_dim = img.shape[1]
                vae = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
                recon_x, z_mean, z_log_var = vae(img)
                loss = vae_loss(recon_x, img, z_mean, z_log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # 
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loaders)}")

# Train the VAE
train_vae(train_loaders)



# visualising generated samples

#load model. 

with torch.no_grad():
    z = torch.randn(1, latent_dim)
    sample = vae.decoder(z) #.view(cropped_input_size).cpu().numpy()
    plt.imshow(sample[64, :, :], cmap="gray")
    plt.show()

