import torch
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from config import Training_config, Database_config
import copy
from VAE.VAE import VAE, VAE_config, ConvVAE
from VAE.dataloader import collect_train_data, separate_into_single_modality, batch_data


# load up el data
train_config = Training_config()
database_config = Database_config()
datasetlist = ["BRATS", "WMH"]
cropped_input_size = [128, 128, 128]
k_fold = None
channels_copy = copy.deepcopy(database_config.channels)
batch_size = 1

# data_size
train_size = database_config.train_size
data_size = 1
#data_size = min(train_size[dataset] for dataset in datasetlist)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# put some data thorugh the trained model and try and recreate it

# load the model
config = VAE_config()
vae = ConvVAE(config).to(device)
vae.load_state_dict(torch.load("VAE/checkpoints/vaeee.pth", map_location=device))


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
test_image = separated_data[0][
    0
]  # Assuming separated_data is a list of (image, label) tuples
reconstructed_image = reconstruct_image(vae, test_image)

# Plot original and reconstructed images
slice_idx = 64  # Choose a slice index to visualize

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image Slice")
plt.imshow(test_image.view(128, 128, 128)[slice_idx, :, :].cpu().numpy(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image Slice")
plt.imshow(reconstructed_image[0].reshape(128, 128, 128)[slice_idx, :, :], cmap="gray")
plt.show()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)


reconstructed_image = reconstruct_image(vae, test_image)
