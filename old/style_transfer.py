#load pre trained unet model
from nets.unet import Unet
import torch
from config import Training_config, Database_config, Test_config
import copy

train_config = Training_config
database_config = Database_config

device_id = 0

# initialize GPU
print("Running on GPU:" + str(device_id))
cuda_id = "cuda:" + str(device_id)
device = torch.device(cuda_id)
# torch.cuda.set_device(cuda_id)

checkpoint = "models/all_in_one/WMH_MSSEG_BRATS_ATLAS_TBI/all_in_one_random_drop_1_2024-12-19_17-40_Epoch_549.pth"

# Initialize the model
model = Unet(in_channels=6).to(device)

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print(model)


# Define a hook to capture the output of a specific layer
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# Choose the layer you want to hook into
layer_name = "conv_1.conv.unit0.conv"
activation = {}
layer = dict([*model.named_modules()])[layer_name]
layer.register_forward_hook(get_activation(layer_name))

# Prepare the style_image tensor and pass it through the model

# load up nifty and pre proess

# get the data.
from dataloader import get_dataloader

datasets = ["ATLAS", "WMH"]
data_size = 0
train_size = database_config.train_size
channels_copy = copy.deepcopy(database_config.channels)

for data in datasets:

    data_size = max(data_size, train_size[data])

train_loaders, val_loader, data_loader_map = get_dataloader(
    train_config=train_config,
    database_config=database_config,
    datasetlist=datasets,
    cropped_input_size=[128, 128, 128],
    data_size=1,
    channels_copy=channels_copy,
    k_fold=None,
)


datasets_trained_on = "WMGH_MSSEG_BRATS_ATLAS_TBI"


for batch_data in zip(*train_loaders):

    for dataset in datasets:
        if dataset =="WMH":
            style_image = batch_data[0][0].to(device)

            # Create a new tensor with 6 channels, where the first channel is the original style_image and the rest are zeros
            new_style_image = torch.zeros((style_image.shape[0], 6, *style_image.shape[2:])).to(device)
            new_style_image[:, 0, :, :, :] = style_image[:, 0, :, :, :]
            model(new_style_image)

        elif dataset == "ATLAS":
            content_image = batch_data[1][0].to(device)
            new_content_image = torch.zeros((content_image.shape[0], 6, *content_image.shape[2:])).to(device)
            new_content_image[:, [0, 1], :, :, :] = content_image[:, [0,1], :, :, :]
            model(new_content_image)

        # Pass the new style and content images through the model
      

# Access the output of the specific layer
# want to do this for different layers and fuse them for different content.

layer_output = activation[layer_name]
print(layer_output.shape)


import matplotlib.pyplot as plt

# Plot one of the feature maps from the layer output
feature_map = layer_output[0, 0, :, :, :].cpu().numpy()

plt.figure(figsize=(10, 10))
plt.imshow(feature_map[feature_map.shape[0] // 2, :, :], cmap="gray")
plt.title("Feature Map from Layer: " + layer_name)
plt.colorbar()
plt.show()




import torch
from torchvision import transforms

import torch.optim as optim

# Define the layers to extract features from
content_layers = ['conv_4.conv.unit0.conv']
style_layers = ['conv_1.conv.unit0.conv', 'conv_2.conv.unit0.conv', 'conv_3.conv.unit0.conv']


# Define the input image (initialized as the content image)
input_image = new_content_image.clone().requires_grad_(True)

# Define the optimizer
optimizer = optim.Adam([input_image], lr=0.01)

# Define the loss functions
mse_loss = torch.nn.MSELoss()

# Define the number of iterations
num_iterations = 300


# Extract features from the content and style images
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model.named_modules():
        x = layer.forward(x)
        if name in layers:
            features[name] = x
    return features

content_features = get_features(new_content_image, model, content_layers)
style_features = get_features(style_image, model, style_layers)

# Compute the Gram matrix for the style features
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Perform style transfer
for i in range(num_iterations):
    optimizer.zero_grad()
    
    input_features = get_features(input_image, model, content_layers + style_layers)
    
    content_loss = mse_loss(input_features[content_layers[0]], content_features[content_layers[0]])
    
    style_loss = 0
    for layer in style_layers:
        input_gram = gram_matrix(input_features[layer])
        style_gram = style_grams[layer]
        style_loss += mse_loss(input_gram, style_gram)
    
    total_loss = content_loss + style_loss
    total_loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Iteration {i}/{num_iterations}, Total Loss: {total_loss.item()}")

# Display the final stylized image
stylized_image = input_image.detach().cpu().squeeze().numpy()
plt.figure(figsize=(10, 10))
plt.imshow(stylized_image[stylized_image.shape[0] // 2, :, :], cmap='gray')
plt.title('Stylized Image')
plt.colorbar()
plt.show()
