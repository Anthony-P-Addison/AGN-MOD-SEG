import torch 
from torch import nn
import pydantic



######  Configuration  ######

class VAE_config(pydantic.BaseModel):
    hidden_dim1: int = 32
    hidden_dim2: int = 64
    hidden_dim_3: int = 128
    latent_dim: int = 2
    input_dim: int = 128 * 128 * 128
   


###### model components  ######

###  for linear model ###

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3_mean = nn.Linear(hidden_dim2, latent_dim)
        self.fc3_log_var = nn.Linear(hidden_dim2, latent_dim)

    def forward(self, x) -> torch.tensor:
        #x = torch.flatten(x, start_dim=1)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        z_mean = self.fc3_mean(h)           
        z_log_var = self.fc3_log_var(h)     
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



#####  Define the VAE model #####
class VAE(nn.Module):
    def __init__(self, config:pydantic.BaseModel):
        super(VAE, self).__init__()
     
        self.encoder = Encoder(config.input_dim, config.hidden_dim1, config.hidden_dim2, config.latent_dim)
        self.decoder = Decoder(config.latent_dim, config.hidden_dim2, config.hidden_dim1, config.input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var
    



############### for 3D convolution model #################

# Define the convolutional encoder
class ConvEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_dim1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_dim1)
        self.dropout1 = nn.Dropout3d(0.2)
        self.conv2 = nn.Conv3d(hidden_dim1, hidden_dim2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_dim2)
        self.dropout2 = nn.Dropout3d(0.2)
        self.conv3 = nn.Conv3d(hidden_dim2, hidden_dim3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(hidden_dim3)
        self.dropout3 = nn.Dropout3d(0.2)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(hidden_dim3 * 64 * 64 * 64, latent_dim)
        self.fc2 = nn.Linear(hidden_dim3 * 64 * 64 * 64, latent_dim)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.dropout1(h)
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.dropout2(h)
        h = torch.relu(self.bn3(self.conv3(h)))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        z_mean = self.fc1(h)
        z_log_var = self.fc2(h)
        return z_mean, z_log_var


# Define the convolutional decoder
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim3, hidden_dim2, hidden_dim1, output_channels):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim3 * 64 * 64 * 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.deconv1 = nn.ConvTranspose3d(hidden_dim3, hidden_dim2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_dim2)
        self.dropout1 = nn.Dropout3d(0.2)
        self.deconv2 = nn.ConvTranspose3d(hidden_dim2, hidden_dim1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_dim1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.deconv3 = nn.ConvTranspose3d(hidden_dim1, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), -1, 8, 8, 8)
        h = self.upsample(h)
        h = torch.relu(self.deconv1(h))
        h = self.dropout1(h)
        h = torch.relu(self.deconv2(h))
        h = self.dropout2(h)
        x_recon = torch.sigmoid(self.deconv3(h))
        x_recon = x_recon.view(x_recon.size(0), 1, 128, 128, 128)  # Reshape to original shape
        return x_recon

# Define the convolutional VAE model
class ConvVAE(nn.Module):
    def __init__(self, config: pydantic.BaseModel):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(1, config.hidden_dim1, config.hidden_dim2, config.hidden_dim_3,config.latent_dim)
        self.decoder = ConvDecoder(config.latent_dim, config.hidden_dim2,config.hidden_dim_3,config.hidden_dim1, 1)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_log_var
    

    