import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, Convolution

class CustomUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(CustomUNet, self).__init__()
        self.invariant_channel_index = in_channels - 1
  
        # Define the main UNet layers
        self.conv_1 = ResidualUnit(spatial_dims=3, in_channels=in_channels - 1, out_channels=16, strides=2, kernel_size=3, dropout=0.2)
        self.conv_2 = ResidualUnit(spatial_dims=3, in_channels=16, out_channels=32, strides=2, kernel_size=3, dropout=0.2)
        self.conv_3 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, dropout=0.2)
        self.conv_4 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, dropout=0.2)
        self.conv_5 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=1, kernel_size=3, dropout=0.2)

        upsample = torch.nn.Upsample(scale_factor=2)

        up_conv_1_a = Convolution(spatial_dims=3, in_channels=384, out_channels=384, strides=1, kernel_size=3, dropout=0.2)
        up_conv_2_a = Convolution(spatial_dims=3, in_channels=128, out_channels=128, strides=1, kernel_size=3, dropout=0.2)
        up_conv_3_a = Convolution(spatial_dims=3, in_channels=64, out_channels=64, strides=1, kernel_size=3, dropout=0.2)
        up_conv_4_a = Convolution(spatial_dims=3, in_channels=32, out_channels=32, strides=1, kernel_size=3, dropout=0.2)

        up_conv_1_b = Convolution(spatial_dims=3, in_channels=384, out_channels=64, strides=1, kernel_size=3, dropout=0.2)
        up_conv_2_b = Convolution(spatial_dims=3, in_channels=128, out_channels=32, strides=1, kernel_size=3, dropout=0.2)
        up_conv_3_b = Convolution(spatial_dims=3, in_channels=64, out_channels=16, strides=1, kernel_size=3, dropout=0.2)
        up_conv_4_b = Convolution(spatial_dims=3, in_channels=32, out_channels=out_channels, strides=1, kernel_size=3, dropout=0.2, conv_only=True)

        self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
        self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
        self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
        self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)

        # Define the invariant channel branch
        self.invar1 = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=16, strides=2, kernel_size=3, dropout=0.2)
        self.invar2 = ResidualUnit(spatial_dims=3, in_channels=16, out_channels=32, strides=2, kernel_size=3, dropout=0.2)
        self.invar3 = ResidualUnit(spatial_dims=3, in_channels=32, out_channels=64, strides=2, kernel_size=3, dropout=0.2)
        self.invar4 = ResidualUnit(spatial_dims=3, in_channels=64, out_channels=128, strides=2, kernel_size=3, dropout=0.2)
        self.invar5 = ResidualUnit(spatial_dims=3, in_channels=128, out_channels=256, strides=1, kernel_size=3, dropout=0.2)

        # Define learnable scaling parameters with an initial value
        # self.scale_1 = nn.Parameter(torch.ones(1) * 0.5)
        # self.scale_2 = nn.Parameter(torch.ones(1) * 0.5)
        # self.scale_3 = nn.Parameter(torch.ones(1) * 0.5)
        # self.scale_4 = nn.Parameter(torch.ones(1) * 0.5)
        # self.scale_5 = nn.Parameter(torch.ones(1) * 0.5)
        # attention mmodules: 
        self.attention_1 = AttentionModule(16) 
        self.attention_2 = AttentionModule(32)  
        self.attention_3 = AttentionModule(64) 
        self.attention_4 = AttentionModule(128) 
        self.attention_5 = AttentionModule(256)
        
        self.attention_1 = AttentionModule(16) 
        self.attention_2 = AttentionModule(32)  
        self.attention_3 = AttentionModule(64) 
        self.attention_4 = AttentionModule(128) 
        self.attention_5 = AttentionModule(256)

  
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the invariant channel separately
        invariant_channel = x[:, [self.invariant_channel_index], :, :, :]
        invariant_features_1 = self.invar1(invariant_channel)
        invariant_features_2 = self.invar2(invariant_features_1)
        invariant_features_3 = self.invar3(invariant_features_2)
        invariant_features_4 = self.invar4(invariant_features_3)
        invariant_features_5 = self.invar5(invariant_features_4)

        # Process the other channels
        
        #x = torch.cat([x[:, :self.invariant_channel_index, :, :, :], x[:, self.invariant_channel_index + 1:, :, :, :]], dim=1)
        conv_out_1 = self.conv_1(x)
        conv_out_2 = self.conv_2(conv_out_1)
        conv_out_3 = self.conv_3(conv_out_2)
        conv_out_4 = self.conv_4(conv_out_3)
        conv_out_5 = self.conv_5(conv_out_4)

        # # Scale the invariant features
        # invariant_features_1 = invariant_features_1 *(self.scale_1+1e-2)
        # invariant_features_2 = invariant_features_2 *(self.scale_2+1e-2)
        # invariant_features_3 = invariant_features_3 *(self.scale_3+1e-2)
        # invariant_features_4 = invariant_features_4 *(self.scale_4+1e-2)
        # invariant_features_5 = invariant_features_5 *(self.scale_5+1e-2)

        # Apply attention to the invariant features and main path features separately
        invariant_features_1_att = self.attention_1(invariant_features_1)
        invariant_features_2_att = self.attention_2(invariant_features_2)
        invariant_features_3_att = self.attention_3(invariant_features_3)
        invariant_features_4_att = self.attention_4(invariant_features_4)
        invariant_features_5_att = self.attention_5(invariant_features_5)

        conv_out_1_att = self.attention_1(conv_out_1)
        conv_out_2_att = self.attention_2(conv_out_2)
        conv_out_3_att = self.attention_3(conv_out_3)
        conv_out_4_att = self.attention_4(conv_out_4)
        conv_out_5_att = self.attention_5(conv_out_5)

        # Combine the invariant features with the main path features at each layer
        # combined_features_1 = torch.cat((conv_out_1, invariant_features_1), dim=1)
        # combined_features_2 = torch.cat((conv_out_2, invariant_features_2), dim=1)
        # combined_features_3 = torch.cat((conv_out_3, invariant_features_3), dim=1)
        # combined_features_4 = torch.cat((conv_out_4, invariant_features_4), dim=1)
        # combined_features_5 = torch.cat((conv_out_5, invariant_features_5), dim=1)

        # Combine the invariant features with the main path features at each layer
        combined_features_1 = (conv_out_1*conv_out_1_att) + (invariant_features_1*invariant_features_1_att)
        combined_features_2 = (conv_out_2*conv_out_2_att) + (invariant_features_2*invariant_features_2_att)
        combined_features_3 = (conv_out_3* conv_out_3_att) + (invariant_features_3*invariant_features_3_att)
        combined_features_4 = (conv_out_4* conv_out_4_att) + (invariant_features_4*invariant_features_4_att)
        combined_features_5 = (conv_out_5*conv_out_5_att) + (invariant_features_5*invariant_features_5_att)

        # Upsample and concatenate with downsampled invariant features for skip connections
        up_in_1 = torch.cat((combined_features_5, combined_features_4), dim=1)
        up_out_1 = self.up_stage_1(up_in_1)

        up_in_2 = torch.cat((up_out_1, combined_features_3), dim=1)
        up_out_2 = self.up_stage_2(up_in_2)

        up_in_3 = torch.cat((up_out_2, combined_features_2), dim=1)
        up_out_3 = self.up_stage_3(up_in_3)

        up_in_4 = torch.cat((up_out_3, combined_features_1), dim=1)
        up_out_4 = self.up_stage_4(up_in_4)

        return up_out_4
    




class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.softmax(attention)
        return x * attention