import torch
import torch.nn as nn
from nets.residual_block import ResidualUnit_changed as ResidualUnit
from monai.networks.blocks import Convolution



class  Unet(nn.Module):

    def __init__(self,
        in_channels: int,
        invariant_channel_index:int, # ADD
        out_channels:int = 1,
        last_layer_conv_only:bool = True
    ) -> None:
        super().__init__()

        print("RES_UNET INIT")
        dropout = 0.2
        print("Dropout: ",dropout)

        self.invariant_channel_index = invariant_channel_index   #ADD

        in_channels= 6
        
        self.conv_1 = ResidualUnit(spatial_dims=3,in_channels=in_channels,out_channels=16,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_2 = ResidualUnit(spatial_dims=3,in_channels=16,out_channels=32,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_3 = ResidualUnit(spatial_dims=3,in_channels=32,out_channels=64,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_4 = ResidualUnit(spatial_dims=3,in_channels=64,out_channels=128,strides=2,kernel_size=3,subunits=2,dropout=0.2)
        self.conv_5 = ResidualUnit(spatial_dims=3,in_channels=128,out_channels=256,strides=1,kernel_size=3,subunits=2,dropout=0.2)

        upsample = torch.nn.Upsample(scale_factor=2)

        up_conv_1_a = Convolution(spatial_dims=3,in_channels=384,out_channels=384,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_a = Convolution(spatial_dims=3,in_channels=128,out_channels=128,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_a = Convolution(spatial_dims=3,in_channels=64,out_channels=64,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_a = Convolution(spatial_dims=3,in_channels=32,out_channels=32,strides=1,kernel_size=3,dropout=0.2)

        up_conv_1_b = Convolution(spatial_dims=3,in_channels=384,out_channels=64,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_b = Convolution(spatial_dims=3,in_channels=128,out_channels=32,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_b = Convolution(spatial_dims=3,in_channels=64,out_channels=16,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_b = Convolution(spatial_dims=3,in_channels=32,out_channels=out_channels,strides=1,kernel_size=3,dropout=0.2,conv_only=last_layer_conv_only)

        self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
        self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
        self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
        self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)

        # TODO: added here
        self.invariant_branch = ResidualUnit(spatial_dims=3, in_channels=1, out_channels=16, strides=2, kernel_size=3, subunits=2, dropout=0.2)
      

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #TODO Process the invariant channel separately
        invariant_channel = x[:,self.invariant_channel_index, :, :, :].unsqueeze(1)
        invariant_features = self.invariant_branch(invariant_channel)

        #TODO Process the other channels
        x = torch.cat([x[:, :self.invariant_channel_index, :, :, :], x[:, self.invariant_channel_index + 1:, :, :, :]], dim=1)

        conv_out_1 = self.conv_1(x)
        conv_out_2 = self.conv_2(conv_out_1)
        conv_out_3 = self.conv_3(conv_out_2)
        conv_out_4 = self.conv_4(conv_out_3)
        conv_out_5 = self.conv_5(conv_out_4)

        # Combine the invariant features with the bottleneck
        invariant_features_upsampled = torch.nn.functional.interpolate(invariant_features, size=conv_out_5.shape[2:])
        combined_features = torch.cat([conv_out_5, invariant_features_upsampled], dim=1)

        #up_in_1 = torch.cat((conv_out_5,conv_out_4),dim=1)

        # Ensure combined_features and conv_out_4 have the same spatial dimensions
        combined_features_resized = torch.nn.functional.interpolate(combined_features, size=conv_out_4.shape[2:])
        up_in_1 = torch.cat((combined_features_resized, conv_out_4), dim=1)
        up_out_1 = self.up_stage_1(up_in_1)

        up_in_2 = torch.cat((up_out_1,conv_out_3),dim=1)
        up_out_2 = self.up_stage_2(up_in_2)

        up_in_3 = torch.cat((up_out_2,conv_out_2),dim=1)
        up_out_3 = self.up_stage_3(up_in_3)

        up_in_4 = torch.cat((up_out_3,conv_out_1),dim=1)
        up_out_4 = self.up_stage_4(up_in_4)

        return up_out_4


    def add_input_channel(self):
        """
        Add an additional input channel with randomly initialized weights.
        """
        # Get the current weights of the first convolutional layer
        old_weights = self.conv_1.conv[0].conv.weight.data

        # Create new weights with an additional input channel
        new_weights = torch.randn((old_weights.shape[0], old_weights.shape[1] + 1, old_weights.shape[2], old_weights.shape[3], old_weights.shape[4]))

        # Copy the old weights to the new weights
        new_weights[:, :-1, :, :, :] = old_weights

        # Randomly initialize the new channel weight 
        # dont think I want the below line as all run in torch.no_grad(). 
        nn.init.kaiming_normal_(new_weights[:, -1, :, :, :], mode='fan_in', nonlinearity='relu')

        # Assign the new weights to the first convolutional layer
        self.conv_1.conv[0].weight = nn.Parameter(new_weights)

        # If the first convolutional layer has a bias term, adjust it accordingly
        if self.conv_1.conv[0].bias is not None:
            old_bias = self.conv_1.conv[0].bias.data
            new_bias = torch.cat((old_bias, torch.randn(1)))
            self.conv_1.conv[0].bias = nn.Parameter(new_bias)







            def get_layer(self, layer_name: str):
                """
                Get a specific layer by its name.
                """
                return dict(self.named_modules()).get(layer_name, None)