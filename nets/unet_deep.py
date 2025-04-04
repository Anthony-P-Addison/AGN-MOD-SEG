import torch
import torch.nn as nn
from nets.residual_block_identity import ResidualUnit_changed as ResidualUnit
from monai.networks.blocks import Convolution


    


class res_unet(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels:int = 1,
        last_layer_conv_only:bool = True,
        invariant_channel: bool = True
    ) -> None:
        super().__init__()

        print("RES_UNET INIT with Invariant Channel Processing")
        dropout = 0.2
        print("Dropout: ",dropout)

        # # Separate processing for invariant channel (no downsampling)
        # self.invariant_stream = nn.Sequential(
        #     ResidualUnit(spatial_dims=3, in_channels=1, out_channels=4, strides=1, kernel_size=3, subunits=1, dropout=0.2),
        #     ResidualUnit(spatial_dims=3, in_channels=4, out_channels=8, strides=1, kernel_size=3, subunits=1, dropout=0.2)
        # )

        # Replace the current invariant_stream with a more robust feature extractor
        self.invariant_stream = nn.Sequential(
        # Initial feature extraction
        ResidualUnit(spatial_dims=3, in_channels=1, out_channels=8, strides=1, kernel_size=3, subunits=2, dropout=0.2),
        # Multi-scale processing
        nn.ModuleList([
            ResidualUnit(spatial_dims=3, in_channels=8, out_channels=16, strides=1, kernel_size=k, subunits=1, dropout=0.2)
            for k in [1, 3, 5]  # Multiple kernel sizes to capture features at different scales
        ]),

        # Feature fusion
        Convolution(spatial_dims=3, in_channels=16*3, out_channels=16, strides=1, kernel_size=1))


        # Modality-specific processing (reduced input channels by 1 for invariant channel)
        modality_channels = in_channels - 1 if invariant_channel else in_channels
        
        # Main encoder path (for modality-specific channels)
        self.conv_1 = ResidualUnit(spatial_dims=3, in_channels=modality_channels, out_channels=modality_channels, strides=1, kernel_size=3, subunits=1, dropout=0.2)
        self.down_conv_1 = Convolution(spatial_dims=3, in_channels=14, out_channels=32, strides=2, kernel_size=3, dropout=0.2)
        conv_2 = ResidualUnit(spatial_dims=3,in_channels=32,out_channels=32,strides=1,kernel_size=3,subunits=1,dropout=0.2)
        down_conv_2 = Convolution(spatial_dims=3,in_channels=32,out_channels=64,strides=2,kernel_size=3,dropout=0.2)
        conv_3 = ResidualUnit(spatial_dims=3,in_channels=64,out_channels=64,strides=1,kernel_size=3,subunits=1,dropout=0.2)
        down_conv_3 = Convolution(spatial_dims=3,in_channels=64,out_channels=128,strides=2,kernel_size=3,dropout=0.2)
        conv_4 = ResidualUnit(spatial_dims=3,in_channels=128,out_channels=128,strides=1,kernel_size=3,subunits=1,dropout=0.2)
        down_conv_4 = Convolution(spatial_dims=3,in_channels=128,out_channels=256,strides=2,kernel_size=3,dropout=0.2)
        conv_5 = ResidualUnit(spatial_dims=3,in_channels=256,out_channels=256,strides=1,kernel_size=3,subunits=1,dropout=0.2)
        down_conv_5 = Convolution(spatial_dims=3,in_channels=256,out_channels=512,strides=1,kernel_size=3,dropout=0.2)  

        self.conv_2 = nn.Sequential(conv_2,down_conv_2)
        self.conv_3 = nn.Sequential(conv_3,down_conv_3)
        self.conv_4 = nn.Sequential(conv_4,down_conv_4)
        self.conv_5 = nn.Sequential(conv_5,down_conv_5)

        upsample = torch.nn.Upsample(scale_factor=2)

        up_conv_1_a = Convolution(spatial_dims=3,in_channels=768,out_channels=768,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_a = Convolution(spatial_dims=3,in_channels=256,out_channels=256,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_a = Convolution(spatial_dims=3,in_channels=128,out_channels=128,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_a = Convolution(spatial_dims=3,in_channels=64,out_channels=64,strides=1,kernel_size=3,dropout=0.2)

        up_conv_1_b = Convolution(spatial_dims=3,in_channels=768,out_channels=128,strides=1,kernel_size=3,dropout=0.2)
        up_conv_2_b = Convolution(spatial_dims=3,in_channels=256,out_channels=64,strides=1,kernel_size=3,dropout=0.2)
        up_conv_3_b = Convolution(spatial_dims=3,in_channels=128,out_channels=32,strides=1,kernel_size=3,dropout=0.2)
        up_conv_4_b = Convolution(spatial_dims=3,in_channels=64,out_channels=out_channels,strides=1,kernel_size=3,dropout=0.2,conv_only=last_layer_conv_only)

        self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
        self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
        self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
        self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into invariant and modality-specific channels
        invariant_x = x[:, -1:, ...]  # Take last channel as invariant
        modality_x = x[:, :-1, ...]    # Rest are modality-specific

        # Process invariant channel at full resolution
        invariant_features = self.invariant_stream(invariant_x)

        # Process modality-specific channels
        conv_out_1 = self.conv_1(modality_x)

        # concat invariant and modality features]
        fused_features = torch.cat((invariant_features, conv_out_1), dim=1)

        down1= self.down_conv_1(fused_features)

        # Continue with regular processing
        conv_out_2 = self.conv_2(down1)
        conv_out_3 = self.conv_3(conv_out_2)
        conv_out_4 = self.conv_4(conv_out_3)
        conv_out_5 = self.conv_5(conv_out_4) 

        up_in_1 = torch.cat((conv_out_5,conv_out_4),dim=1)
        up_out_1 = self.up_stage_1(up_in_1)

        up_in_2 = torch.cat((up_out_1,conv_out_3),dim=1)
        up_out_2 = self.up_stage_2(up_in_2)

        up_in_3 = torch.cat((up_out_2,conv_out_2),dim=1)
        up_out_3 = self.up_stage_3(up_in_3)

        up_in_4 = torch.cat((up_out_3,down1),dim=1)
        up_out_4 = self.up_stage_4(up_in_4)

        return up_out_4