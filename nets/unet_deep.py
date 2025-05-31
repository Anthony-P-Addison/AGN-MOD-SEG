import torch
import torch.nn as nn
from nets.residual_block_identity import ResidualUnit_changed as ResidualUnit
from monai.networks.blocks import Convolution


    

class res_unet(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels:int = 1,
        last_layer_conv_only:bool = True,
        invariant_channel: bool = False,
        aux_loss: bool = False
        
    ) -> None:
        super().__init__()

        print("RES_UNET INIT with Deeper Invariant Channel")
        dropout = 0.2
        self.invariant_channel_enabled = invariant_channel
        invariant_out_channels = 8
        self.invariant_out_channels = invariant_out_channels
        self.aux_loss = aux_loss

    
        if self.invariant_channel_enabled:
            
            self.invariant_stream = nn.Sequential(
                ResidualUnit(spatial_dims=3, in_channels=1, out_channels=8, strides=1, kernel_size=3, subunits=1, dropout=dropout),
                Convolution(spatial_dims=3,in_channels=8,out_channels=16,strides=1,kernel_size=3,dropout=0.2),
                Convolution(spatial_dims=3,in_channels=16,out_channels=invariant_out_channels,strides=1,kernel_size=3,dropout=0.2))
            modality_channels = in_channels-1
            if self.aux_loss:
                self.aux_head = AuxHeadWithPreproc( self.invariant_out_channels)
        else:
            self.invariant_stream = None
            modality_channels = in_channels
            invariant_out_channels = 0

        self.conv_1 = ResidualUnit(spatial_dims=3, in_channels=modality_channels, out_channels=modality_channels, strides=1, kernel_size=3, subunits=1, dropout=dropout)
        
        downstream_in_channels = modality_channels + invariant_out_channels
        self.down_conv_1 = Convolution(spatial_dims=3, in_channels=downstream_in_channels, out_channels=32, strides=2, kernel_size=3, dropout=dropout)
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
        if self.invariant_channel_enabled:
            invariant_x = x[:, -1:, ...]
            modality_x = x[:, :-1, ...]

            invariant_features = self.invariant_stream(invariant_x)
            modality_features = self.conv_1(modality_x)
            fused_features = torch.cat((modality_features, invariant_features), dim=1)

            if self.aux_loss:
                aux_output = self.aux_head(invariant_features)
            else:
                aux_output = None

        else:
            modality_features = self.conv_1(x)
            fused_features = modality_features
            aux_output = None
            

        down1 = self.down_conv_1(fused_features)
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

        
        
        return up_out_4, aux_output


class AuxHeadWithPreproc(nn.Module):
    def __init__(self, invariant_stream):
        super().__init__()
        self.invariant_stream = invariant_stream  # existing nn.Sequential
        self.final_conv = Convolution(spatial_dims=3,in_channels=8,out_channels=1,strides=1,kernel_size=3,dropout=0.2,conv_only=True)
    def forward(self, x):
        return self.final_conv(x)


