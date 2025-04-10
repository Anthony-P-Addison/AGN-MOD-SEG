import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = ConvNormReLU(in_channels, out_channels, kernel_size, stride, dropout=dropout)
        self.conv2 = ConvNormReLU(out_channels, out_channels, kernel_size, dropout=dropout)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x

class res_unet(nn.Module):
    def __init__(self, in_channels, out_channels=1, last_layer_conv_only=True, invariant_channel=False):
        super().__init__()
        
        print("RES_UNET INIT with Deeper Invariant Channel")
        dropout = 0.2
        self.invariant_channel_enabled = invariant_channel
        invariant_out_channels = 8

        if self.invariant_channel_enabled:
            self.invariant_stream = nn.Sequential(
                BasicResBlock(1, 8, dropout=dropout),
                nn.ReLU(inplace=True),
                ConvNormReLU(8, 16, dropout=dropout),
                nn.ReLU(inplace=True),
                ConvNormReLU(16, invariant_out_channels, dropout=dropout)
            )
            modality_channels = in_channels - 1
        else:
            self.invariant_stream = None
            modality_channels = in_channels
            invariant_out_channels = 0

        self.conv_1 = BasicResBlock(modality_channels, modality_channels, dropout=dropout)
        
        downstream_in_channels = modality_channels + invariant_out_channels
        self.down_conv_1 = ConvNormReLU(downstream_in_channels, 32, stride=2, dropout=dropout)
        
        conv_2 = BasicResBlock(32, 32, dropout=dropout)
        down_conv_2 = ConvNormReLU(32, 64, stride=2, dropout=dropout)
        conv_3 = BasicResBlock(64, 64, dropout=dropout)
        down_conv_3 = ConvNormReLU(64, 128, stride=2, dropout=dropout)
        conv_4 = BasicResBlock(128, 128, dropout=dropout)
        down_conv_4 = ConvNormReLU(128, 256, stride=2, dropout=dropout)
        conv_5 = BasicResBlock(256, 256, dropout=dropout)
        down_conv_5 = ConvNormReLU(256, 512, stride=1, dropout=dropout)

        self.conv_2 = nn.Sequential(conv_2, down_conv_2)
        self.conv_3 = nn.Sequential(conv_3, down_conv_3)
        self.conv_4 = nn.Sequential(conv_4, down_conv_4)
        self.conv_5 = nn.Sequential(conv_5, down_conv_5)

        upsample = nn.Upsample(scale_factor=2)

        up_conv_1_a = ConvNormReLU(768, 768, dropout=dropout)
        up_conv_2_a = ConvNormReLU(256, 256, dropout=dropout)
        up_conv_3_a = ConvNormReLU(128, 128, dropout=dropout)
        up_conv_4_a = ConvNormReLU(64, 64, dropout=dropout)

        up_conv_1_b = ConvNormReLU(768, 128, dropout=dropout)
        up_conv_2_b = ConvNormReLU(256, 64, dropout=dropout)
        up_conv_3_b = ConvNormReLU(128, 32, dropout=dropout)
        up_conv_4_b = nn.Conv3d(64, out_channels, kernel_size=3, padding=1) if last_layer_conv_only else ConvNormReLU(64, out_channels, dropout=dropout)

        self.up_stage_1 = nn.Sequential(upsample, up_conv_1_a, up_conv_1_b)
        self.up_stage_2 = nn.Sequential(upsample, up_conv_2_a, up_conv_2_b)
        self.up_stage_3 = nn.Sequential(upsample, up_conv_3_a, up_conv_3_b)
        self.up_stage_4 = nn.Sequential(upsample, up_conv_4_a, up_conv_4_b)

    def forward(self, x):
        if self.invariant_channel_enabled:
            invariant_x = x[:, -1:, ...]
            modality_x = x[:, :-1, ...]

            invariant_features = self.invariant_stream(invariant_x)
            modality_features = self.conv_1(modality_x)
            fused_features = torch.cat((modality_features, invariant_features), dim=1)
        else:
            modality_features = self.conv_1(x)
            fused_features = modality_features

        down1 = self.down_conv_1(fused_features)
        conv_out_2 = self.conv_2(down1)
        conv_out_3 = self.conv_3(conv_out_2)
        conv_out_4 = self.conv_4(conv_out_3)
        conv_out_5 = self.conv_5(conv_out_4)

        up_in_1 = torch.cat((conv_out_5, conv_out_4), dim=1)
        up_out_1 = self.up_stage_1(up_in_1)

        up_in_2 = torch.cat((up_out_1, conv_out_3), dim=1)
        up_out_2 = self.up_stage_2(up_in_2)

        up_in_3 = torch.cat((up_out_2, conv_out_2), dim=1)
        up_out_3 = self.up_stage_3(up_in_3)

        up_in_4 = torch.cat((up_out_3, down1), dim=1)
        up_out_4 = self.up_stage_4(up_in_4)

        return up_out_4 