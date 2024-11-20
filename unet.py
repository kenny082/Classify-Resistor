import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Helper function for creating a 3x3 convolutional layer
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

# Helper function for creating a 1x1 convolutional layer
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

# Helper function for upsampling (transpose convolution or bilinear upsampling)
def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))

# Block for the down-convolution part (encoder)
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

# Block for the up-convolution part (decoder)
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.merge_mode = merge_mode
        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)

        # Create a convolution block that takes the merged features
        if self.merge_mode == 'concat':
            self.conv = nn.Sequential(
                conv3x3(2 * out_channels, out_channels),
                nn.ReLU(inplace=True),
                conv3x3(out_channels, out_channels),
                nn.ReLU(inplace=True)
            )
        else:  # 'add' mode
            self.conv = nn.Sequential(
                conv3x3(out_channels, out_channels),
                nn.ReLU(inplace=True),
                conv3x3(out_channels, out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), dim=1)
        else:  # 'add' mode
            x = from_up + from_down
        x = self.conv(x)
        return x

# The full U-Net model
class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat'):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.up_mode = up_mode
        self.merge_mode = merge_mode

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Encoder: Create the downsampling layers
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            self.down_convs.append(DownConv(ins, outs, pooling))

        # Decoder: Create the upsampling layers
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConv(ins, outs, merge_mode=merge_mode, up_mode=up_mode))

        # Final layer: Convolution to output the desired number of classes
        self.conv_final = conv1x1(outs, num_classes)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []

        # Forward pass through the down-convolutions (encoder)
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # Forward pass through the up-convolutions (decoder)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Final convolution to map to the output classes
        x = self.conv_final(x)
        return x
