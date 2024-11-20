import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

# Please note that this code is not original and was taken from:
# https://github.com/jaxony/unet-pytorch/blob/master/model.py
# The only modification I made was to change the loss function from CrossEntropyLoss to BCEWithLogitsLoss 
# because this model is used for binary segmentation with a single class.

# Helper functions to create convolutional layers and up-convolutions (transposed convolutions)
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    """
    3x3 Convolutional layer with padding and optional bias and groups.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    """
    2x2 Up-convolution (also called transposed convolution).
    Alternatively, uses bilinear upsampling followed by a 1x1 convolution if 'mode' is 'upsample'.
    """
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # Upsample the input and then apply 1x1 convolution to adjust the channels
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    """
    1x1 Convolutional layer used for adjusting the channel dimensions.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

class DownConv(nn.Module):
    """
    Down-sampling block consisting of 2 convolutions followed by a MaxPooling layer.
    ReLU activation is applied after each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        # Two 3x3 convolutions
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        before_pool = x  # Save the output before pooling (needed for skip connections)
        
        # Apply max-pooling if needed
        if self.pooling:
            x = self.pool(x)
        
        return x, before_pool

class UpConv(nn.Module):
    """
    Up-sampling block consisting of an up-convolution followed by 2 convolutions.
    ReLU activation is applied after each convolution.
    The upsampling method can be either transpose convolution or bilinear upsampling.
    """
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        # Up-convolution (transpose or bilinear upsampling)
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        # Define convolution layers based on merge mode (concat or add)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """
        Forward pass of the upsampling block.
        Arguments:
            from_down: The feature map from the encoder (downward pathway).
            from_up: The feature map after upsampling (decoder pathway).
        """
        # Upsample the incoming feature map from the decoder
        from_up = self.upconv(from_up)
        
        # Merge the feature map from the encoder and the upsampled map
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)  # Concatenate along the channel dimension
        else:
            x = from_up + from_down  # Element-wise addition (used in residual connections)

        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        return x

class UNet(nn.Module):
    """
    The U-Net model, consisting of an encoder and decoder with skip connections.
    Designed for image segmentation tasks (e.g., binary segmentation with a single class).
    """
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            num_classes: The number of output classes (e.g., 1 for binary segmentation).
            in_channels: The number of input channels (3 for RGB images).
            depth: The number of down-sampling layers in the encoder.
            start_filts: The number of filters in the first convolutional layer.
            up_mode: The method used for upsampling ('transpose' or 'upsample').
            merge_mode: How to merge the skip connections ('concat' or 'add').
        """
        super().__init__()

        # Validate the upsampling and merging modes
        if up_mode not in ('transpose', 'upsample'):
            raise ValueError(f"\"{up_mode}\" is not a valid mode for upsampling. Only \"transpose\" and \"upsample\" are allowed.")
        
        if merge_mode not in ('concat', 'add'):
            raise ValueError(f"\"{merge_mode}\" is not a valid mode for merging skip connections. Only \"concat\" and \"add\" are allowed.")
        
        if up_mode == 'upsample' and merge_mode == 'add':
            raise ValueError("The combination of 'upsample' and 'add' is not supported because it does not make sense to add feature maps of different depths.")

        # Initialize model parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # Create the encoder pathway (down-sampling layers)
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False  # No pooling in the last down-sample layer

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # Create the decoder pathway (up-sampling layers)
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2  # The number of channels halves in the decoder
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # Final 1x1 convolution to reduce channels to the number of classes
        self.conv_final = conv1x1(outs, self.num_classes)

        # Convert lists to nn.ModuleList for PyTorch module tracking
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        """
        Initialize the weights of convolutional layers using Xavier initialization.
        Bias is initialized to zero.
        """
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        """
        Apply weight initialization to all layers in the model.
        """
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):
        """
        Forward pass through the network.
        Arguments:
            x: The input tensor (e.g., an image).
        """
        encoder_outs = []

        # Encoder pathway (down-sampling)
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)  # Save the feature map before pooling for skip connections

        # Decoder pathway (up-sampling)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]  # Skip connection from the encoder
            x = module(before_pool, x)

        # Final convolution to get the output with the required number of classes
        x = self.conv_final(x)
        
        return x
