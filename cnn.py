# Import necessary libraries for computer vision, numerical operations, plotting, and deep learning.
import cv2 as cv  # OpenCV library for image processing
import os  # Library for interacting with the file system
import numpy as np  # Numerical computations and array manipulations
import matplotlib.pyplot as plt  # Plotting library for visualization
import torch  # PyTorch library for deep learning
from torch import nn  # Neural network module from PyTorch
import torch.nn.functional as F  # Functional API for activation functions and more

# Define the ResistorClassifier class, inheriting from PyTorch's nn.Module base class.
class ResistorClassifier(nn.Module):
    def __init__(self):
        # Call the parent class (nn.Module) constructor.
        super(ResistorClassifier, self).__init__()
        
        # Define the neural network layers as a sequential module.
        self.network = nn.Sequential(
            # First convolutional layer: 3 input channels (RGB), 32 output channels, kernel size 5x5, stride 5.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=5),
            # ReLU activation function applied in place for efficiency.
            nn.ReLU(inplace=True),
            
            # Second convolutional layer: 32 input channels, 64 output channels, kernel size 3x3, stride 3.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3),
            # ReLU activation function applied in place.
            nn.ReLU(inplace=True),
            
            # Third convolutional layer: 64 input channels, 256 output channels, kernel size 3x3, stride 3.
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=3),
            # ReLU activation function applied in place.
            nn.ReLU(inplace=True),
            
            # Flatten the feature maps into a 1D vector starting from the first dimension (batch size excluded).
            nn.Flatten(start_dim=1),
            
            # Fully connected layer: 256 input features, 4 output classes
            nn.Linear(256, 4, bias=True),
        )
        
        # Initialize weights for the network.
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(layer):
            # Check if the layer is a convolutional layer.
            if isinstance(layer, nn.Conv2d):
                # Apply Xavier initialization to the weights.
                nn.init.xavier_uniform_(layer.weight)
                # Initialize biases to zeros if biases exist.
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Apply the `init_weights` function to all layers in the module.
        self.apply(init_weights)

    def forward(self, inputs):
        # Pass the input data through the defined network layers.
        return self.network(inputs)
