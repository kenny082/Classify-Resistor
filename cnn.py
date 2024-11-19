import torch
from torch import nn
import torch.nn.functional as F

class ResistorClassifier(nn.Module):
    def __init__(self):
        super(ResistorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=3)
        self.fc = nn.Linear(256, 4)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
