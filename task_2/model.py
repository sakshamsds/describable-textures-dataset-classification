import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()     
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=32)
        self.fc1 = torch.nn.Linear(in_features=1*8*8, out_features=47)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
