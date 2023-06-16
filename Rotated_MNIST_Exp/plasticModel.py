import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import OrderedDict

class PlasticModel(nn.Module): 
    def __init__(self,n_classes,channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        output_size = n_classes

        self.model = nn.Sequential(
                nn.Conv2d(self.channel_dim, 16, stride=1, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(64, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),
                nn.Linear(1016, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size)
        )   

    def forward(self, x):  
        return self.model(x)