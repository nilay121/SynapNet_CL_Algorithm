import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.optim import Adam
from collections import OrderedDict

class WorkingModel(nn.Module): 
    def __init__(self,output_size,channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        output_size = output_size 

        self.model = nn.Sequential(OrderedDict([
                ("conv1",nn.Conv2d(self.channel_dim, 16, stride=1, kernel_size=(3, 3), padding=1)),
                ("batch1", nn.BatchNorm2d(num_features=16)),
                ("relu1", nn.ReLU(inplace=False)),

                ("conv2", nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=1)),
                ("batch2",nn.BatchNorm2d(num_features=32)),
                ("relu2",nn.ReLU(inplace=False)),

                ("conv3",nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=1)),
                ("batch3",nn.BatchNorm2d(num_features=64)),
                ("relu3",nn.ReLU(inplace=False)),

                ("conv4",nn.Conv2d(64, 128, stride=2, kernel_size=(3, 3), padding=1)),
                ("batch4",nn.BatchNorm2d(num_features=128)),
                ("relu4",nn.ReLU(inplace=False)),

                ("conv5",nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1)),
                ("batch5",nn.BatchNorm2d(num_features=254)),
                ("flatten",nn.Flatten()),

                ("Linear1",nn.Linear(1016, 2000)),
                ("lrelu1",nn.ReLU(inplace=False)),
                ("Linear2",nn.Linear(2000, 1000)),
                ("lrelu2",nn.ReLU(inplace=False)),
                ("Linear3",nn.Linear(1000,output_size))])
        )   

    def forward(self, x):  
        return self.model(x)