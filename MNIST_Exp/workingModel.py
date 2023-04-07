import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import OrderedDict

class WorkingModel(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, (3,3))
        self.pool = nn.MaxPool2d(2,1,1)
        self.fc1 = nn.Linear(32 * 27 * 27, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = x.reshape(-1, 32 * 27 * 27)          
        x = F.relu(self.fc1(x))             
        x = F.relu(self.fc2(x))              
        x = self.fc3(x)                       
        return x