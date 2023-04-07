from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin,ReplayPlugin,EWCPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training import Naive
import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
import torchvision


class ModelLWF(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()
        self.flag = False
        self.channel_dim = input_channel
        output_size = output_size

        self.loss_inhibit_num=0
        self.loss_inhibit_deno=0
        self.loss_inihibt_sum=0
        
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

                nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),
            
                # nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm2d(num_features=128),
                # nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),

                nn.Linear(254, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size),
        )   

    def forward(self, x):  
        return self.model(x)


class ModelEWC(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()
        self.flag = False
        self.channel_dim = input_channel
        output_size = output_size

        self.loss_inhibit_num=0
        self.loss_inhibit_deno=0
        self.loss_inihibt_sum=0
        
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

                nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),
            
                # nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm2d(num_features=128),
                # nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),

                nn.Linear(254, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size),
        )    

    def forward(self, x):  
        return self.model(x)

class ModelSI(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()
        self.channel_dim = input_channel
        output_size = output_size
        
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

                nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),
            
                # nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm2d(num_features=128),
                # nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),

                nn.Linear(254, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size),
        )    

    def forward(self, x):  
        return self.model(x)
    
    
class ModelJoint(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()
        self.channel_dim = input_channel
        output_size = output_size

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

                nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),
            
                # nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm2d(num_features=128),
                # nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),

                nn.Linear(254, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size),
        )  

    def forward(self, x):  
        return self.model(x)
    

class ModelNaive(nn.Module): 
    def __init__(self,output_size,input_channel):
        super().__init__()

        self.channel_dim = input_channel
        output_size = output_size

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

                nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),
            
                # nn.Conv2d(128, 128, stride=2, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm2d(num_features=128),
                # nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),

                nn.Linear(254, 2000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,output_size),
        )   

    def forward(self, x):  
        return self.model(x)

