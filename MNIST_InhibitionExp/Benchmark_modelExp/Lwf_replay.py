from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin,ReplayPlugin,EWCPlugin,SynapticIntelligencePlugin,LwFPlugin
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


class Model(nn.Module): 
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

class Lwf_replay:
    def __init__(self,n_epochs,buffer_size,n_classes,input_channel=1):
        self.n_epochs = n_epochs
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.input_channel=input_channel
    def Lwf_replay_benchmark(self):
        model = Model(self.n_classes,self.input_channel) # CNN model 

        # log to Tensorboard
        #tb_logger = TensorboardLogger()

        # log to text file
       # text_logger = TextLogger(open('log_ewc.txt', 'a'))

        # print to stdout
        interactive_logger = InteractiveLogger()

        replay = ReplayPlugin(mem_size=self.buffer_size)
        lwf = LwFPlugin()

        # DEFINE THE EVALUATION PLUGIN and LOGGERS
        # The evaluation plugin manages the metrics computation.
        # It takes as argument a list of metrics, collectes their results and returns 
        # them to the strategy it is attached to.

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=False, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=False, experience=True, stream=True),
        #    timing_metrics(epoch=True, epoch_running=True),
        #    forgetting_metrics(experience=True, stream=True),
        #    cpu_usage_metrics(experience=True),
        #    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
        #                             stream=True),
        #    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=[interactive_logger])

        # CREATE THE STRATEGY INSTANCE (NAIVE)
        cl_strategyBenchLwf = Naive(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=self.n_epochs, eval_mb_size=100,
            evaluator=eval_plugin,plugins=[lwf,replay],device= "cuda")
        return cl_strategyBenchLwf