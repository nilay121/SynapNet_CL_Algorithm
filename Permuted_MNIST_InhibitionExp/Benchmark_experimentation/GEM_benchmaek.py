import torch
import numpy
from avalanche.training import GEM

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

class GEM_bench:
    def __init__(self,n_epochs,buffer_size,n_classes,input_channel=1):
        self.n_epochs = n_epochs
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.input_channel=input_channel
        self.device = "cuda"

    def gem_benchmark(self):
        model = Model(self.n_classes,self.input_channel) # CNN model 

        # print to stdout
        interactive_logger = InteractiveLogger()

        gem_model = GEM(model=model,optimizer=self.optimizer,criterion=self.criterion,device=self.device,train_mb_size=32,
        eval_mb_size=32)

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
