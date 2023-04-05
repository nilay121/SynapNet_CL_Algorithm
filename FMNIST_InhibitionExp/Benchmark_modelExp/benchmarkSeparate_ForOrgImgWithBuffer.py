from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST
from Benchmark_GenerativeModel import ModelLWF,ModelEWC,ModelSI
#from Lwf_Generativereplay import LWF

from avalanche.training import EWC
from avalanche.training import LwF
from avalanche.training import SynapticIntelligence
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin,ReplayPlugin,EWCPlugin,SynapticIntelligencePlugin,LwFPlugin
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.utils import AvalancheDataset
from torchvision import transforms

from vae_model import VAE
from vae_training import Vae_Cls_Generator
from utils import utility_funcs

def main():
    benchResultArrayEWC=[]
    benchResultArrayLWF=[]
    benchResultArraySI = []

    num_runs=1
    n_classes=10
    epochs = 30
    n_experiences=5
    device = "cuda"

    #  model parameters
    input_channel = 1

    #Train data transformations
    train_transformInput = Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])

    test_transformInput = Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    
    #Tranformations for the GR model
    train_transformGR = Compose([ToTensor()])
    fmnist_train = torchvision.datasets.FashionMNIST('./data/',train=True,download=True,transform=train_transformInput)  #shape(1,28,28)
    fmnist_test = torchvision.datasets.FashionMNIST('./data/',train=False,download=True,transform=test_transformInput) 

    scenario_trainTest = nc_benchmark(fmnist_train, fmnist_test, n_experiences=n_experiences, shuffle=False, seed=None,
                                    fixed_class_order=[0,1,2,3,4,5,6,7,8,9],task_labels=False)

    train_stream = scenario_trainTest.train_stream
    test_stream =  scenario_trainTest.test_stream

    interactive_logger = InteractiveLogger()
    modellwf = ModelLWF(output_size=n_classes,input_channel=input_channel)
    modelewc = ModelEWC(output_size=n_classes,input_channel=input_channel)
    modelsi = ModelSI(output_size=n_classes,input_channel=input_channel)

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

    for counter in range(num_runs):
        print("*"*20)
        print(f" Starting Repeatation Number {counter} out of {num_runs}")
        print("*"*20)

        buffer_images = []
        buffer_labels = []

        LwfModel = LwF(model=modellwf,optimizer=SGD(modellwf.parameters(), lr=0.001, momentum=0.9),train_mb_size=64,eval_mb_size=64,
        criterion=CrossEntropyLoss(),temperature=2.0,alpha=1.0,train_epochs=epochs,device=device) # EWC replay model

        ewcModel = EWC(model=modelewc,optimizer=SGD(modelewc.parameters(), lr=0.001, momentum=0.9),train_mb_size=64,eval_mb_size=64,
        ewc_lambda=100,criterion=CrossEntropyLoss(),train_epochs=epochs,device=device)
        #lambda=0.001

        siModel = SynapticIntelligence(model=modelsi,optimizer=SGD(modelewc.parameters(), lr=0.001, momentum=0.9),train_mb_size=64,eval_mb_size=64,
        si_lambda=2700,criterion=CrossEntropyLoss(),train_epochs=epochs,device=device) 

        ## Training and Evaluation
        print('Starting experiment...')
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print("Training Model on current experience")
            
            ewcModel.train(experience)  #Avalanche Benchmark strategy
            print('Training completed')
            print('Computing accuracy on the whole test set')
            ## eval also returns a dictionary which contains all the metric values
            print("computing accuracy for EWC Model")
            bench_resultsEWC = ewcModel.eval(test_stream)
            print("*"*20)


        #Saving the result for plots
        benchResultArrayEWC.append(dataPrepToPlot(bench_resultsEWC,len(train_stream)))
    meanBenchEWC = np.sum(benchResultArrayEWC,axis=0)/num_runs

    print(f"The mean value after 5 experinces for {num_runs} for EWC model is {np.sum(meanBenchEWC)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for EWC model model is {np.std(meanBenchEWC)/n_experiences}")

def dataPrepToPlot(bench_results,testDataExpLen):
    benchResultArray=[]
    for i in range (0,testDataExpLen):
        benchResultArray.append(bench_results["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00"+str(i)])
    return np.round(benchResultArray,decimals=2)


def barPlotMeanPred(meanBenchEWC,meanBenchLWF,meanBenchSI):
    N = 5
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots()
 
    bar_ewc = ax.bar(ind, meanBenchEWC, width, color = 'r',label="EWC Model")
    bar_lwf = ax.bar(ind+width, meanBenchLWF, width, color='g',label="LWF Model")
    bar_si = ax.bar(ind+2*width, meanBenchSI, width, color='b',label="SI Model")
    
    ax.bar_label(bar_ewc, padding=3)
    ax.bar_label(bar_lwf, padding=3)
    ax.bar_label(bar_si, padding=3)
    
    ax.set_title("FMNIST Buffer size 5000 Epochs 30")
    ax.set_xlabel("Experiences & Models")
    ax.set_ylabel("accuarcy")
    ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5"])
    ax.legend((bar_ewc, bar_lwf,bar_si), ('EWC Model', 'LWF Model','Synaptic Intelligence'),loc=4 )
    fig.tight_layout()
    plt.show()
    plt.savefig("FMNIST/buffer_size5000benchmark_after5runs.png")

if __name__=="__main__":
    main()
