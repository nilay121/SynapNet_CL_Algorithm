from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST,PermutedMNIST
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

import optuna


class Hyperparametr:
    def __init__(self):
        pass
    def dataGeneration(self,n_experiences):
        #Tranformations for the GR model
        #Tranformations for the GR model
        train_transformGR = Compose([ToTensor()])
        test_transformInput = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])

        mnist_train = torchvision.datasets.MNIST('./data/',train=True,download=True,transform=train_transformGR)
        mnist_test = torchvision.datasets.MNIST('./data/',train=False,download=True,transform=test_transformInput)


        scenario_trainVal = nc_benchmark(mnist_train, mnist_test, n_experiences=5, shuffle=True, seed=None,
        fixed_class_order=[0,1,2,3,4,5,6,7,8,9],task_labels=False)

        return scenario_trainVal

    def dataPrepToPlot(self, bench_results,testDataExpLen):
        benchResultArray=[]
        for i in range (0,testDataExpLen):
            benchResultArray.append(bench_results["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00"+str(i)])
        return np.round(benchResultArray,decimals=5)

    def objective(self,trial):

        params = {
            "temperature": trial.suggest_float("temperature",2,21,step = 3),
            "alpha": trial.suggest_float("alpha",1,20,step = 3)
            # "learning_rate":trial.suggest_float("learning_rate",1e-3,1e-1,step=None,log=True)

        }
        # stablePredN = []
        # plasticPredN = []
        # cls_outputPredN = []
        
        # total_epochs= 30      #params['total_epochs']
        # n_classes=10

        # device = "cuda"
        # n_experiences=5
        # batch_sizeCLS = 128#64#params['batch_sizeCLS']  #64
        # mini_batchGR = 128#64#params['mini_batchGR']  #64

        # stable_model_update_freq = params['stable_model_update_freq']#0.10#
        # plastic_model_update_freq = params['plastic_model_update_freq'] #0.65 #
        # reg_weight = params['reg_weight']  #0.75#
        
        # learning_rate = 1e-2    #params['learning_rate']
        # patience = 15 #params['patience']

        n_classes=10
        epochs = 25
        n_experiences=5
        train_batch_size = 64
        eval_batch_size = 64

        temperature = params['temperature']
        alpha = params['alpha']

        learning_rate = 1e-3

        #################### Hyperparameters Generator #########################
        learning_rateGR = 1e-4 
        batch_sizeGR = 32
        num_epochsGR = 15
        patienceGR = 50  # No patience

        synthetic_imgHeight = 28
        synthetic_imgWidth = 28

        img_channel_dim = 1
        latent_embedding = 100

        # buffer size = num_syntheticExamplesPerDigit * 10
        num_syntheticExamplesPerDigit = 500

        num_originalExamplesPerDigit = 10

        device = "cuda"

        buffer_images = []
        buffer_labels = []

        ####################################################################################
        #  model parameters
        input_channel = 1
        
        #################### Initialize Transformations #########################
        #Buffer transformations
        train_transformBuffer = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])

        #Train data transformations
        train_transformInput = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])

        
        
        scenario_trainVal = self.dataGeneration(n_experiences=n_experiences)     
        
        #getting the scenario
        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        ###################################
        # CLS model
        ###################################
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

        ## Benchamrk models Initialization

        LwfModel = LwF(model=modellwf,optimizer=SGD(modellwf.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size,
        eval_mb_size=eval_batch_size,criterion=CrossEntropyLoss(),temperature=temperature,alpha=alpha,train_epochs=epochs,device=device) 

        # ewcModel = EWC(model=modelewc,optimizer=SGD(modelewc.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size,
        # eval_mb_size=eval_batch_size, ewc_lambda=ewc_lambda,criterion=CrossEntropyLoss(),train_epochs=epochs,device=device)
        # #lambda=0.001

        # siModel = SynapticIntelligence(model=modelsi,optimizer=SGD(modelewc.parameters(), lr=0.001, momentum=0.9),train_mb_size=64,eval_mb_size=64,
        # si_lambda=1e3,criterion=CrossEntropyLoss(),train_epochs=epochs,device=device) 

        ################################ 
        # Generator model
        #################################
        modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,img_width=synthetic_imgWidth)
        Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                                batch_size=batch_sizeGR,device=device,patience=patienceGR,img_channel_dim=img_channel_dim)

        ## Training and Evaluation for Custom Method
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print("Training Generator on current experience")
            print("Training the Generator")
            Vae_Cls_Obj.train(experience)
            for digit in experience.classes_in_this_experience:
                temp_img, temp_labels = utility_funcs.buffer_dataGeneration(digit=digit, experience=experience, num_examples=num_syntheticExamplesPerDigit,
                                                device=device,model=modelGR,numbOf_orgExamples=num_originalExamplesPerDigit)
                buffer_images.append(temp_img)
                buffer_labels.append(temp_labels)
            print("Training CL model on current experience")

            ##################################################################
            # Benchmark and buffer data generation 
            ##################################################################
            print("generating benchmark dataset")
            exp_data = utility_funcs.benchmarkDataPrep(experience=experience,device=device,train_transformBuffer=train_transformBuffer,buffer_data=buffer_images,
            buffer_label=buffer_labels,synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,train_transformInput=train_transformInput)
            # for temp_data in DataLoader(exp_data,batch_size=len(exp_data)):
            scenario_exp = nc_benchmark(exp_data,exp_data,n_experiences=1,task_labels=False)
            data_expTrain = scenario_exp.train_stream
            print("----------------------------------Done----------------------------------")

            ## Training and Evaluation of Benchmark models

            for new_exp in data_expTrain:
                print("CLasses in the modified experience ",new_exp.classes_in_this_experience)
                # print("Training EWC Model")
                # ewcModel.train(new_exp)  #Avalanche Benchmark strategy
                # print('Training completed')

                print("Training LWF Model")
                LwfModel.train(new_exp)
                print('Training completed')

                # print("Training SI Model")
                # siModel.train(new_exp)
                # print('Training completed')

            ## Evaluation
            print('Computing accuracy on the whole test set')
            ## eval also returns a dictionary which contains all the metric values
            # print("computing accuracy for EWC Model")
            # bench_resultsEWC = ewcModel.eval(val_stream)
            # print("*"*20)

            print("computing accuracy for LWF Model")
            bench_resultsLWF = LwfModel.eval(val_stream)
            print("*"*20)

            # print("computing accuracy for SI Model")
            # bench_resultsSI = siModel.eval(val_stream)
            # print("*"*20)

        ##Extracting the results
        # MeanbenchResultArrayEWC = np.sum(self.dataPrepToPlot(bench_resultsEWC,len(train_stream)))/n_experiences

        MeanbenchResultArrayLWF = np.sum(self.dataPrepToPlot(bench_resultsLWF,len(train_stream)))/n_experiences

        # MeanbenchResultArraySI = self.dataPrepToPlot(bench_resultsSI,len(train_stream))
        
        #torch.save(cl_strategy, "models/VaeModelBufferIF5k_WithS{}.pickle".format(trial.number))
        print(f"The mean value after 5 experinces for stable model is {np.round(MeanbenchResultArrayLWF,decimals=4)}")

        return MeanbenchResultArrayLWF


def main():
    ## Optuna trainer

    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=12)#

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)

    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["temperature","alpha"])
    plt.tight_layout()
    plt.savefig(f"tb_results/MNISToptuna_plotBenchmarkModelLWF.png")

if __name__=="__main__":
    main()
