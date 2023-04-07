from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from sampling_technique import Buffer
#from cls_algo import CustomStrategy
from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST
from EWC_replay import EWC_replay
from Lwf_replay import Lwf_replay
import numpy as np
import matplotlib.pyplot as plt
from cls_inhibition_algo import CustomInhibitStrategy
import torch
import torchvision
import pickle
import optuna
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.utils import AvalancheDataset

class Hyperparametr:
    def __init__(self):
        pass

    def dataGeneration(self):

        train_transform = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])

        test_transform = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])

        mnist_train = torchvision.datasets.MNIST('./data/',train=True,download=True,transform=train_transform)
        mnist_test = torchvision.datasets.MNIST('./data/',train=False,download=True,transform=test_transform)

        VAL_SIZE = 0.2
        BATCH_SIZE = 64
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices, _, _ = train_test_split(range(len(mnist_train)),mnist_train.targets,stratify=mnist_train.targets,
        test_size=VAL_SIZE,)

        # generate subset based on indices
        train_split = Subset(mnist_train, train_indices)
        val_split = Subset(mnist_train, val_indices)

        scenario_trainVal = nc_benchmark(train_split.dataset, val_split.dataset, n_experiences=5, shuffle=True, seed=None,
        fixed_class_order=[1,3,4,6,7,0,8,2,9,5],task_labels=False)

        return scenario_trainVal

    def dataPrepToPlot(self,acc_dict):
        y_stable=[]
        y_plastic=[]
        cls_output = []

        for i in range(0,len(acc_dict)):
            y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
            y_plastic.append(np.array(list(acc_dict.values())[i][1].cpu()))
        '''
        The accuracy of the plastic model for the recent experiences are better than the stable model,
        whereas the accuracy of the stable model on the old experiences are better.
        We use both the stable model and the plastic model and store the accuracy that is the highest from 
        either of the model
        '''
        for outputs in range(len(y_stable)):
            if (y_plastic[outputs]>y_stable[outputs]):
                cls_output.append(y_plastic[outputs])
            else:
                cls_output.append(y_stable[outputs])

        y_stable = np.array(y_stable)
        y_plastic = np.array(y_plastic)
        cls_output = np.array(cls_output)
        return np.round(y_stable,decimals=6),np.round(y_plastic,decimals=6),np.round(cls_output,decimals=6)

    def objective(self,trial):

        params = {
            "stable_model_update_freq": trial.suggest_float("stable_model_update_freq", 0.65, 0.85, step=0.05),
            "plastic_model_update_freq":trial.suggest_float("plastic_model_update_freq", 0.8, 1.0, step=0.05),
          #  "total_epochs": trial.suggest_int("total_epochs",5,20,5),   #don't use num_epochs, it matches with some reserved names and throws error
            "reg_weight": trial.suggest_float("reg_weight", 0.75, 0.95, step=0.05),
            "patience": trial.suggest_int("patience",3,7,2),  # patience has very little impact, and a value of 3 is ideal for most of the cases
        #    "learning_rate":trial.suggest_float("learning_rate",1e-3,1e-1,step=None,log=True),
            "inhibit_factor":trial.suggest_float("inhibit_factor",1e-2,3*1e-1,step=None,log=True), # using a log uniform distribution to find the parameter
            "rho":trial.suggest_float("rho", 0.5, 3, step=0.5)

        }
        stablePredN = []
        plasticPredN = []
        cls_outputPredN = []
        
        total_epochs= 15      #params['total_epochs']
        buffer_size=500
        n_classes=10
        num_runs=1

        stable_model_update_freq=params['stable_model_update_freq']
        plastic_model_update_freq=params['plastic_model_update_freq']
        n_experiences=5
        learning_rate = 1e-3    #params['learning_rate']
        reg_weight = params['reg_weight']
        patience = params['patience']


        inhibit_factor=params['inhibit_factor']    #0.001
        to_do = True # Start Inhibition
        rho=params['rho']

    
        scenario_trainVal = self.dataGeneration()     #getting the scenario

        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        num_epochs=total_epochs,reg_weight=reg_weight,buffer_size=buffer_size,inhibit_factor=inhibit_factor,rho=rho,n_classes=n_classes,to_do=to_do,
        n_channel=1,learning_rate=learning_rate,patience=patience) #CLS model

        ## Training and Evaluation for Custom Method
        results = []
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            print(len(experience.dataset))
            cl_strategy.train(experience) # Custom Strategy
            #print('Training completed')
            #print('Computing accuracy on the whole test set')
            final_accuracy,acc_dict = cl_strategy.evaluate(val_stream)
            results.append(final_accuracy)

        y_stable,y_plastic,cls_output = self.dataPrepToPlot(acc_dict)

        stablePredN.append(y_stable)
        plasticPredN.append(y_plastic)
        cls_outputPredN.append(cls_output)

        #Save a trained model to a file.
        torch.save(cl_strategy, "models/tunedIFmodelbuffer200{}.pickle".format(trial.number))

        meanStablePred = np.sum(stablePredN)/n_experiences    #Mean after n experiences
        meanPlasticPred = np.sum(plasticPredN)/n_experiences
        meanClsOutput = np.sum(cls_outputPredN)/n_experiences

        #average_scoreStable = np.sum(meanStablePred)/n_experiences

        print(f"The mean value after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        print(f"The mean value after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")

        return meanStablePred

    


def main():
    

    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    # Optuna trainer

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=20)

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)


if __name__=="__main__":
    main()
