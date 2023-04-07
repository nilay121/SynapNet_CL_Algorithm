from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST
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
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.utils import AvalancheDataset

from vae_model import VAE
from vae_training import Vae_Cls_Generator
from utils import utility_funcs

class Hyperparametr:
    def __init__(self):
        pass

    def dataGeneration(self):
        
        #Tranformations for the GR model
        train_transformGR = Compose([ToTensor()])
        test_transformInput = Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])

        fmnist_train = torchvision.datasets.FashionMNIST('./data/',train=True,download=True,transform=train_transformGR)  #shape(1,28,28)
        #fmnist_test = torchvision.datasets.FashionMNIST('./data/',train=False,download=True,transform=test_transformInput) 

        VAL_SIZE = 0.2
        BATCH_SIZE = 64
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices, _, _ = train_test_split(range(len(fmnist_train)),fmnist_train.targets,stratify=fmnist_train.targets,
        test_size=VAL_SIZE,)

        # generate subset based on indices
        train_split = Subset(fmnist_train, train_indices)
        val_split = Subset(fmnist_train, val_indices)

        scenario_trainVal = nc_benchmark(train_split.data, val_split.data, n_experiences=5, shuffle=False, seed=None, fixed_class_order=[0,1,2,3,4,5,6,7,8,9],task_labels=False)

        return scenario_trainVal

    def dataPrepToPlot(self,acc_dict,exp_numb):
        y_stable=[]
        y_plastic=[]
        cls_output = []

        for i in range(0,len(acc_dict)):
            y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
            y_plastic.append(np.array(list(acc_dict.values())[i][1].cpu()))

        for outputs in range(len(y_stable)):
            '''
                taking the last output from the plastic model instead of the stable model
            '''
            if (outputs==(exp_numb-1)):
                cls_output.append(y_plastic[outputs])
            else:
                cls_output.append(y_stable[outputs])

        y_stable = np.array(y_stable)
        y_plastic = np.array(y_plastic)
        cls_output = np.array(cls_output)
        return np.round(y_stable,decimals=6),np.round(y_plastic,decimals=6),np.round(cls_output,decimals=6)

    def objective(self,trial):

        params = {
             "total_epochs": trial.suggest_int("total_epochs", 15, 30, step = 5), #cHECKING HOW model behaves without the step
             "gradMaskEpoch":trial.suggest_int("gradMaskEpoch", 3, 15, step = 3),
          #  "total_epochs": trial.suggest_int("total_epochs",5,20,5),   #don't use num_epochs, it matches with some reserved names and throws error
           # "reg_weight": trial.suggest_float("reg_weight", 0.75, 0.95, step = 0.05),
        #    "patience": trial.suggest_int("patience",3,7,2),  # patience has very little impact, and a value of 3 is ideal for most of the cases
        #    "learning_rate":trial.suggest_float("learning_rate",1e-3,1e-1,step=None,log=True),
           #  "inhibit_factor":trial.suggest_float("inhibit_factor",1e-2,3*1e-1,step=None,log=True), # using a log uniform distribution to find the parameter
           # "rho":trial.suggest_float("rho", 0.5, 3, step=0.5),
          #  "batch_sizeCLS": trial.suggest_int("batch_sizeCLS",32,64,32),
          #  "mini_batchGR": trial.suggest_int("mini_batchGR",32,64,32)


        }

        total_epochs= 15#params['total_epochs']
        n_classes=10
        gradMaskEpoch = params['gradMaskEpoch']
        length_LIC = 3

        device = "cuda"
        n_experiences=5
        batch_sizeCLS = 32#64#params['batch_sizeCLS']  #64
        mini_batchGR = 32#64#params['mini_batchGR']  #64

        stable_model_update_freq = 0.15#params['stable_model_update_freq']
        plastic_model_update_freq = 0.90#params['plastic_model_update_freq']
        reg_weight = 0.75 #params['reg_weight']
        
        learning_rate = 1e-4  #0.1   #params['learning_rate']
        patience = 35 #params['patience']

        buffer_images = []
        buffer_labels = []

        #################### Hyperparameters Generator #########################
        learning_rateGR = 1e-4
        batch_sizeGR = 32 #128
        num_epochsGR = 40#30#50
        device = "cuda"
        patienceGR = 80

        synthetic_imgHeight = 28
        synthetic_imgWidth = 28
        img_channel_dim = 1
        latent_embedding = 100

        # buffer size = num_syntheticExamplesPerDigit * 10
        num_syntheticExamplesPerDigit = 500

        num_originalExamplesPerDigit = 10
        
        ## Train data Transformation
        train_transformBuffer = Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        ## Test data Transformation
        train_transformInput = Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
       
        scenario_trainVal = self.dataGeneration()     
        #getting the scenario
        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        ## CLS model
        cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        num_epochs=total_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
        n_channel=img_channel_dim,patience=patience,learning_rate=learning_rate,mini_batchGR=mini_batchGR,train_transformBuffer=train_transformBuffer,
        train_transformInput=train_transformInput,gradMaskEpoch=gradMaskEpoch,clipping=False,length_LIC=length_LIC) 

        ## Generator model
        modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,img_width=synthetic_imgWidth)
        Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                                batch_size=batch_sizeGR,device=device,patience=patienceGR,img_channel_dim=img_channel_dim)

        ## Training and Evaluation for Custom Method
        results = []
        exp_numb = 0
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
            cl_strategy.train(experience,synthetic_imgHeight=synthetic_imgHeight,
            synthetic_imgWidth=synthetic_imgWidth,buf_inputs=buffer_images,buf_labels=buffer_labels) # Comment for running pre trained cl model
            print('Training completed')

            ## Sleep
            if (exp_numb == n_experiences-1):  
                print("Starting offline learning for reorganizing memories")
                cl_strategy.offline_reorganizing(buf_inputs=buffer_images,buf_labels=buffer_labels,synthetic_imgHeight=synthetic_imgHeight,
                synthetic_imgWidth=synthetic_imgWidth,epochs=30,lr_offline=1e-4,offline_batch=32,patience=7) 
                print("Reorganization done")

            print('Computing accuracy on the whole test set')
            final_accuracy,acc_dict,_,_ = cl_strategy.evaluate(val_stream,validationFlag = False)
            results.append(final_accuracy)
            exp_numb+=1

        y_stable,y_plastic,cls_output = self.dataPrepToPlot(acc_dict,exp_numb=exp_numb)

        #Mean after n experiences
        meanStablePred = np.sum(y_stable)/n_experiences    
        meanPlasticPred = np.sum(y_plastic)/n_experiences
        meanClsOutput = np.sum(cls_output)/n_experiences

        print(f"The mean value after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        print(f"The mean value after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")

        return meanClsOutput


def main():
    
    ##### Optuna trainer
    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=19)

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)

    # saving the plots for intermediate values
    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["total_epochs", "gradMaskEpoch"])
    plt.tight_layout()
    plt.savefig(f"tb_results/optuna_plotfmnist.png")

if __name__=="__main__":
    main()
