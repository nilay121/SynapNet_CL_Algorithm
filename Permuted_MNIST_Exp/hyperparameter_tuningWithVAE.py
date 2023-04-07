from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST,PermutedMNIST
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
        self.buffer_images = []
        self.buffer_labels = []

    def dataGeneration(self):
        n_experiences = 5
        
        train_transformInput = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])
        test_transformInput = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])

        scenario_trainVal = PermutedMNIST(n_experiences=n_experiences,train_transform=train_transformInput,eval_transform=test_transformInput,seed=9)

        # train_stream = scenario_trainTest.train_stream
        # test_stream =  scenario_trainTest.test_stream
        return scenario_trainVal

    def dataPrepToPlot(self,acc_dict,exp_numb):
        y_stable=[]
        y_plastic=[]
        cls_output=[]

        for i in range(0,len(acc_dict)):
            y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
            y_plastic.append(np.array(list(acc_dict.values())[i][1].cpu()))

        for outputs in range(len(y_stable)):

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
             "stable_model_update_freq": trial.suggest_float("stable_model_update_freq", 0.25, 0.45, step=0.05),
             "plastic_model_update_freq":trial.suggest_float("plastic_model_update_freq", 0.75, 0.95, step=0.05),
          #  "total_epochs": trial.suggest_int("total_epochs",5,20,5),   #don't use num_epochs, it matches with some reserved names and throws error
            "reg_weight": trial.suggest_float("reg_weight", 0.2, 0.4, step=0.05),
        #    "patience": trial.suggest_int("patience",3,7,2),  # patience has very little impact, and a value of 3 is ideal for most of the cases
        #    "learning_rate":trial.suggest_float("learning_rate",1e-3,1e-1,step=None,log=True),
            # "inhibit_factor":trial.suggest_float("inhibit_factor",1e-2,3*1e-1,step=None,log=True), # using a log uniform distribution to find the parameter
            #"rho":trial.suggest_float("rho", 0.5, 3, step=0.5),
          #  "batch_sizeCLS": trial.suggest_int("batch_sizeCLS",32,64,32),
          #  "mini_batchGR": trial.suggest_int("mini_batchGR",32,64,32)


        }
        
        total_epochs= 35      #params['total_epochs']
        n_classes=10

        device = "cuda"
        n_experiences=5
        batch_sizeCLS = 1024#32#64 # DOnot increase the batch size or decease or increase the lr
        mini_batchGR = 128#32#64

        stable_model_update_freq = params['stable_model_update_freq']#0.10#
        plastic_model_update_freq = params['plastic_model_update_freq'] #0.65 #
        reg_weight = params['reg_weight']  #0.75#
        
        learning_rate = 1e-2    #params['learning_rate']
        patience = 15 #params['patience']

        #################### Hyperparameters Generator #########################
        learning_rateGR = 0.0001 #0.0001 #0.001
        batch_sizeGR = 32 #128
        num_epochsGR = 15
        patienceGR = 50  # No patience

        synthetic_imgHeight = 28
        synthetic_imgWidth = 28
        img_channel_dim = 1
        latent_embedding = 100
        
        # buffer size = num_syntheticExamplesPerDigit * 10
        num_syntheticExamplesPerDigit = 100#50

        num_originalExamplesPerDigit = 3

        bufferSizePMNIST = 5000
        
        # #Buffer transformations
        # train_transformBuffer = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])

        # #Train data transformations
        # train_transformInput = Compose([transforms.ToPILImage(),ToTensor(),Normalize((0.1307,), (0.3081,))])
        
        scenario_trainVal = self.dataGeneration()     
        
        #getting the scenario
        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        # CLS model
        # cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        # stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        # num_epochs=total_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
        # n_channel=1,patience=patience,learning_rate=learning_rate,mini_batchGR=mini_batchGR,train_transformBuffer=train_transformBuffer,
        # train_transformInput=train_transformInput,clipping=False) #CLS strategy
        cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        num_epochs=total_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
        n_channel=1,patience=patience,learning_rate=learning_rate,mini_batchGR=mini_batchGR,clipping=True) #CLS strategy


        # Generator model
        modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,batch_size=batch_sizeGR,img_width=synthetic_imgWidth)
        Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                                batch_size=batch_sizeGR,device=device,patience=patienceGR)

        ## Training and Evaluation for Custom Method
        results = []
        exp_numb = 0
        for experience in train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print("Training Generator on current experience")

            Vae_Cls_Obj.train(experience)
            for digit in experience.classes_in_this_experience:
                temp_img, temp_labels = utility_funcs.buffer_dataGeneration(digit=digit, experience=experience, num_examples=num_syntheticExamplesPerDigit,
                                                device=device,model=modelGR,numbOf_orgExamples=num_originalExamplesPerDigit)
                self.buffer_images.append(temp_img)
                self.buffer_labels.append(temp_labels)
            print("Training CL model on current experience")

            cl_strategy.train(experience,synthetic_imgHeight=synthetic_imgHeight,
            synthetic_imgWidth=synthetic_imgWidth,buf_inputs=self.buffer_images,buf_labels=self.buffer_labels) # Comment for running pre trained cl model
            print('Training completed')

            # Sleep Phase
            if (exp_numb == n_experiences-1):  
                print("Starting offline learning for reorganizing memories")
                cl_strategy.offline_reorganizing(buf_inputs=self.buffer_images,buf_labels=self.buffer_labels,synthetic_imgHeight=synthetic_imgHeight,
                synthetic_imgWidth=synthetic_imgWidth,epochs=20,lr_offline=1e-4,offline_batch=32,patience=7) 
                print("Reorganization done")

            print('Computing accuracy on the whole test set')
            final_accuracy,acc_dict,_,_ = cl_strategy.evaluate(val_stream,validationFlag = False)
            results.append(final_accuracy)
            exp_numb+=1

        y_stable,y_plastic,cls_output = self.dataPrepToPlot(acc_dict,exp_numb)

        # stablePredN.append(y_stable)
        # plasticPredN.append(y_plastic)
        # cls_outputPredN.append(cls_output)

        #Save a trained model to a file.
       # torch.save(cl_strategy, "models/permuted_MNIST{}.pickle".format(trial.number))

        #Mean after n experiences
        meanStablePred = np.sum(y_stable)/n_experiences    
        meanPlasticPred = np.sum(y_plastic)/n_experiences
        meanCLSPred = np.sum(cls_output)/n_experiences        

        #average_scoreStable = np.sum(meanStablePred)/n_experiences

        print(f"The mean value after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")
        # print(f"The Corresponding std. after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")
        # print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanCLSPred,decimals=4)}")
        # print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        #return meanStablePred
        return meanCLSPred


def main():
    
    ##########################################################
    ##### Optuna trainer

    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=12)

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)

    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["stable_model_update_freq", "plastic_model_update_freq","reg_weight"])
    plt.tight_layout()
    plt.savefig(f"tb_results/optuna_plotModified.png")

if __name__=="__main__":
    main()
