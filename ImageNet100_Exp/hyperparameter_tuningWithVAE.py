from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
import numpy as np
import matplotlib.pyplot as plt
from cls_inhibition_algo import CustomInhibitStrategy
import torch
import torchvision
import pickle
import optuna
from utils import CustomDatasetForDataLoader

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

    def dataGeneration(self, n_experiences):               
        imagenet_train = torch.load("data/train_scenario_TIN.pt")
        imagenet_test = torch.load("data/test_scenario_TIN.pt")

        scenario_trainTest = nc_benchmark(imagenet_train, imagenet_test, n_experiences=n_experiences, 
                                        shuffle=True, seed=9,task_labels=False) 

        return scenario_trainTest

    def dataPrepToPlot(self,acc_dict,exp_numb):
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
            # if (y_plastic[outputs]>y_stable[outputs]):
            #     cls_output.append(y_plastic[outputs])
            # else:
            #     cls_output.append(y_stable[outputs])
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
             "stable_model_update_freq": trial.suggest_float("stable_model_update_freq", 0.5, 0.75, step=0.05),
             "plastic_model_update_freq":trial.suggest_float("plastic_model_update_freq", 0.7, 1.0, step=0.05),
          #  "total_epochs": trial.suggest_int("total_epochs",5,20,5),   #don't use num_epochs, it matches with some reserved names and throws error
            "reg_weight": trial.suggest_float("reg_weight", 0.1, 0.3, step=0.05),
        #    "patience": trial.suggest_int("patience",3,7,2),  # patience has very little impact, and a value of 3 is ideal for most of the cases
        #    "learning_rate":trial.suggest_float("learning_rate",1e-3,1e-1,step=None,log=True),
           #  "inhibit_factor":trial.suggest_float("inhibit_factor",1e-2,3*1e-1,step=None,log=True), # using a log uniform distribution to find the parameter
           # "rho":trial.suggest_float("rho", 0.5, 3, step=0.5),
          #  "batch_sizeCLS": trial.suggest_int("batch_sizeCLS",32,64,32),
          #  "mini_batchGR": trial.suggest_int("mini_batchGR",32,64,32)


        }
        ## Inhibition
        ## Lateral Inhibition Parameters
        toDo_supression = False
        gradMaskEpoch = 40
        length_LIC = 7
        avg_term = 0.2
        diff_term = 0.8

        self.buffer_images = []
        self.buffer_labels = []        
        total_epochs= 50#100      #params['total_epochs']
        clipping = True
        n_classes=100

        device = "cuda"
        n_experiences=5
        batch_sizeCLS = 128#64#params['batch_sizeCLS']  #64
        mini_batchGR = 64#64#params['mini_batchGR']  #64

        stable_model_update_freq = params['stable_model_update_freq']
        plastic_model_update_freq = params['plastic_model_update_freq']
        reg_weight = params['reg_weight']
        
        learning_rate = 1e-1   #params['learning_rate']
        patience = 30 #params['patience']

        #################### Hyperparameters Generator #########################
        learning_rateGR = 1e-4#0.0001 #0.001
        batch_sizeGR = 32 #128
        num_epochsGR = 100
        device = "cuda"
        patienceGR = 50

        synthetic_imgHeight = 32
        synthetic_imgWidth = 32
        img_channel_dim = 3
        latent_embedding = 100

        num_syntheticExamplesPerDigit = 50
        num_originalExamplesPerDigit = 50
        
        #################### Initialize Transformations #########################
        #Buffer transformations
        train_transformBuffer = Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        #Train data transformations
        train_transformInput = Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_transformInput = Compose([transforms.ToPILImage(), ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
        
        scenario_trainVal = self.dataGeneration(n_experiences=n_experiences)     
        
        #getting the scenario
        train_stream = scenario_trainVal.train_stream
        val_stream = scenario_trainVal.test_stream

        ###################################
        # CLS model
        ###################################
        cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
        stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
        num_epochs=total_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
        n_channel=img_channel_dim,patience=patience,learning_rate=learning_rate,mini_batchGR=mini_batchGR,train_transformBuffer=train_transformBuffer,
        train_transformInput=train_transformInput,gradMaskEpoch=gradMaskEpoch,clipping=clipping,length_LIC=length_LIC,avg_term = avg_term, 
        diff_term=diff_term,toDo_supression=toDo_supression, val_transformInput=val_transformInput)  #CLS strategy


        ################################ 
        # Generator model
        #################################
        modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,batch_size=batch_sizeGR,img_width=synthetic_imgWidth)
        Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                                batch_size=batch_sizeGR,device=device,patience=patienceGR,img_channel_dim=img_channel_dim)

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

            cl_strategy.train(experience,synthetic_imgHeight=synthetic_imgHeight,
            synthetic_imgWidth=synthetic_imgWidth,buf_inputs=self.buffer_images,buf_labels=self.buffer_labels) # Comment for running pre trained cl model
            print('Training completed')

            # **********************For sleep******************************** 
            if (exp_numb == n_experiences-1):  
                print("Starting offline learning for reorganizing memories")
                cl_strategy.offline_reorganizing(buf_inputs=self.buffer_images,buf_labels=self.buffer_labels,synthetic_imgHeight=synthetic_imgHeight,
                synthetic_imgWidth=synthetic_imgWidth,epochs=30,lr_offline=1e-4,offline_batch=32,patience=9) 
                print("Reorganization done")
            #########################################

            print('Computing accuracy on the whole test set')
            final_accuracy,acc_dict = cl_strategy.evaluate(val_stream,validationFlag = True)
            results.append(final_accuracy)
            exp_numb+=1

        y_stable,y_plastic,cls_output = self.dataPrepToPlot(acc_dict,exp_numb=exp_numb)

        # stablePredN.append(y_stable)
        # plasticPredN.append(y_plastic)
        # cls_outputPredN.append(cls_output)

        #Save a trained model to a file.
        torch.save(cl_strategy, "models/VaeModelBufferIF5k_WithS{}.pickle".format(trial.number))

        #Mean after n experiences
        meanStablePred = np.sum(y_stable)/n_experiences    
        meanPlasticPred = np.sum(y_plastic)/n_experiences
        meanClsOutput = np.sum(cls_output)/n_experiences

        #average_scoreStable = np.sum(meanStablePred)/n_experiences

        print(f"The mean value after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for stable model is {np.round(meanStablePred,decimals=4)}")

        print(f"The mean value after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for plastic model is {np.round(meanPlasticPred,decimals=4)}")

        print(f"The mean value after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")
        print(f"The Corresponding std. after 5 experinces for CLS output model is {np.round(meanClsOutput,decimals=4)}")

        #return meanStablePred
        return meanClsOutput


def main():
    
    ##########################################################
    ##### Optuna trainer
    ##########################################################

    hyperparametr_obj = Hyperparametr()
    StableAccuracyPerConfig = hyperparametr_obj.objective # objective function

    study = optuna.create_study(direction="maximize")
    study.optimize(StableAccuracyPerConfig, n_trials=15)

    print("best trial")
    trial_ = study.best_trial
    print(trial_.values)
    print("*"*20)
    print("best parameters")
    print(trial_.params)
    print("*"*20)

    # saving the plots for intermediate values
    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["stable_model_update_freq", "plastic_model_update_freq","reg_weight"])
    plt.tight_layout()
    plt.savefig(f"tb_results/optuna_plot.png")




if __name__=="__main__":
    main()
