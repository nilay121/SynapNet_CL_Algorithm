'''
SynapNet is Brain-Inspired Complementary Learning System implementation with a fast Learner (hippocampus), 
a slow learner (Neocortex), along with a variational autoencoder (VAE) based pseudo memory for rehearsal. 
In addition, we incorporate  lateral inhibition masks on convolutional layer gradients to suppress neighboring 
neuron activity and a sleep phase for reorganizing learned representations.
'''
from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST
import numpy as np
import matplotlib.pyplot as plt
from cls_inhibition_algo import CustomInhibitStrategy
import torch
import torchvision
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

from vae_model import VAE
from vae_training import Vae_Cls_Generator
from utils import utility_funcs


def singleRun(n_experiences):

    ## Lateral Inhibition Parameters
    toDo_supression = True
    gradMaskEpoch = 6 
    length_LIC = 7 
    avg_term = 0.2
    diff_term = 0.8
    
    ## CLS Model Parameters
    num_epochs = 15
    n_classes=10
    device = "cuda"
    n_experiences = n_experiences
    batch_sizeCLS = 32
    mini_batchGR = 32

    stable_model_update_freq = 0.15 
    plastic_model_update_freq = 0.90 
    reg_weight = 0.75 
    
    patience = 20 
    learning_rate = 1e-4 

    ## Generator Parameters
    learning_rateGR = 0.0001 #0.001
    batch_sizeGR = 32 #128
    num_epochsGR = 40
    patienceGR = 80

    synthetic_imgHeight = 28
    synthetic_imgWidth = 28
    img_channel_dim = 1
    latent_embedding = 100

    num_syntheticExamplesPerDigit = 250#50#100#500
    num_originalExamplesPerDigit = 10

    #Buffer data transformations
    train_transformBuffer = Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    #Train data transformations
    train_transformInput = Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    #Test data transformations
    test_transformInput = Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    
    #Tranformations for the GR model
    train_transformGR = Compose([ToTensor()])

    fmnist_train = torchvision.datasets.FashionMNIST('./data/',train=True,download=True,transform=train_transformGR)  #shape(1,28,28)
    fmnist_test = torchvision.datasets.FashionMNIST('./data/',train=False,download=True,transform=test_transformInput)       

    # Initialize CL scenario
    scenario_trainTest = nc_benchmark(fmnist_train, fmnist_test, n_experiences=n_experiences, shuffle=False,seed=None,fixed_class_order=[0,1,2,3,4,5,6,7,8,9],
                                      task_labels=False) # Add fixed_class_order=[0,1,2,3,4,5,6,7,8,9] for fixed clsss runing

    train_stream = scenario_trainTest.train_stream
    test_stream =  scenario_trainTest.test_stream

    ## Initialize CLS model
    cl_strategy = CustomInhibitStrategy(working_model=WorkingModel,modelstable=StableModel,modelplastic=PlasticModel,\
    stable_model_update_freq=stable_model_update_freq,plastic_model_update_freq=plastic_model_update_freq,\
    num_epochs=num_epochs,reg_weight=reg_weight,batch_size=batch_sizeCLS,n_classes=n_classes,
    n_channel=img_channel_dim,patience=patience,learning_rate=learning_rate,mini_batchGR=mini_batchGR,train_transformBuffer=train_transformBuffer,
    train_transformInput=train_transformInput,gradMaskEpoch=gradMaskEpoch,clipping=False,length_LIC=length_LIC, avg_term = avg_term, 
    diff_term=diff_term, toDo_supression=toDo_supression) #CLS strategy

    ## Load saved CLS model    
    #cl_strategy = torch.load("best_models/VaeModelBuffer50039.pickle")

    ## Initialize Generator model
    modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,img_width=synthetic_imgWidth)
    Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                            batch_size=batch_sizeGR,device=device,patience=patienceGR,img_channel_dim=img_channel_dim)

    ## Training and Evaluation
    print('Starting experiment...')
    results = []
    buffer_images = []
    buffer_labels = []
    exp_numb = 0
    labelsForCF = []
    for experience in train_stream:

        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        print("Training Generator on current experience") 
        Vae_Cls_Obj.train(experience)
        
        for digit in experience.classes_in_this_experience:
            temp_img, temp_labels = utility_funcs.buffer_dataGeneration(digit=digit, experience=experience, num_examples=num_syntheticExamplesPerDigit,
                                            device=device,model=modelGR,numbOf_orgExamples=num_originalExamplesPerDigit)
            buffer_images.append(temp_img)
            buffer_labels.append(temp_labels)
            labelsForCF.append(digit)

        print("Generator tarining completed")
        print("Training CL model on current experience")

        # Train the CL Model
        cl_strategy.train(experience,synthetic_imgHeight=synthetic_imgHeight,
        synthetic_imgWidth=synthetic_imgWidth,buf_inputs=buffer_images,buf_labels=buffer_labels)     
        print('Training completed')     
        
        # Sleep Phase
        if (exp_numb == n_experiences-1):  
            print("Starting offline learning for reorganizing memories")
            cl_strategy.offline_reorganizing(buf_inputs=buffer_images,buf_labels=buffer_labels,synthetic_imgHeight=synthetic_imgHeight,
            synthetic_imgWidth=synthetic_imgWidth,epochs=30,lr_offline=1e-4,offline_batch=32,patience=7) 
            print("Reorganization done")
            
        print('Computing accuracy on the whole test set')
        final_accuracy, acc_dict, predictionsForCF_stable, predictionsForCF_plastic = cl_strategy.evaluate(test_stream)
        results.append(final_accuracy)
        exp_numb+=1

        ## Confusion Matrix for each experience
        print(" Plotting the confusion matrix for each experience ")
        utility_funcs.ConfusionMatrixPerExp(predictionsForCF_stable= predictionsForCF_stable,predictionsForCF_plastic = predictionsForCF_plastic 
        , ground_truth = test_stream, exp_numb=exp_numb, n_experiences=n_experiences)

    ##Saving the result for plots
    y_stable,y_plastic,cls_output = utility_funcs.dataPrepToPlot(acc_dict,exp_numb)

    ##Save a trained model to a file.
    # torch.save(cl_strategy, "models/VaeModelBufferFinalWithSleep.pickle")

    # ## Saving few images of each class from the buffer
    # utility_funcs.toPlotGRImages(buffer_images,image_height=synthetic_imgHeight,image_width=synthetic_imgWidth,img_channel=img_channel_dim
    # ,step_size=num_syntheticExamplesPerDigit)

    return y_stable,y_plastic,cls_output

def main():
    stablePredN = []
    plasticPredN = []
    cls_outputPredN = []
    num_runs = 3
    n_experiences = 5
    counter=0
    for i in range(num_runs):
        print("*"*10)
        print(f" Starting Repeatation Number {counter} out of {num_runs}")
        print("*"*10)
        y_stable, y_plastic, cls_output = singleRun(n_experiences=n_experiences)

        stablePredN.append(y_stable)
        plasticPredN.append(y_plastic)
        cls_outputPredN.append(cls_output)
        counter+=1

    #Mean, std after N runs runs for n experinces
    meanStablePred = np.round(np.sum(stablePredN,axis=0)/num_runs,decimals=2)    
    meanPlasticPred = np.round(np.sum(plasticPredN,axis=0)/num_runs,decimals=2)
    meanClsOutput = np.round(np.sum(cls_outputPredN,axis=0)/num_runs, decimals=2)

    stdStablePred = np.round(np.std(stablePredN,axis=0), decimals=4)   
    stdPlasticPred = np.round(np.std(plasticPredN,axis=0), decimals=4)
    stdClsOutput = np.round(np.std(cls_outputPredN,axis=0), decimals=4)

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} for stable model is {np.sum(meanStablePred)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} for stable model is {np.sum(stdStablePred)/n_experiences}")

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} for plastic model is {np.sum(meanPlasticPred)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} for plastic model is {np.sum(stdPlasticPred)/n_experiences}")

    print(f"The mean accuracy after {n_experiences} experiences for {num_runs} CLS output model is {np.sum(meanClsOutput)/n_experiences}")
    print(f"The Corresponding std. {n_experiences} experiences for {num_runs} CLS output model is {np.sum(stdClsOutput)/n_experiences}")

    utility_funcs.barPlotMeanPred(y_plotPlastic= meanPlasticPred, y_plotStable = meanStablePred, y_clsOutput= meanClsOutput,stdStablePred=stdStablePred,
    stdPlasticPred=stdPlasticPred,stdClsOutput=stdClsOutput,n_experinces = n_experiences )


if __name__=="__main__":
    main()
