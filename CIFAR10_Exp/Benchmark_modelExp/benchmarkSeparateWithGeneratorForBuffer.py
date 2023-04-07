'''
A separate script to run the benchmark models to compare with the CLS results

'''
from Benchmark_GenerativeModel import ModelLWF,ModelEWC,ModelSI,ModelJoint,ModelNaive
from avalanche.training import EWC
from avalanche.training import LwF
from avalanche.training import SynapticIntelligence
from avalanche.training import Naive
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
from utils import utility_funcs,JointTraining

def main():

    benchResultArrayEWC=[]
    benchResultArrayLWF=[]
    benchResultArraySI = []
    benchResultArrayNaive = []
    benchResultArrayJoint = []

    num_runs=3
    n_classes=10
    epochs = 45
    joint_epochs = 100
    n_experiences=5
    device = "cuda"
    patienceJoint = 10

    train_batch_size = 64
    eval_batch_size = 64
    joint_batchSize = 64

    ewc_lambda = 3100
    temperature = 26
    alpha = 9

    learning_rate = 1e-3
    learning_ratejoint = 1e-2
    ##Hyperparameters Generator 
 
    learning_rateGR = 1e-4
    batch_sizeGR = 32 #128
    num_epochsGR = 50

    patienceGR = 35

    synthetic_imgHeight = 32
    synthetic_imgWidth = 32
    img_channel_dim = 3
    latent_embedding = 100

    # buffer size = num_syntheticExamplesPerDigit * 10
    num_syntheticExamplesPerDigit = 500
    num_originalExamplesPerDigit = 500

    #  model parameters
    input_channel = 3
    
    
    #Buffer transformations
    train_transformBuffer = Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2615))])

    #Train data transformations
    train_transformInput = Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2615))])

    #Joint Train data transformations
    train_transformJoint = Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2615))])

    test_transformInput = Compose([ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2615))])
    
    #Tranformations for the GR model
    train_transformGR = Compose([ToTensor()])

    cifar_trainJoint = torchvision.datasets.CIFAR10('./data/',train=True,download=True,transform=train_transformJoint) #only for joint training
    cifar_train = torchvision.datasets.CIFAR10('./data/',train=True,download=True,transform=train_transformGR)
    cifar_test = torchvision.datasets.CIFAR10('./data/',train=False,download=True,transform=test_transformInput)

    scenario_trainTest = nc_benchmark(cifar_train, cifar_test, n_experiences=n_experiences, shuffle=True, seed=9, task_labels=False)

    train_stream = scenario_trainTest.train_stream
    test_stream =  scenario_trainTest.test_stream

    interactive_logger = InteractiveLogger()
    modellwf = ModelLWF(output_size=n_classes,input_channel=input_channel)
    modelewc = ModelEWC(output_size=n_classes,input_channel=input_channel)
    modelsi = ModelSI(output_size=n_classes,input_channel=input_channel)
    modelJoint = ModelJoint(output_size=n_classes,input_channel=input_channel)
    modelNaive = ModelNaive(output_size=n_classes,input_channel=input_channel)

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

        LwfModel = LwF(model=modellwf,optimizer=SGD(modellwf.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size
        ,eval_mb_size=eval_batch_size, criterion=CrossEntropyLoss(),temperature=temperature, alpha=alpha, train_epochs=epochs,device=device) # EWC replay model

        ewcModel = EWC(model=modelewc,optimizer=SGD(modelewc.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size
        ,eval_mb_size=eval_batch_size, ewc_lambda=ewc_lambda, criterion=CrossEntropyLoss(),train_epochs=epochs,device=device)
        #lambda=0.001

        siModel = SynapticIntelligence(model=modelsi,optimizer=SGD(modelewc.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size,
        eval_mb_size=eval_batch_size, si_lambda=ewc_lambda, criterion=CrossEntropyLoss(),train_epochs=epochs,device=device)

        naiveModel = Naive(model=modelNaive,optimizer=SGD(modelNaive.parameters(), lr=learning_rate, momentum=0.9),train_mb_size=train_batch_size,
        eval_mb_size=eval_batch_size,criterion=CrossEntropyLoss(),train_epochs=epochs,device=device)

        jointModel = JointTraining(model=modelJoint, epochs=joint_epochs,learning_rate=learning_ratejoint, batch_size=joint_batchSize,num_class=n_classes,
                                   patience=patienceJoint,device=device) 

        ## Generator model
        modelGR = VAE(channel_dim=img_channel_dim, latent_embedding = latent_embedding,img_width=synthetic_imgWidth)
        Vae_Cls_Obj = Vae_Cls_Generator(num_epochs=num_epochsGR,model=modelGR,learning_rate=learning_rateGR,
                                batch_size=batch_sizeGR,device=device,patience=patienceGR,img_channel_dim=img_channel_dim)
        
        # Joint Training
        print(" Training the Joint Model ")
        jointModel.train(cifar_trainJoint)
        print("Training completetd")

        ## Training and Evaluation
        print('Starting experiment...')
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
                buffer_images.append(temp_img)
                buffer_labels.append(temp_labels)
           
            # Naive model training
            print("Training the Naive Model")
            naiveModel.train(experience)
            print('Training completed')

            # Finally training the Generator on the complete psuedo buffer images
            if (exp_numb == n_experiences-1):
                print("Training the Generator on the complete psuedo buffer images")
               # Vae_Cls_Obj.train(experience,synthetic_imgHeight,synthetic_imgWidth,buff_images=buffer_images)

            ## Benchmark and buffer data generation 
            print("generating benchmark dataset")
            exp_data = utility_funcs.benchmarkDataPrep(experience=experience,device=device,train_transformBuffer=train_transformBuffer,buffer_data=buffer_images,
            buffer_label=buffer_labels,synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,
            train_transformInput=train_transformInput,img_channelDim=img_channel_dim)
        
            # for temp_data in DataLoader(exp_data,batch_size=len(exp_data)):
            scenario_exp = nc_benchmark(exp_data,exp_data,n_experiences=1,task_labels=False)
            data_expTrain = scenario_exp.train_stream
            print("----------------------------------Done----------------------------------")
            for new_exp in data_expTrain:
                print("CLasses in the modified experience ",new_exp.classes_in_this_experience)
                print("Training EWC Model")
                ewcModel.train(new_exp)  #Avalanche Benchmark strategy
                print('Training completed')

                print("Training LWF Model")
                LwfModel.train(new_exp)
                print('Training completed')

                print("Training SI Model")
                siModel.train(new_exp)
                print('Training completed')

            # Evaluation
            print('Computing accuracy on the whole test set')
            ## eval also returns a dictionary which contains all the metric values
            print("computing accuracy for EWC Model")
            bench_resultsEWC = ewcModel.eval(test_stream)
            print("*"*20)
            print("computing accuracy for LWF Model")
            bench_resultsLWF = LwfModel.eval(test_stream)
            print("*"*20)
            print("computing accuracy for SI Model")
            bench_resultsSI = siModel.eval(test_stream)
            print("*"*20)

            print("computing accuracy for Naive Model")
            bench_resultsNaive = naiveModel.eval(test_stream)
            print("*"*20)

            print("computing accuracy for Joint Model")
            bench_resultsJoint = jointModel.eval(test_stream)
            print("*"*20)

        #Saving the result for plots
        benchResultArrayEWC.append(dataPrepToPlot(bench_resultsEWC,len(train_stream)))
        benchResultArrayLWF.append(dataPrepToPlot(bench_resultsLWF,len(train_stream)))
        benchResultArraySI.append(dataPrepToPlot(bench_resultsSI,len(train_stream)))
        benchResultArrayNaive.append(dataPrepToPlot(bench_resultsNaive,len(train_stream)))
        # benchResultArrayJoint.append(dataPrepToPlot(bench_resultsJoint,len(train_stream)))
        benchResultArrayJoint.append(bench_resultsJoint)

    meanBenchEWC = np.round(np.sum(benchResultArrayEWC,axis=0)/num_runs, decimals=2)
    meanBenchLWF = np.round(np.sum(benchResultArrayLWF,axis=0)/num_runs, decimals=2)
    meanBenchSI = np.round(np.sum(benchResultArraySI,axis=0)/num_runs, decimals=2)
    meanBenchNaive = np.round(np.sum(benchResultArrayNaive,axis=0)/num_runs, decimals=2)
    meanBenchJoint = np.round(np.sum(benchResultArrayJoint,axis=0)/num_runs, decimals=2)

    stdBenchEWC = np.round(np.std(benchResultArrayEWC,axis=0), decimals=8)    #Mean after n runs runs
    stdBenchLWF = np.round(np.std(benchResultArrayLWF,axis=0), decimals=8)
    stdBenchSI = np.round(np.std(benchResultArraySI,axis=0), decimals=8)
    stdBenchNaive = np.round(np.std(benchResultArrayNaive,axis=0), decimals=8)
    stdBenchJoint = np.round(np.std(benchResultArrayJoint,axis=0), decimals=8)

    barPlotMeanPred(meanBenchEWC = meanBenchEWC,meanBenchLWF = meanBenchLWF, meanBenchSI = meanBenchSI, meanBenchNaive = meanBenchNaive, meanBenchJoint=meanBenchJoint, 
                    stdBenchEWC = stdBenchEWC, stdBenchLWF = stdBenchLWF, stdBenchSI = stdBenchSI, stdBenchNaive = stdBenchNaive, stdBenchJoint = stdBenchJoint, 
                    n_experinces=n_experiences)

    print(f"The mean value after 5 experinces for {num_runs} for EWC model is {np.sum(meanBenchEWC)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for EWC model model is {np.sum(stdBenchEWC)/n_experiences}")

    print(f"The mean value after 5 experinces for {num_runs} for LWF model is {np.sum(meanBenchLWF)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for LWF model is {np.sum(stdBenchLWF)/n_experiences}")
    
    print(f"The mean value after 5 experinces for {num_runs} for SI model is {np.sum(meanBenchSI)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for SI model is {np.sum(stdBenchSI)/n_experiences}")

    print(f"The mean value after 5 experinces for {num_runs} for Naive model is {np.sum(meanBenchNaive)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for Naive model is {np.sum(stdBenchNaive)/n_experiences}")

    print(f"The mean value after 5 experinces for {num_runs} for Joint model is {np.sum(meanBenchJoint)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for Joint model is {np.sum(stdBenchJoint)/n_experiences}")

def dataPrepToPlot(bench_results,testDataExpLen):
    benchResultArray=[]
    for i in range (0,testDataExpLen):
        benchResultArray.append(bench_results["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00"+str(i)])
    return np.round(benchResultArray,decimals=2)


def barPlotMeanPred(meanBenchEWC,meanBenchLWF,meanBenchSI,meanBenchNaive,meanBenchJoint,stdBenchEWC,stdBenchLWF,stdBenchSI,stdBenchNaive,stdBenchJoint,n_experinces):
    N = n_experinces + 1
    ind = np.arange(N)
    width = 0.15
    fig, ax = plt.subplots()

    ymax = 0
    max_calc = [meanBenchEWC,meanBenchLWF,meanBenchSI]
    for i in range(3):
        temp = np.max(max_calc[i])
        if temp>ymax:
            ymax = temp

    EWC_avgOutputMean = np.round(np.sum(meanBenchEWC)/n_experinces,decimals=2)
    EWC_avgOutputStd = np.round(np.sum(stdBenchEWC)/n_experinces,decimals=2)

    LWF_avgOutputMean = np.round(np.sum(meanBenchLWF)/n_experinces,decimals=2)
    LWF_avgOutputStd = np.round(np.sum(stdBenchLWF)/n_experinces,decimals=2)

    SI_avgOutputMean = np.round(np.sum(meanBenchSI)/n_experinces,decimals=2)
    SI_avgOutputStd = np.round(np.sum(stdBenchSI)/n_experinces,decimals=2)

    Naive_avgOutputMean = np.round(np.sum(meanBenchNaive)/n_experinces,decimals=2)
    Naive_avgOutputStd = np.round(np.sum(stdBenchNaive)/n_experinces,decimals=2)

    Joint_avgOutputMean = np.round(np.sum(meanBenchJoint)/n_experinces,decimals=2)
    Joint_avgOutputStd = np.round(np.sum(stdBenchJoint)/n_experinces,decimals=2)

    meanBenchEWC = np.insert(meanBenchEWC,obj=n_experinces,values=EWC_avgOutputMean)
    stdBenchEWC = np.insert(stdBenchEWC,obj=n_experinces,values=EWC_avgOutputStd)

    meanBenchLWF = np.insert(meanBenchLWF,obj=n_experinces,values=LWF_avgOutputMean)
    stdBenchLWF = np.insert(stdBenchLWF,obj=n_experinces,values=LWF_avgOutputStd)

    meanBenchSI = np.insert(meanBenchSI,obj=n_experinces,values=SI_avgOutputMean)
    stdBenchSI = np.insert(stdBenchSI,obj=n_experinces,values=SI_avgOutputStd)

    meanBenchNaive = np.insert(meanBenchNaive,obj=n_experinces,values=Naive_avgOutputMean)
    stdBenchNaive = np.insert(stdBenchNaive,obj=n_experinces,values=Naive_avgOutputStd)

    meanBenchJoint = np.insert(meanBenchJoint,obj=n_experinces,values=Joint_avgOutputMean)
    stdBenchJoint = np.insert(stdBenchJoint,obj=n_experinces,values=Joint_avgOutputStd)
 
    bar_ewc = ax.bar(ind, meanBenchEWC, width, color = 'r',label="EWC Model",yerr=stdBenchEWC)
    bar_lwf = ax.bar(ind+width, meanBenchLWF, width, color='g',label="LWF Model",yerr=stdBenchLWF)
    bar_si = ax.bar(ind+2*width, meanBenchSI, width, color='b',label="SI Model",yerr=stdBenchSI)

    bar_naive = ax.bar(ind+3*width, meanBenchNaive, width, color='cyan',label="Naive Model",yerr=stdBenchNaive)
    bar_joint = ax.bar(ind+4*width, meanBenchJoint, width, color='gold',label="Joint Model",yerr=stdBenchJoint)

    ax.axvline(x=4.8,ymin=0,ymax=ymax,color='black', linestyle='dotted', linewidth=2.5)
    
    ax.bar_label(bar_ewc, padding=3)
    ax.bar_label(bar_lwf, padding=3)
    ax.bar_label(bar_si, padding=3)
    ax.bar_label(bar_naive, padding=3)
    ax.bar_label(bar_joint, padding=3)
    
    ax.set_title("CIFAR10 Benchmark")
    ax.set_xlabel("Experiences & Models")
    ax.set_ylabel("Accuarcy")
    ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5","Avg Output"])
    ax.legend((bar_ewc, bar_lwf,bar_si, bar_naive, bar_joint), ('EWC Model', 'LWF Model','Synaptic Intelligence','Naive','Joint'),loc=0)
    fig.tight_layout()
    plt.show()
    plt.savefig("CIFAR/buffer_size5000benchmark_after3runs.png")

if __name__=="__main__":
    main()