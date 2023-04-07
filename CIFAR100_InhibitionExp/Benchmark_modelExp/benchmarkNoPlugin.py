from avalanche.benchmarks.classic import SplitMNIST,SplitFMNIST
from EWC_replay import EWC_replay
from Lwf_replay import Lwf_replay
import numpy as np
import matplotlib.pyplot as plt
from avalanche.training import GEM


def main():
    '''
    A separate script uto run the benchmark models to compare with the CLS results, all the variables are set equal to the CLS,
    the experineces are also made same by keeping the class order fixed and the number of runs and epochs to be same as the 
    CLS model.
    '''

    benchPred5=[]

    num_runs=5
    num_epochs=10
    buffer_size=200
    n_classes=10

    n_experiences=5

    benchmark = SplitMNIST(n_experiences=n_experiences, seed=1,fixed_class_order=[0,8,4,7,6,3,1,5,9,2])  # MNIST benchmark

    #benchmark = SplitFMNIST(n_experiences=5, seed=None,fixed_class_order=[0,8,4,7,6,3,1,5,9,2])  # Fashion MNIST

    for counter in range(num_runs):
        print("*"*10)
        print(f" Starting Repeatation Number {counter} out of 5")
        print("*"*10)
    
        LwfReplayModel = Lwf_replay(n_epochs=num_epochs,buffer_size=buffer_size,n_classes=n_classes,input_channel=1).Lwf_replay_benchmark() # EWC replay model
        #ewcReplayModel = EWC_replay(n_epochs=num_epochs,buffer_size=buffer_size,n_classes=n_classes,input_channel=1).ewc_replay_benchmark()
        gem_model = GEM()

        ## Training and Evaluation for Custom Method
        print('Starting experiment...')
        for experience in benchmark.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            
            #ewcReplayModel.train(experience)  #Avalanche Benchmark strategy

            LwfReplayModel.train(experience)
            print('Training completed')

            print('Computing accuracy on the whole test set')
            ## eval also returns a dictionary which contains all the metric values
            bench_results = LwfReplayModel.eval(benchmark.test_stream)

            #bench_results = LwfReplayModel.eval(benchmark.test_stream)

        #Saving the result for plots
        benchResultArray = dataPrepToPlot(bench_results,len(benchmark.train_stream))
        benchPred5.append(benchResultArray)
    meanBenchPred = np.sum(benchPred5,axis=0)/num_runs
    barPlotMeanPred(meanBenchPred)

    print(f"The mean value after 5 experinces for {num_runs} for benchmark model is {np.sum(meanBenchPred)/n_experiences}")
    print(f"The Corresponding std. after 5 experinces for {num_runs} for benchmark model is {np.std(meanBenchPred)/n_experiences}")
    

def dataPrepToPlot(bench_results,testDataExpLen):
    benchResultArray=[]
    for i in range (0,testDataExpLen):
        benchResultArray.append(bench_results["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00"+str(i)])
    return np.round(benchResultArray,decimals=2)


def barPlotMeanPred(benchResultArray):
    N = 5
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots()
    bar_benchmark = ax.bar(ind, benchResultArray, width, color = 'b')
    
    ax.bar_label(bar_benchmark, padding=3)
    
    ax.set_title("Buffer size300,SMF0.80,Epoch10")
    ax.set_xlabel("Experiences & Models")
    ax.set_ylabel("accuarcy")
    ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5"])
    ax.legend(['Benchmark Model (LWF Replay)'],loc=4 )
    fig.tight_layout()
    plt.show()
    plt.savefig("pics/FMNIST/benchmarkLWF_After5Runstest2.png")

if __name__=="__main__":
    main()
