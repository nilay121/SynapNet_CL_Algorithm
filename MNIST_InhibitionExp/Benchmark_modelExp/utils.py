import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.optim import Adam,SGD
from torchmetrics import Accuracy

class CustomDatasetForDataLoader(Dataset):
    def __init__(self,data,targets):
        # convet labels to 1 hot
        self.data = data
        self.targets = targets
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.targets[idx]

class utility_funcs:
    def buffer_dataGeneration(digit, experience,device,model,num_examples,numbOf_orgExamples):
        images = []
        synthetic_imgs = []
        labelsForSyntheticImages = []
        batch_size = 64
        idx = 0
        constraining_term = 0.15
        originalImage_example = 0

        data = experience.dataset
        dataset = DataLoader(data,batch_size=batch_size,num_workers=4,shuffle=True)

        for data in dataset:
            x = data[0].to(device)
            y = data[1].to(device)
            if y[idx] == digit:
                images.append(x[idx])
                originalImage_example += 1
            if originalImage_example == numbOf_orgExamples:
                break
            idx+=1
            
        # images = torch.Tensor(images).to(device=device)
        # print("the shape of the image is ",images.shape)

        encodings_digit = []
        for i in range(numbOf_orgExamples):
            with torch.no_grad():
                _, mu, sigma = model.encoding_fn(images[i]) #.view(1, 784)
            encodings_digit.append([mu.squeeze(0).squeeze(0).cpu().detach().numpy(),
                                sigma.squeeze(0).squeeze(0).cpu().detach().numpy()])
        
        encodings_digit = np.array(encodings_digit)
        
        # take average of the mean and sigma for N examples of the same digit
        mean_encodings_digit = encodings_digit.mean(axis=0)
        
        # make the dimension of mu and sigma as its original dimension
        mu = mean_encodings_digit[0]
        mu = torch.as_tensor(mu).unsqueeze(0).unsqueeze(1).cuda()

        sigma = mean_encodings_digit[1]
        sigma = torch.as_tensor(sigma).unsqueeze(0).unsqueeze(1).cuda()

        for example in range(num_examples):
            with torch.no_grad():
                epsilon = torch.randn_like(sigma)
                z = mu + constraining_term*sigma * epsilon
                out = model.decoder(z).cpu().detach().numpy()
            synthetic_imgs.append(out)
            labelsForSyntheticImages.append(digit)

        synthetic_imgs = np.array(synthetic_imgs)
        return synthetic_imgs, labelsForSyntheticImages

    def toPlotGRImages(images_tensor,image_height,image_width,img_channel,step_size):
        buffer_images = torch.as_tensor(np.array(images_tensor))
        buffer_images = buffer_images.squeeze(2).reshape(-1,img_channel,image_height,image_width)
        
        num_images = 20
        color_channels = buffer_images.shape[1]
        high_limit = 10 * step_size

        for i in range(0,high_limit,step_size):
            fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
            new_images = buffer_images[i:i+num_images]
            for ax, img in zip(axes, new_images):
                curr_img = img.detach().to(torch.device('cpu'))        

                if color_channels > 1:
                    curr_img = np.transpose(curr_img, (1, 2, 0))
                    ax.imshow(curr_img)
                else:
                    ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')
            fig.savefig(f"synBuffImages/test.png{i/step_size}.png")

    def get_dataBuffer(buffer_data, buffer_labels,img_channelDim, size,synthetic_imgHeight,synthetic_imgWidth,device,transform= None):
        '''
        Getting data from the Generator buffer in a mini batch
        '''

        buffer_inputs = torch.as_tensor(np.array(buffer_data))
        buffer_inputs = buffer_inputs.squeeze(2).reshape(-1,img_channelDim,synthetic_imgHeight,synthetic_imgWidth).to(device)
        buffer_label = torch.as_tensor(np.array(buffer_labels).reshape(-1)).to(device)

        choice = np.random.choice(buffer_inputs.shape[0],size=size, replace=False)

        if transform is None: transform = lambda x: x
        temp_images = torch.stack([transform(ee.cpu()) for ee in buffer_inputs[choice]]).to("cuda")
        temp_labels = buffer_label[choice]
        
        return temp_images, temp_labels

    def inputDataTransformation(input_data,transform=None):
        if transform is None: transform = lambda x: x
        transformed_images = torch.stack([transform(ee.cpu()) for ee in input_data]).to("cuda")

        return transformed_images

    def benchmarkDataPrep(experience, device, synthetic_imgHeight=28, synthetic_imgWidth=28, train_transformBuffer=None, train_transformInput=None,
    img_channelDim = 1,buffer_data=[],buffer_label=[]):
        if len(buffer_data) and len(buffer_label) > 0:
            train_dataset = experience.dataset
            total_dataLength = train_dataset.__len__()
            total_bufferLength = np.array(buffer_label).reshape(-1)

            buffer_inputs, buffer_labels = utility_funcs.get_dataBuffer(buffer_data=buffer_data,buffer_labels=buffer_label,
            size=len(total_bufferLength),synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,device=device,
            transform=train_transformBuffer,img_channelDim=img_channelDim)

            buffer_inputs = buffer_inputs.cpu().detach().numpy()
            buffer_labels = buffer_labels.cpu().detach().numpy()

            train_data_loader = DataLoader(train_dataset, num_workers=4, batch_size=total_dataLength,shuffle=True) 
            for data in  train_data_loader:
                input_dataBT = data[0]
                input_data = utility_funcs.inputDataTransformation(input_data=input_dataBT,transform=train_transformInput)
                input_data = input_data.cpu().detach().numpy()
                input_labels = data[1].cpu().detach().numpy()
        
            concatenated_inputData = np.concatenate((input_data,buffer_inputs),axis=0)
            concatenated_inputLabels = np.concatenate((input_labels,buffer_labels),axis=0)

            concatenated_inputData = torch.as_tensor(concatenated_inputData)
            concatenated_inputLabels = torch.as_tensor(concatenated_inputLabels)

            newExpDataset = CustomDatasetForDataLoader(data=concatenated_inputData,targets=concatenated_inputLabels)
            return newExpDataset
        

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
        max_calc = [meanBenchEWC,meanBenchLWF,meanBenchSI,meanBenchJoint]
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
        
        ax.set_title("MNIST Benchmark")
        ax.set_xlabel("Experiences & Models")
        ax.set_ylabel("accuarcy")
        ax.set_xticks(ind+width,["exp1","exp2","exp3","exp4","exp5","Avg Output"])
        ax.legend((bar_ewc, bar_lwf,bar_si, bar_naive, bar_joint), ('EWC Model', 'LWF Model','Synaptic Intelligence','Naive','Joint'),loc=0)
        fig.tight_layout()
        plt.show()
        plt.savefig("MNIST/buffer_size5000benchmark_after5runstest3_trans.png")


class JointTraining:
    def __init__(self, model, epochs, learning_rate, batch_size, num_class, patience, device):
        self.model = model.to(device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_classes = num_class
        self.patience = patience
        self.loss = nn.CrossEntropyLoss()
        self.device  = device
        self.optimizer = SGD(self.model.parameters(),lr=self.learning_rate)
        #self.optimizer = Adam(self.model.parameters(),lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, verbose=True)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
    def train(self,train_data):
        train_data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,shuffle=True)
        for epochs in range(self.epochs):
            loader_loop = tqdm(train_data_loader,leave=False)  
            for data, input_labels in loader_loop:
                data = data.to(self.device)
                input_labels = input_labels.to(self.device)
                pred_train = self.model(data)
                loss = self.loss(pred_train,input_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loader_loop.set_description(f"Epoch {epochs}/{self.epochs}")
                loader_loop.set_postfix(loss = loss.item())
            self.scheduler.step(loss)

    def eval(self,test_stream):
        exp_counter=0
        acc_exp=[]
        # acc_dict={}
        self.model.eval()
        with torch.no_grad():
            for experiences in test_stream:
                eval_dataset = experiences.dataset
                eval_data_loader = DataLoader(eval_dataset,num_workers=4,batch_size=self.batch_size,shuffle=False)
                total_accuracy=0
                batch_counter=0
                for data in tqdm(eval_data_loader):
                    input_dataBT =  data[0].to(self.device)        
                    input_label = data[1].to(self.device)
                    pred_test = self.model(input_dataBT)
                    pred_label = torch.argmax(pred_test.data,1)
                    total_accuracy+=self.accuracy(pred_label,input_label)
                    batch_counter+=1

                print(f"exp {exp_counter} model accuracy : {total_accuracy/batch_counter}")
                acc_exp.append((total_accuracy/batch_counter).item())
                exp_counter+=1
        return acc_exp



