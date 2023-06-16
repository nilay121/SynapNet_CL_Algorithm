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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

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
        constraining_term = 0.1
        originalImage_example = 0

        data = experience
        dataset = DataLoader(data,batch_size=batch_size,num_workers=4,shuffle=True)

        for data in dataset:
            x = data[0].to(device)
            y = data[1].cpu().detach().numpy()
            indices_img, = np.where(y==digit)
            for i in indices_img:
                images.append(x[i])
                originalImage_example += 1
                if (originalImage_example == numbOf_orgExamples):
                    break

        encodings_digit = []
        model.eval()
        for i in range(numbOf_orgExamples):
            with torch.no_grad():
                _,mu, sigma = model.encoding_fn(images[i]) #.view(1, 784)
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

    def toPlotGRImages(images_tensor,image_height,image_width,step_size):

        buffer_images = torch.as_tensor(np.array(images_tensor))
        buffer_images = buffer_images.squeeze(2).reshape(-1,1,image_height,image_width)

        print(f"The size of the buffer is {buffer_images.shape[0]}")

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

    def get_dataBuffer(buffer_data, buffer_labels, size,synthetic_imgHeight,synthetic_imgWidth,device,transform= None):
        '''
        Getting data from the Generator buffer in a mini batch
        '''

        buffer_inputs = torch.as_tensor(np.array(buffer_data))
        buffer_inputs = buffer_inputs.squeeze(2).reshape(-1,1,synthetic_imgHeight,synthetic_imgWidth).to(device)
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

    def benchmarkDataPrep(experience, device, synthetic_imgHeight=28, synthetic_imgWidth=28, train_transformBuffer=None, train_transformInput=None, buffer_data=[],buffer_label=[]):
        if len(buffer_data) and len(buffer_label) > 0:
            train_dataset = experience.dataset
            total_dataLength = train_dataset.__len__()
            total_bufferLength = np.array(buffer_label).reshape(-1)

            buffer_inputs, buffer_labels = utility_funcs.get_dataBuffer(buffer_data=buffer_data,buffer_labels=buffer_label,
            size=len(total_bufferLength),synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,device=device,
            transform=train_transformBuffer)

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

    def dataPrepToPlot(y_stable, y_plastic):
        cls_output = []

        '''
        The accuracy of the plastic model for the recent experiences are better than the stable model,
        whereas the accuracy of the stable model on the old experiences are better.
        '''
        for outputs in range(len(y_stable)):

            if (y_plastic[outputs] > y_stable[outputs]):
                cls_output.append(y_plastic[outputs])
            else:
                cls_output.append(y_stable[outputs])

        y_stable = np.array(y_stable)
        y_plastic = np.array(y_plastic)
        cls_output = np.array(cls_output)
        return np.round(y_stable,decimals=6),np.round(y_plastic,decimals=6),np.round(cls_output,decimals=6)

    def barPlotMeanPredBacku(y_plotPlastic,y_plotStable,y_clsOutput,n_experinces):
        N = n_experinces + 1
        ind = np.arange(N)
        width = 0.25
        fig, ax = plt.subplots()

        cls_avgOutputMean = np.round(np.sum(y_clsOutput)/n_experinces,decimals=2)

        y_plotPlastic = np.insert(y_plotPlastic,obj=n_experinces,values=0)

        y_plotStable = np.insert(y_plotStable,obj=n_experinces,values=cls_avgOutputMean)
        
        bar_plastic = ax.bar(ind, y_plotPlastic, width, color = 'r',label="Plastic Model")
        bar_stable = ax.bar(ind+width, y_plotStable, width, color='g',label="Stable Model")

        ax.axvline(x=2.6,ymin=0,ymax=np.max(y_plotPlastic),color='black', linestyle='dotted', linewidth=2.5)
        
        ax.bar_label(bar_plastic, padding=3)
        ax.bar_label(bar_stable, padding=3)
        
        ax.set_title("MNIST Unbalanced Experiences")
        ax.set_xlabel("Experiences & Models")
        ax.set_ylabel("Accuarcy")
        ax.set_xticks(ind+width,["exp1","exp2","exp3","Avg Output"])
        ax.legend((bar_plastic, bar_stable), ('Plastic Model', 'Stable Model'),loc=0)
        fig.tight_layout()
        plt.show()
        plt.savefig("pics/MNIST_temp.png")

    def barPlotMeanPred(y_plotPlastic,y_plotStable,y_clsOutput,stdStablePred,stdPlasticPred,stdClsOutput,n_experinces=3):
        N = n_experinces + 1
        ind = np.arange(N)
        width = 0.25
        fig, ax = plt.subplots()

        cls_avgOutputMean = np.round(np.sum(y_clsOutput)/n_experinces,decimals=2)
        cls_avgOutputstd = np.round(np.sum(stdClsOutput)/n_experinces,decimals=2)

        y_plotPlastic = np.insert(y_plotPlastic,obj=n_experinces,values=0)
        stdPlasticPred = np.insert(stdPlasticPred,obj=n_experinces,values=0)

        y_plotStable = np.insert(y_plotStable,obj=n_experinces,values=cls_avgOutputMean)
        stdStablePred = np.insert(stdStablePred,obj=n_experinces,values=cls_avgOutputstd)
        
        bar_plastic = ax.bar(ind, y_plotPlastic, width, color = 'lightgray',label="Plastic Model",yerr=stdPlasticPred)
        bar_stable = ax.bar(ind+width, y_plotStable, width, color='lightsteelblue',label="Stable Model",yerr=stdStablePred)

        ax.axvline(x=2.6,ymin=0,ymax=np.max(y_plotPlastic),color='black', linestyle='dotted', linewidth=2.5)
        
        ax.bar_label(bar_plastic, padding=3)
        ax.bar_label(bar_stable, padding=3)
        
        ax.set_title("MNIST Unbalanced Experiences")
        ax.set_xlabel("Experiences & Models")
        ax.set_ylabel("Accuarcy")
        ax.set_xticks(ind+width,["exp1","exp2","exp3","Avg Output"])
        ax.legend((bar_plastic, bar_stable), ('Plastic Model', 'Stable Model'),loc=0)
        fig.tight_layout()
        plt.show()
        plt.savefig("pics/MNIST_temp.png")


    def ConfusionMatrixPerExp(predictionsForCF_stable, predictionsForCF_plastic, ground_truth, labels, exp_numb, n_experiences):

        # Extracting the ground truth from the experiences
        org_class = []
        exp=0
        last_expLength = 0
        for experiences in ground_truth:
            eval_dataset = experiences.dataset
            total_dataLength = eval_dataset.__len__()
            eval_data_loader = DataLoader(eval_dataset,num_workers=4,batch_size=total_dataLength,shuffle=False)
            for data in  eval_data_loader:
                eval_dataLabels = data[1]
            org_class.append(eval_dataLabels.detach().numpy())
            exp+=1
            if exp == (n_experiences):
                last_expLength = eval_dataset.__len__()

        ## Flatten the array 

        predictionsForCF_stable = np.array(predictionsForCF_stable,dtype="object")
        itr_stable = predictionsForCF_stable.shape[0]
        predictionsForCF_stableFalttened = []
        for i in range(itr_stable):
            itr2 = len(predictionsForCF_stable[i])
            for j in range(itr2):
                predictionsForCF_stableFalttened.append(predictionsForCF_stable[i][j])

        predictionsForCF_plastic = np.array(predictionsForCF_plastic,dtype="object")
        itr_plastic = predictionsForCF_plastic.shape[0]
        predictionsForCF_plasticFalttened = []
        for i in range(itr_plastic):
            itr2 = len(predictionsForCF_plastic[i])
            for j in range(itr2):
                predictionsForCF_plasticFalttened.append(predictionsForCF_plastic[i][j])


        org_class = np.array(org_class,dtype="object")
        itr_ground = org_class.shape[0]
        org_classFlattened = []
        for i in range(itr_ground):
            itr2 = len(org_class[i])
            for j in range(itr2):
                org_classFlattened.append(org_class[i][j])

        predictionsForCF_stableFalttened = np.array(predictionsForCF_stableFalttened)
        predictionsForCF_plasticFalttened = np.array(predictionsForCF_plasticFalttened)
        org_classFlattened = np.array(org_classFlattened)
        
        ## Comaprision between stable model and plastic model
        if (exp_numb==n_experiences):
            temp_stablePred = predictionsForCF_stableFalttened[0:(len(predictionsForCF_stableFalttened)-last_expLength)]
            temp_plasticPred = predictionsForCF_plasticFalttened[(len(predictionsForCF_stableFalttened)-last_expLength):]
            cls_output = np.concatenate((temp_stablePred,temp_plasticPred))
        else:
            cls_output = predictionsForCF_stableFalttened
        
        cls_output = np.array(cls_output)
        
        cd_matrix = confusion_matrix(y_true = org_classFlattened, y_pred = cls_output, labels = labels)
        cf_plot = ConfusionMatrixDisplay(cd_matrix,display_labels=labels)
        cf_plot.plot(cmap="BuGn",include_values=False)
        plt.show()
        #plt.savefig(f"confusionMatrix/confusionMatrix{exp_numb}experienceBuffer500.png")