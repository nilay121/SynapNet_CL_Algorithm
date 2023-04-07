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
        batch_size = 512
        idx = 0
        constraining_term = 0.1
        originalImage_example = 0

        data = experience.dataset
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

