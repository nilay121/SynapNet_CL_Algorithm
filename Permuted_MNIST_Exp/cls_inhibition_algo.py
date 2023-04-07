'''
BioINet (Biological Inspired Network) is Biological Inspired Complementary Learning System implementation with a fast Learner (hippocampus), 
a slow learner (Neocortex), lateral Inhibition and a sleep phase for re-organizing the memories.

Note: "LateralInhibition" class has been taken from the Paper "Lateral Inhibition-Inspired Convolutional Neural Network 
for Visual Attention and Saliency Detection (2018)" with minor changes. The credit for this class goes to the respective authors.

'''
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam,SGD
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from torchmetrics import Accuracy
from tqdm import tqdm
from plasticModel import PlasticModel
from workingModel import WorkingModel 
from stableModel import StableModel
from utils import utility_funcs

class CustomInhibitStrategy():
  def __init__(self,working_model,modelstable,modelplastic,criterion_ce=torch.nn.CrossEntropyLoss(),
               stable_model_update_freq=0.75,plastic_model_update_freq = 1.0,
               stable_model_alpha = 0.999,plastic_model_alpha=0.999,
               num_epochs=10,reg_weight=0.9,inhibit_factor=0.03,batch_size=32,
               learning_rate=1e-3,rho=0.5,n_classes=10,to_do=False,n_channel=1,patience=3,mini_batchGR=32,train_transformBuffer=None,
               train_transformInput=None,val_transformInput=None,clipping=False):
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.n_classes=n_classes
    self.to_do = to_do
    self.n_channel = n_channel
    self.rho=rho
    self.patience = patience
    self.clipping = clipping

    self.working_model = working_model(self.n_classes,self.n_channel).to(self.device)
    self.modelstable = modelstable(self.n_classes,self.n_channel).to(self.device) 
    self.modelplastic = modelplastic(self.n_classes,self.n_channel).to(self.device)
    self.learning_rate=learning_rate

    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.mini_batchGR = mini_batchGR
    self.stable_model_update_freq = stable_model_update_freq  # stable model update frequency
    self.plastic_model_update_freq = plastic_model_update_freq  # plastic model update frequency
    self.stable_model_alpha = stable_model_alpha
    self.plastic_model_alpha=plastic_model_alpha
    self.criterion_ce = criterion_ce  

    self.consistency_loss = nn.MSELoss(reduction='none') 
    self.criterion_mse = nn.MSELoss()
    self.current_task = 0
    self.global_step = 0
    self.reg_weight=reg_weight
    self.inhibit_factor = inhibit_factor

    #buffer and Input transformation 
    self.train_transformBuffer = train_transformBuffer
    self.train_transformInput = train_transformInput
    self.val_transformInput = val_transformInput

  def train(self,experience,synthetic_imgHeight=28,synthetic_imgWidth=28,buf_inputs=[],buf_labels=[]):
    train_dataset = experience.dataset
    train_data_loader = DataLoader(train_dataset, num_workers=4, batch_size=self.batch_size,shuffle=True)

    optimizer = SGD(self.working_model.parameters(),lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True)

    for epoch in range(self.num_epochs):
      loader_loop = tqdm(train_data_loader,leave=False)
      for data in loader_loop:
        input_dataBT = data[0].to(self.device) #Input data before transformations

        # transformation on the input data
        input_data = utility_funcs.inputDataTransformation(input_data=input_dataBT,transform=self.train_transformInput).to(self.device)         
        input_label = data[1].to(self.device)
        
        optimizer.zero_grad()
        loss = 0 

        # transformations on the buffer data
        buffer_inputs, buffer_labels = utility_funcs.get_dataBuffer(buffer_data=buf_inputs,buffer_labels=buf_labels,size=self.mini_batchGR,
        synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,device=self.device,transform=self.train_transformBuffer)
        buffer_inputs = buffer_inputs.to(self.device)
        buffer_labels = buffer_labels.to(self.device)

         #Logits from the semantic memory
        stable_model_logits = self.modelstable(buffer_inputs) # check
        plastic_model_logits = self.modelplastic(buffer_inputs) # check
        stable_model_prob = F.softmax(stable_model_logits, 1)
        plastic_model_prob = F.softmax(plastic_model_logits, 1)

        label_mask = F.one_hot(buffer_labels, num_classes=stable_model_logits.shape[-1]) > 0
        sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
        sel_idx = sel_idx.unsqueeze(1)
        
        ema_logits = torch.where(sel_idx,stable_model_logits,plastic_model_logits,)
        l_cons = torch.mean(self.consistency_loss(self.working_model(buffer_inputs), ema_logits.detach()))
        l_reg = self.reg_weight * l_cons
        loss += l_reg
        input_data = torch.cat((input_data, buffer_inputs))
        input_label = torch.cat((input_label, buffer_labels))

        outputs = self.working_model(input_data)

        # ce loss and inhibition loss
        ce_loss = self.criterion_ce(outputs, input_label) 
        loss += ce_loss
        loss.backward()

        # Gradient Clipping
        if self.clipping:
          torch.nn.utils.clip_grad_norm_(self.working_model.parameters(), max_norm =5)

        optimizer.step()

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
          self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
          self.update_stable_model_variables()       
    
        loader_loop.set_description(f"Epoch {epoch}/{self.num_epochs}")
        loader_loop.set_postfix(loss = loss.item())

      scheduler.step(loss)
            
    return loss.item()

  def update_plastic_model_variables(self):
    alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
    for ema_param, param in zip(self.modelplastic.parameters(), self.working_model.parameters()):
      ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

  def update_stable_model_variables(self):
    alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
    for ema_param, param in zip(self.modelstable.parameters(), self.working_model.parameters()):
      ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

  ## Sleep Phase for re-organizing the memories

  def offline_reorganizing(self,buf_inputs,buf_labels,patience=5,synthetic_imgHeight=28,synthetic_imgWidth=28,epochs=10,lr_offline=1e-5,offline_batch=32):
    optimizer_offline = SGD(self.working_model.parameters(),lr=lr_offline)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_offline, patience=patience, verbose=True)
    batch_epochs = int(np.array(buf_labels).reshape(-1).shape[0]/self.mini_batchGR)

    for i in range(epochs):
      for j in range(batch_epochs):
        buffer_inputs, buffer_labels = utility_funcs.get_dataBuffer(buffer_data=buf_inputs,buffer_labels=buf_labels,size=offline_batch,
        synthetic_imgHeight=synthetic_imgHeight,synthetic_imgWidth=synthetic_imgWidth,device=self.device,transform=self.train_transformBuffer)
        buffer_inputs = buffer_inputs.to(self.device)
        buffer_labels = buffer_labels.to(self.device)

        # sleep only for stable model
        stable_model_logits = self.modelstable(buffer_inputs) 
        stable_model_prob = F.softmax(stable_model_logits, 1)
        loss_offline = self.criterion_ce(stable_model_prob,buffer_labels)

        optimizer_offline.zero_grad()
        loss_offline.backward()
        optimizer_offline.step()
      scheduler.step(loss_offline)
      print(f"Loss after {i} epoch for stbale model during sleep phase {loss_offline}")

  ## Evaluation phase  
  def evaluate(self,test_stream,validationFlag=False):
    accuracy = Accuracy(task="multiclass", num_classes=10).to(self.device)
    exp_counter=0
    acc_exp=[]
    predictionsForCF_stable = []
    predictionsForCF_plastic = []
    acc_dict={}
    self.modelstable.eval()
    self.modelplastic.eval() 
    self.working_model.eval()
    with torch.no_grad():
      for experiences in test_stream:
        eval_dataset = experiences.dataset
        eval_data_loader = DataLoader(eval_dataset,num_workers=4,batch_size=self.batch_size,shuffle=False)
        accuracy_expStable=0
        accuracy_expPlastic=0
        accuracy_expWorking=0
        batch_counter=0
        for data in tqdm(eval_data_loader):
          predictions_stable=[]
          predictions_working=[]
          predictions_plastic=[]

          input_dataBT =  data[0].to(self.device) 
          if validationFlag:
            input_data = utility_funcs.inputDataTransformation(input_data=input_dataBT,transform=self.val_transformInput).to(self.device)
          else:
            input_data = input_dataBT
            
          input_label = data[1].to(self.device)

          output_stable = self.modelstable(input_data)
          output_plastic = self.modelplastic(input_data)
          output_working = self.working_model(input_data)

          predictions_stable.append(torch.argmax(output_stable.data,1))
          predictions_plastic.append(torch.argmax(output_plastic.data,1))
          predictions_working.append(torch.argmax(output_working.data,1))

          accuracy_expStable+=accuracy(predictions_stable[0],input_label)
          accuracy_expPlastic+=accuracy(predictions_plastic[0],input_label)
          accuracy_expWorking+=accuracy(predictions_working[0],input_label)
          batch_counter+=1

          predictionsForCF_stable.append(predictions_stable[0].cpu().detach().numpy())
          predictionsForCF_plastic.append(predictions_plastic[0].cpu().detach().numpy())
        

        acc_exp.append(f"exp {exp_counter} Stable model accuracy : {accuracy_expStable/batch_counter},\
        Plastic model accuracy : {accuracy_expPlastic/batch_counter}, working model accuracy : {accuracy_expWorking/batch_counter}")
        
        acc_dict[str(exp_counter)]=[accuracy_expStable/batch_counter,accuracy_expPlastic/batch_counter]
        
        print(f"exp {exp_counter} Stable model accuracy : {accuracy_expStable/batch_counter},\
        Plastic model accuracy : {accuracy_expPlastic/batch_counter}, working model accuracy : {accuracy_expWorking/batch_counter}")
        exp_counter+=1
    return acc_exp,acc_dict,predictionsForCF_stable,predictionsForCF_plastic