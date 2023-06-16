import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time

class Vae_Cls_Generator:
  def __init__(self,num_epochs, model, device,learning_rate,batch_size,patience,reconstruction_term_weight=1):
    self.num_epochs = num_epochs
    self.device = device
    self.model =  model.to(self.device) 
    self.patience = patience

    self.loss_fn = F.mse_loss

    self.reconstruction_term_weight = reconstruction_term_weight
    self.learning_rate = learning_rate
    self.batch_size = batch_size

  def train(self,cl_exp,synthetic_imgHeight=None,synthetic_imgWidth=None,buff_images=[]):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, verbose=True)

    train_dataset = cl_exp
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=self.batch_size,shuffle=True)
    for epoch in range(self.num_epochs):
        self.model.train()
        loader_loop = tqdm(train_loader,leave=False)
        for data,_ in loader_loop:
          features = data.reshape(-1,1,28,28).to(self.device) 
          #features = features.astype("float32")
          if len(buff_images) > 0:
            buffer_images = torch.as_tensor(np.array(buff_images))
            buffer_images = buffer_images.squeeze(2).reshape(-1,1,synthetic_imgHeight,synthetic_imgWidth).to(self.device)
            features = buffer_images

          encoded, z_mean, z_log_var, decoded = self.model(features)

          kl_div = -0.5 * torch.sum(1 + z_log_var 
                                    - z_mean**2 
                                    - torch.exp(z_log_var), 
                                    axis=1) 

          batchsize = kl_div.size(0)
          kl_div = kl_div.mean() 

          pixelwise = self.loss_fn(decoded, features, reduction='none')  
          pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) 
          pixelwise = pixelwise.mean() 

          loss = self.reconstruction_term_weight*pixelwise + kl_div
          
          optimizer.zero_grad()

          loss.backward()

          # UPDATE MODEL PARAMETERS
          optimizer.step()
          loader_loop.set_description(f"Epoch {epoch}/{self.num_epochs}")
          loader_loop.set_postfix(loss = loss.item())

        scheduler.step(loss)

  def evaluate(self):
    pass