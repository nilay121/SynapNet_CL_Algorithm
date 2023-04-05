import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn import BCELoss
from tqdm import tqdm

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, img_width, *args):
        super().__init__()
        self.img_width = img_width

    def forward(self, x):
        return x[:, :, :self.img_width, :self.img_width]


class VAE(nn.Module):
    def __init__(self,channel_dim,latent_embedding,batch_size,img_width):
        super().__init__()
        self.latent_embedding = latent_embedding
        self.channel_dim = channel_dim
        self.batch_size = batch_size
        self.img_width = img_width
        
        self.encoder = nn.Sequential(
                nn.Conv2d(self.channel_dim, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
        )    
        
        self.z_mean = torch.nn.Linear(3136, 50)
        self.z_log_var = torch.nn.Linear(3136, 50)
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(50, 3136),
                Reshape(-1, 64, 7, 7),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01,inplace=False),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),                
                nn.LeakyReLU(0.01,inplace=False),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),                
                nn.LeakyReLU(0.01,inplace=False),
                nn.ConvTranspose2d(32, self.channel_dim, stride=(1, 1), kernel_size=(3, 3), padding=0), 
                Trim(self.img_width),
                nn.Sigmoid()
                )

    def encoding_fn(self, x):
        x = self.encoder(x)
        x = x.view(1,1,-1) 
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
