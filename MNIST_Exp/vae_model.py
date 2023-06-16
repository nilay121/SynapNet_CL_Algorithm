import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn import BCELoss
from tqdm import tqdm

### MODEL

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, image_width,  *args):
        super().__init__()
        self.image_width = image_width

    def forward(self, x):
        return x[:, :, :self.image_width, :self.image_width]


class VAE(nn.Module):
    def __init__(self,channel_dim,latent_embedding,img_width):
        super().__init__()
        self.latent_embedding = latent_embedding
        self.channel_dim = channel_dim
        self.img_width = img_width
        
        self.encoder = nn.Sequential(
                nn.Conv2d(self.channel_dim, 16, stride=1, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(64, 128, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),

                nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(num_features=254),
                nn.Flatten(),
                nn.Linear(1016, 1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000, 500),
        )    
        
        self.z_mean = torch.nn.Linear(500, self.latent_embedding)
        self.z_log_var = torch.nn.Linear(500, self.latent_embedding)
        
        self.decoder = nn.Sequential(
                torch.nn.Linear(self.latent_embedding, 500),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(500,1000),
                nn.LeakyReLU(0.01,inplace=False),
                nn.Linear(1000,1016),
                nn.LeakyReLU(0.01,inplace=False),
                Reshape(-1, 254, 2, 2),

                nn.ConvTranspose2d(254, 128, stride=2, kernel_size=(4, 4), padding=1),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose2d(128, 64, stride=2, kernel_size=(4, 4), padding=1),    
                nn.BatchNorm2d(num_features=64),            
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose2d(64, 32, stride=2, kernel_size=(4, 4), padding=1),    
                nn.BatchNorm2d(num_features=32),            
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose2d(32, 16, stride=2, kernel_size=(4, 4), padding=1),  
                nn.BatchNorm2d(num_features=16),              
                nn.LeakyReLU(0.01,inplace=False),

                nn.ConvTranspose2d(16, self.channel_dim, stride=(1, 1), kernel_size=(3, 3), padding=0), 
                Trim(self.img_width),  
                nn.Sigmoid()
                )

    def encoding_fn(self, x):
        x = x.reshape(1,self.channel_dim,self.img_width,self.img_width)
    #   #  print(x.shape)
        x = self.encoder(x)
        #x = x.view(1,self.image_channel_dim,-1) 
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

