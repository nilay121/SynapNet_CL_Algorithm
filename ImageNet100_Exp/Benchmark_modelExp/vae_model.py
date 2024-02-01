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
    def __init__(self,channel_dim,latent_embedding,img_width):
        super().__init__()
        self.latent_embedding = latent_embedding
        self.channel_dim = channel_dim
        self.img_width = img_width
        self.relu = nn.LeakyReLU(0.01,inplace=False)
        
    # Encoder Layers
        self.conv1 = nn.Conv2d(self.channel_dim, 16, stride=1, kernel_size=(3, 3), padding=1)
        self.batch1 =  nn.BatchNorm2d(num_features=16)
        self.conv2 =  nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.conv3 =  nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=1)
        self.batch3 =  nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(64, 128, stride=2, kernel_size=(3, 3), padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=128)
        self.conv5 =  nn.Conv2d(128, 254, stride=2, kernel_size=(3, 3), padding=1)
        self.batch5 =  nn.BatchNorm2d(num_features=254)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1016, 2000)
        self.linear2 = nn.Linear(2000, 2000)

    # Latent Embedding    
        self.z_mean = torch.nn.Linear(2000, self.latent_embedding)
        self.z_log_var = torch.nn.Linear(2000, self.latent_embedding)
    # Decoder layers    
        self.LE_to_Linear = torch.nn.Linear(self.latent_embedding, 2000)
        self.Dlinear2 = nn.Linear(2000,2000)
        self.Dlinear1 = nn.Linear(2000,1016)
        self.Reshapeoutput = Reshape(-1, 254, 2, 2)
        self.deconv5 = nn.ConvTranspose2d(254, 128, stride=2, kernel_size=(4, 4), padding=1)
        self.Dbatch5 =  nn.BatchNorm2d(num_features=128)

        self.DbatchSC1 =  nn.BatchNorm2d(num_features=254) # skip connection bn
        #self.DbatchSC2 =  nn.BatchNorm2d(num_features=64) # skip connection bn

        self.deconv4 = nn.ConvTranspose2d(128, 64, stride=2, kernel_size=(4, 4), padding=1)    
        self.Dbatch4 = nn.BatchNorm2d(num_features=64)          
        self.deconv3 = nn.ConvTranspose2d(64, 32, stride=2, kernel_size=(4, 4), padding=1)
        self.Dbatch3 = nn.BatchNorm2d(num_features=32)         
        self.deconv2 = nn.ConvTranspose2d(32, 16, stride=2, kernel_size=(4, 4), padding=1) 
        self.Dbatch2 = nn.BatchNorm2d(num_features=16)              
        self.deconv1 = nn.ConvTranspose2d(16, self.channel_dim, stride=(1, 1), kernel_size=(3, 3), padding=0)
        self.trimming = Trim(self.img_width)  
        self.normalize = nn.Sigmoid()

    def decoder(self,x,conv3_outSkip):
        # Decoder 
        LE_to_Linear = self.relu(self.LE_to_Linear(x))
        linear2_out = self.relu(self.Dlinear2(LE_to_Linear))
        linear1_out = self.relu(self.Dlinear1(linear2_out)) 
        reshaped_out = self.Reshapeoutput(linear1_out)
        
        deconv5_out = self.relu(self.Dbatch5(self.deconv5(reshaped_out)))
        deconv4_out = self.deconv4(deconv5_out)

        deconv4_out = torch.add(conv3_outSkip,deconv4_out) # Skip Connection
        deconv4_out = self.relu(self.Dbatch4(deconv4_out))

        deconv3_out = self.relu(self.Dbatch3(self.deconv3(deconv4_out)))
        deconv2_out = self.relu(self.Dbatch2(self.deconv2(deconv3_out)))
        
        deconv1_out = self.deconv1(deconv2_out)
        trim_out = self.trimming(deconv1_out)
        decoded = self.normalize(trim_out)
        return decoded

    def encoding_fn(self, x):
        x = x.reshape(1,self.channel_dim,self.img_width,self.img_width)
        #x = self.encoder(x)
        # Encoder
        conv1_out = self.relu(self.batch1(self.conv1(x)))
        conv2_out = self.relu(self.batch2(self.conv2(conv1_out)))
        conv3_out = self.relu(self.batch3(self.conv3(conv2_out)))

        conv3_outSkip = self.conv3(conv2_out)

        conv4_out = self.relu(self.batch4(self.conv4(conv3_out)))
        conv5_out = self.relu(self.batch5(self.conv5(conv4_out)))
        faltten = self.flatten(conv5_out)
        linear1_out = self.relu(self.linear1(faltten))
        linear2_out = self.relu(self.linear2(linear1_out))

        z_mean, z_log_var = self.z_mean(linear2_out), self.z_log_var(linear2_out)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var,conv3_outSkip
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        # Encoder
        conv1_out = self.relu(self.batch1(self.conv1(x)))
        conv2_out = self.relu(self.batch2(self.conv2(conv1_out)))
        conv3_out = self.relu(self.batch3(self.conv3(conv2_out)))

        conv3_outSkip = self.conv3(conv2_out)

        conv4_out = self.relu(self.batch4(self.conv4(conv3_out)))
        #conv5_out = self.relu(self.batch5(self.conv5(conv4_out)))
        conv5_out = (self.batch5(self.conv5(conv4_out)))
        faltten = self.flatten(conv5_out)
        linear1_out = self.relu(self.linear1(faltten))
        linear2_out = self.relu(self.linear2(linear1_out))
        # Latent space embedding
        z_mean, z_log_var = self.z_mean(linear2_out), self.z_log_var(linear2_out)
        # Reparameterize
        encoded = self.reparameterize(z_mean, z_log_var)
        # Decoder 
        LE_to_Linear = self.relu(self.LE_to_Linear(encoded))
        linear2_out = self.relu(self.Dlinear2(LE_to_Linear))
        linear1_out = self.relu(self.Dlinear1(linear2_out))         
        reshaped_out = self.Reshapeoutput(linear1_out)

        deconv5_out = self.relu(self.Dbatch5(self.deconv5(reshaped_out)))       
        deconv4_out = self.deconv4(deconv5_out)

        deconv4_out = torch.add(conv3_outSkip,deconv4_out) # Skip Connection
        deconv4_out = self.relu(self.Dbatch4(deconv4_out))

        deconv3_out = self.relu(self.Dbatch3(self.deconv3(deconv4_out)))
        deconv2_out = self.relu(self.Dbatch2(self.deconv2(deconv3_out)))
        
        deconv1_out = self.deconv1(deconv2_out)
        trim_out = self.trimming(deconv1_out)
        decoded = self.normalize(trim_out)
    
        return encoded, z_mean, z_log_var, decoded