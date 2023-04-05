# import numpy as np
# import torch
# import torch.nn as nn

# class InhibitionLayer(nn.Module):
#     def __init__(self, input_size, to_do=False,rho=1):
#         super().__init__()
#         self.input_size= input_size
#         self.rho = rho
#         self.to_do = to_do
#         inhibition_weight = torch.Tensor(input_size)
#         self.inhibition_weight = nn.Parameter(inhibition_weight)  # to make the weights of the layer trainable
#         self.sigma = torch.rand((1,input_size)) # size of the tesnor
#         # initialize inihibtion weights
#         torch.nn.init.normal_(self.inhibition_weight, mean=0.0, std=1)
       
#     def forward(self, x):
#         '''
#         #dim=which index to apply softmax function, must be 1 or the columns of the received tensor
#         '''
#         softmax = nn.Softmax(dim=1)  #Instead of sigmoid    
#         if self.to_do:
#             self.sigma = softmax(self.rho * x * self.inhibition_weight)
#             x_temp = torch.where(self.sigma > 0.5, x * self.sigma + x, x)
#             return x_temp
#         else:
#             return x
     
