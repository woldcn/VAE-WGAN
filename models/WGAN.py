# Author：woldcn
# Create Time：2022/10/25 8:42
# Description：

import torch
from torch import nn,optim,autograd
import numpy as np
import random


class Generator(nn.Module):

    def __init__(self, input_dim, h_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, input_dim),
        )

    def forward(self, x):
        output = self.net(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()  # [0,1]分布内
        )

    def forward(self, x):
        output = self.net(x)
        # return output.view(-1)
        return output

input_dim = 6
real_data = torch.rand((2,3,6))
G = Generator(input_dim=6, h_dim=3)
D= Discriminator(input_dim=6, h_dim=3)
output = G(torch.rand(2,3,6))
o = D(output)
print(output.shape)
print(o.shape)



