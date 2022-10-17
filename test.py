import torch
import torch.nn as nn
from models.TCN import TemporalConvNet
from models.TVAE import TVAE
import torch.nn.functional as F

def criterion(x_refactor, x, mu, log_signma):
    refactor_loss = F.binary_cross_entropy(x_refactor, x)
    kl_div = - 0.5 * torch.sum(1 + log_signma - mu.pow(2) - log_signma.exp())
    return refactor_loss + kl_div

input_dim = 1
hidden_dim = 100
latent_dim = 3
num_channels = [1, 20]
input = torch.rand(100, 237)
print(input.dtype)

# model = TVAE(input_dim, num_channels, latent_dim)
# x_refactor, mu, log_sigma = model(input)
# print(x_refactor.size())
# print(mu.size())
# print(log_sigma.size())
# loss = criterion(x_refactor, input, mu, log_sigma)
# print(loss)
