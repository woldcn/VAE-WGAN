# Author：woldcn
# Create Time：2022/10/17 8:52
# Description：Temporal Convolution Network-Variational Autoencoder model definition.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TCN import TemporalConvNet


class TVAE(nn.Module):
    def __init__(self, input_dim, num_channels, latent_dim):
        super(TVAE, self).__init__()

        # encoder
        self.encoder_tcn = TemporalConvNet(input_dim, num_channels)
        self.mu = nn.Linear(num_channels[-1], latent_dim, bias=True)
        self.log_sigma = nn.Linear(num_channels[-1], latent_dim, bias=True)

        # decoder
        self.decoder_fc = nn.Linear(latent_dim, num_channels[-1])
        self.decoder_tcn = TemporalConvNet(num_channels[-1], num_channels[::-1])

    def encode(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        x = self.encoder_tcn(x)
        x = torch.sigmoid(x)
        mu = self.mu(x)
        mu = torch.sigmoid(mu)
        log_sigma = self.log_sigma(x)
        log_sigma = torch.sigmoid(log_sigma)
        return mu, log_sigma

    def latent(self, mu, log_sigma):
        std = torch.exp(log_sigma/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = self.decoder_tcn(x)
        x = x.reshape(x.shape[0], x.shape[1])
        return x

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.latent(mu, log_sigma)
        x_refactor = self.decode(z)
        x_refactor = torch.sigmoid(x_refactor)
        return x_refactor, mu, log_sigma

    def criterion(x_refactor, x, mu, log_signma):
        refactor_loss = F.binary_cross_entropy(x_refactor, x)
        kl_div = - 0.5 * torch.sum(1 + log_signma - mu.pow(2) - log_signma.exp())
        return refactor_loss + kl_div
