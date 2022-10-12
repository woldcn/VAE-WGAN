# Author：woldcn
# Create Time：2022/10/4 21:18
# Description：GRU model definition.

import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(GRU, self).__init__()
        self.hidden = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=False, num_layers=2, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(input_dim * hidden_dim, 1)

        # self.fnn1 = nn.Linear(input_dim, 50)
        # self.fnn2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, h = self.gru(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        # x = torch.sigmoid(x)
        # x = torch.relu(x)
        x = x.reshape(x.shape[0])

        return x
