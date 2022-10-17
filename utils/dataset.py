# Author：woldcn
# Create Time：2022/10/4 20:27
# Description：implement Dataset class.

import torch
from torch.utils.data import Dataset

class Protein_dataset(Dataset):
    def __init__(self, seqs, labels):
        super(Protein_dataset, self).__init__()
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq = torch.tensor(seq)
        label = self.labels[index]
        label = torch.tensor(label)
        return seq, label