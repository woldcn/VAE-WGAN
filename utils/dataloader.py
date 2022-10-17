# Author：woldcn
# Create Time：2022/10/4 17:20
# Description：functions for data load

import json
from torch.utils.data import DataLoader
from utils.dataset import Protein_dataset


# get tvae data
# tvae model only has train data
def get_tvae_data(path, batch_size, shuffle):
    max_seq_len = 0
    train_seqs = []
    train_labels = []
    # get train seqs and labels
    with open(path, 'r') as f:
        json_data = json.load(f)
        # calc max_seq_len
        for item in json_data:
            seq_len = item['protein_length']
            max_seq_len = max(max_seq_len, seq_len)
        for item in json_data:
            train_seqs.append(str2num(item['primary'], max_seq_len))
            train_labels.append(item['log_fluorescence'][0])

    print("train dataset length: {}".format(len(train_labels)))
    print("max_seq_len: {}\n".format(max_seq_len))

    # transform to Dataset
    train_data = Protein_dataset(train_seqs, train_labels)

    # transform to DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader, max_seq_len

# get predictor data by file
def get_data_by_file(train_path, valid_path, test_path, batch_size, shuffle=False):
    max_seq_len = 0
    train_seqs = []
    valid_seqs = []
    test_seqs = []
    train_labels = []
    valid_labels = []
    test_labels = []
    # get train seqs and labels
    with open(train_path, 'r') as f:
        json_data = json.load(f)
        # calc max_seq_len
        for item in json_data:
            seq_len = item['protein_length']
            max_seq_len = max(max_seq_len, seq_len)
        for item in json_data:
            train_seqs.append(str2num(item['primary'], max_seq_len))
            train_labels.append(item['log_fluorescence'][0])

    # get valid seqs and labels
    with open(valid_path, 'r') as f:
        json_data = json.load(f)
        # calc max_seq_len
        for item in json_data:
            seq_len = item['protein_length']
            max_seq_len = max(max_seq_len, seq_len)
        for item in json_data:
            valid_seqs.append(str2num(item['primary'], max_seq_len))
            valid_labels.append(item['log_fluorescence'][0])

    # get test seqs and labels
    with open(test_path, 'r') as f:
        json_data = json.load(f)
        # calc max_seq_len
        for item in json_data:
            seq_len = item['protein_length']
            max_seq_len = max(max_seq_len, seq_len)
        for item in json_data:
            test_seqs.append(str2num(item['primary'], max_seq_len))
            test_labels.append(item['log_fluorescence'][0])

    print("train dataset length: {}".format(len(train_labels)))
    print("valid dataset length: {}".format(len(valid_labels)))
    print("test dataset length: {}".format(len(test_labels)))
    print("max_seq_len: {}\n".format(max_seq_len))

    # transform to Dataset
    train_data = Protein_dataset(train_seqs, train_labels)
    valid_data = Protein_dataset(valid_seqs, valid_labels)
    test_data = Protein_dataset(test_seqs, test_labels)

    # transform to DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader, valid_loader, test_loader, max_seq_len


# get predictor data by percent
def get_data_by_percent(path, percent_train, percent_valid, percent_test, batch_size, shuffle=False):
    with open(path, 'r') as f:
        json_data = json.load(f)
        max_seq_len = 0
        seqs = []
        labels = []

        # calc max_seq_len
        for item in json_data:
            seq_len = item['protein_length']
            max_seq_len = max(max_seq_len, seq_len)

        # get seqs and labels
        for item in json_data:
            seqs.append(str2num(item['primary'], max_seq_len))
            labels.append(item['log_fluorescence'][0])

        # divide train, valid, test
        train_seqs = seqs[:int(percent_train * len(json_data))]
        train_labels = labels[:int(percent_train * len(json_data))]
        valid_seqs = seqs[int(percent_train * len(json_data)):int((percent_train + percent_valid) * len(json_data))]
        valid_labels = labels[int(percent_train * len(json_data)):int((percent_train + percent_valid) * len(json_data))]
        test_seqs = seqs[int((percent_train + percent_valid) * len(json_data)):]
        test_labels = labels[int((percent_train + percent_valid) * len(json_data)):]
        print("train dataset length: {}".format(len(train_seqs)))
        print("valid dataset length: {}".format(len(valid_seqs)))
        print("test dataset length: {}".format(len(test_seqs)))
        print("max_seq_len: {}\n".format(max_seq_len))

        # transform to Dataset
        train_data = Protein_dataset(train_seqs, train_labels)
        valid_data = Protein_dataset(valid_seqs, valid_labels)
        test_data = Protein_dataset(test_seqs, test_labels)

        # transform to DataLoader
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, valid_loader, test_loader, max_seq_len


def str2num(s, max_seq_len):
    s = s.upper()
    result = []
    for c in s:
        result.append(ord(c) - 64)
    # padding with 0
    result += [0] * (max_seq_len - len(result))
    return result
