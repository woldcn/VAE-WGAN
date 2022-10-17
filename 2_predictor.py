# Author：woldcn
# Create Time：2022/10/4 16:59
# Description：predictor runnable file.

from args import predicotr_args as cf
from utils.dataloader import get_data_by_file
from utils.dataloader import get_data_by_percent
from models.GRU import GRU
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from datetime import datetime


def main():
    # print args
    for arg in dir(cf):
        if arg[0] != '_':
            print('{}: {}'.format(arg, getattr(cf, arg)))

    # 1. load data
    train_loader, valid_loader, test_loader, max_seq_len = get_data_by_percent(cf.test_path, cf.percent_train,
                                                                            cf.percent_valid, cf.percent_test,
                                                                            cf.batch_size, cf.shuffle)
    # train_loader, valid_loader, test_loader, max_seq_len = get_data_by_file(cf.train_path, cf.valid_path, cf.test_path, cf.batch_size, cf.shuffle)


    # 2. load model, optimizer, criterion
    torch.manual_seed(cf.rand_seed)  # fix rand_seed
    model = GRU(max_seq_len, cf.hidden_dim, cf.dropout)
    model.to(cf.device)
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.wd)
    criterion = nn.MSELoss()


    # 3. train, valid, test
    max_valid_pcc = 0
    max_pcc_epoch = 0
    for epoch in range(cf.epochs):
        train_loss = valid_loss = test_loss = 0.0
        train_pcc = valid_pcc = test_pcc = 0.0

        # train
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(cf.device)
            targets = targets.to(cf.device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            train_pcc += np.corrcoef(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())[0][1]
        train_loss /= len(train_loader)
        train_pcc /= len(train_loader)

        # valid
        for batch in valid_loader:
            inputs, targets = batch
            inputs = inputs.to(cf.device)
            targets = targets.to(cf.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.data.item()
            valid_pcc += np.corrcoef(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())[0][1]
        valid_loss /= len(valid_loader)
        valid_pcc /= len(valid_loader)

        # test
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(cf.device)
            targets = targets.to(cf.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data.item()
            test_pcc += np.corrcoef(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())[0][1]
        test_loss /= len(test_loader)
        test_pcc /= len(test_loader)

        print(
            'Epoch: {}, train pcc: {:.4f}, train loss: {:.4f}, valid pcc: {:.4f}, valid loss: {:.4f}, test pcc: {:.4f}, test loss: {:.4f}'.format(
                epoch, train_pcc, train_loss, valid_pcc, valid_loss, test_pcc, test_loss))

        max_valid_pcc = max(max_valid_pcc, valid_pcc)
        # save model
        if valid_pcc > max_valid_pcc:
            max_valid_pcc = valid_pcc
            max_pcc_epoch = epoch
            torch.save(model, cf.save)
    print('=====================max_valid_pcc: {}, at epoch: {}\n\n'.format(max_valid_pcc, max_pcc_epoch))


if __name__ == '__main__':
    print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
    main()
    print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
