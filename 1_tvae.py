# Author：woldcn
# Create Time：2022/10/4 16:44
# Description：tvae runnable file.

from args import tvae_args as cf
from utils.dataloader import get_tvae_data
from models.TVAE import TVAE
import torch
from torch import optim
from datetime import datetime


def main():
    # print args
    for arg in dir(cf):
        if arg[0] != '_':
            print('{}: {}'.format(arg, getattr(cf, arg)))

    # 1. load data
    train_loader, max_seq_len = get_tvae_data(cf.file, cf.batch_size, cf.shuffle)

    # 2. load model, optimizer, criterion
    torch.manual_seed(cf.rand_seed)  # fix rand_seed
    model = TVAE(cf.input_dim, cf.num_channels, cf.latent_dim)
    model.to(cf.device)
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.wd)

    # 3. train
    min_loss = 1 << 32
    min_loss_epoch = 0
    for epoch in range(cf.epochs):
        loss = 0
        loss_list = []
        # 3.1 train
        for batch in train_loader:
            inputs, _ = batch
            inputs = inputs.to(cf.device)
            inputs = inputs.float()
            inputs = torch.sigmoid(inputs)
            x_refactor, mu, log_sigma = model(inputs)
            optimizer.zero_grad()
            loss = TVAE.criterion(x_refactor, inputs, mu, log_sigma)
            loss.backward()
            optimizer.step()
            loss += loss.data.item()
        loss /= len(train_loader)
        loss_list.append(loss)

        print(
            'Epoch: {}, loss: {:.4f}'.format(epoch, loss))

        # save model
        # if loss < min_loss:
        #     min_loss = loss
        #     min_loss_epoch = epoch
        #     torch.save(model, cf.save)
    # save model
    torch.save(model, cf.save)
    print('=====================min_loss: {}, at epoch: {}\n\n'.format(min_loss, min_loss_epoch))


if __name__ == '__main__':
    print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
    main()
    print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
