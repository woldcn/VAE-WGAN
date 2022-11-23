# Author：woldcn
# Create Time：2022/10/4 16:44
# Description：tvae runnable file.

from args import tvae_args as cf
from utils.dataloader import get_tvae_data
from models.TVAE import TVAE
import torch
from torch import optim
from utils.pic import plt_single_line
from utils.log import Log

def main():
    # init Log class
    log = Log(cf.log, cf)

    # 1. load data
    train_loader, max_seq_len, length_train_dataset = get_tvae_data(cf.file, cf.batch_size, cf.shuffle)
    log.print("train dataset length: {}".format(length_train_dataset))
    log.print("max_seq_len: {}\n".format(max_seq_len))

    # 2. load model, optimizer, criterion
    torch.manual_seed(cf.rand_seed)  # fix rand_seed
    model = TVAE(cf.input_dim, cf.num_channels, cf.latent_dim)
    model.to(cf.device)
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.wd)

    # 3. train
    loss_list = []
    for epoch in range(cf.epochs):
        loss = 0
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
        loss_list.append(loss.cpu().detach().numpy())

        log.print('Epoch: {}, loss: {:.4f}'.format(epoch, loss))

    # save model
    torch.save(model, cf.save)
    log.save()
    # painting picture
    plt_single_line(loss_list[1:], cf.loss_pic, label_x='epoch', label_y='loss', title='TVAE Train Loss')


if __name__ == '__main__':
    main()


