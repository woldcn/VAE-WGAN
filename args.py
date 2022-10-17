# Author：woldcn
# Create Time：2022/10/4 16:44
# Description：all args set here.


class PredictorArgs:
    device = 'cuda'
    rand_seed = 42
    file = 'data/predictor/fluorescence/fluorescence_train.json'
    train_path = 'data/predictor/fluorescence/fluorescence_train.json'
    valid_path = 'data/predictor/fluorescence/fluorescence_valid.json'
    test_path = 'data/predictor/fluorescence/fluorescence_test.json'
    shuffle = False
    percent_train = 0.6
    percent_valid = 0.2
    percent_test = 0.2
    hidden_dim = 100
    epochs = 500
    batch_size = 1000  # 32200  #32168
    lr = 0.001
    wd = 0.01
    dropout = 0.3
    save = './output/predictor/model/predictor.pth'
    result = './output/predictor/result/result.txt'


class TvaeArgs:
    device = 'cuda'
    rand_seed = 42
    file = 'data/predictor/fluorescence/fluorescence_train.json'
    shuffle = False
    input_dim = 1
    hidden_dim = 100
    num_channels = [1, 100]
    latent_dim = 3
    epochs = 100
    batch_size = 1000
    lr = 0.001
    wd = 0.01
    save = './output/tvae/model/tvae.pth'
    result = './output/tvae/result/result.txt'


predictor_args = PredictorArgs()
tvae_args = TvaeArgs()
