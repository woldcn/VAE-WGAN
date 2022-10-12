# Author：woldcn
# Create Time：2022/10/4 16:44
# Description：

class predicotr_args:
    device = 'cuda'
    rand_seed = 42
    file = r'E:\lian_paper\VAE-WGAN\data\predictor\fluorescence\fluorescence_train.json'
    train_path = r'E:\lian_paper\VAE-WGAN\data\predictor\fluorescence\fluorescence_train.json'
    valid_path = r'E:\lian_paper\VAE-WGAN\data\predictor\fluorescence\fluorescence_valid.json'
    test_path = r'E:\lian_paper\VAE-WGAN\data\predictor\fluorescence\fluorescence_test.json'
    shuffle = False
    percent_train = 0.6
    percent_valid = 0.2
    percent_test = 0.2
    hidden_dim = 100
    epochs = 500
    batch_size = 1000 #32200  #32168
    lr = 0.001
    wd = 0.01
    dropout = 0.3
    model_path = './ouput/predictor/model/predictor.pth'
    model_result = './ouput/predictor/result/result.txt'

predicotr_args = predicotr_args()