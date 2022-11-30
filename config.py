from easydict import EasyDict as edict

config = edict()

# config.DATAPATH = 'D:\\Projects\\Datasets\\train'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate'
config.DATAPATH = 'dataset'
# config.DATAPATH = 'D:\Projects\Datasets\\3dsmax\\evaluate\\deepshading\Test'

# config.VALDATAPATH = 'Datastes\\Unity_sceenshot\\test1'
config.train_save_dir = 'result'
# config.save_name = '9-1-Unet-4'
# config.save_name = '8-6-Depth-sobel_gan_only_use_sobel_0.1+masked'
# config.save_name = '8-8-Sobel_gan=>0.8sl(local)+0.5depthGan(local)+0.1sobelGan(local)'
# config.save_name = '8-17-swim-transformer(b=32)-0,12,12' # 85 89
config.save_name = '11-17-RGB-LM-Normal-Boundary'

# Settings
config.dropout = 0.5
config.learning_rate = 1e-4
config.TRAIN_EPOCH = 100000
config.save_interval = 2000

# config.fc_lst = [128, 64, config.n_classes]

# config.pooling_ratio = 0.8
#
config.load_model = False

# config.load_model = T
#config.model_name = 'model.pkl'
#
# config.model_name = '%s/net_params_58000_2021-08-07_09-09.pkl' % config.save_name # only sobel_0.1
config.model_name = '%s/net_params_848469_2021-11-14_06-29.pkl-0.9453740235002217-0.673966199476171-0.839287229712592-0.890269381113504' % config.save_name
# config.model_name = '%s/' % config.save_name
# 'net_params_61161_2021-09-05_08-38.pkl-0.9627976627628311-0.924348873943962-0.3345514129853605-0.3747545005492552'
# net_params_42180_2021-09-05_04-19.pkl-0.8808338218943547-0.7382311202056965-0.4371414266447017-0.4107329613686921
# config.model_name = '%s/net_params_42150_2021-08-17_19-59.pkl' % config.save_name

# net_params_108000_2021-08-06_10-07.pkl
# net_params_94000_2021-08-06_10-07.pkl good
# config.model_name = '%s/net_params_94000_2021-08-02_10-29.pkl' % config.save_name
#
# config.model_name = '%s\\net_params_1390000_2020-02-26_10-58.pkl' % config.save_name


# Hyperparameters
config.epochs = 100000
bs = 128
config.bs = bs

# config.decay_every = int(665600/bs*20)
# print()

config.clip_value = 0.01
config.n_critic = 1