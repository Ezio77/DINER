import numpy as np
import os
import torch
from torch import optim, nn
import configargparse
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def HyperParameters():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config', required=False, is_config_file=True,
                    help='Path to config file.')

    # General training options
    p.add_argument('--random_seed',action='store_true', default=False,
                    help = 'Random seed')
    p.add_argument('--seed', type = int, default=1,
                    help = 'Training seed')
    p.add_argument('--training_data_type',type = str,default='image',
                    help = 'Training data type')
    p.add_argument('--lr', type = float, default = 1e-4,
                    help = 'learning rate. default=1e-4')
    p.add_argument('--steps', type = int, default = 3000,
                    help = 'Number of iterations to train for.')
    p.add_argument('--gpu', type = int, default = 1,
                    help = 'GPU ID to use')
    p.add_argument('--epochs',type=int, default = 3000)

    # logging options
    p.add_argument('--save_mod_prefix',type = str)
    p.add_argument('--log_psnr_prefix',type = str)
    p.add_argument('--render_img_prefix',type = str)
    p.add_argument('--log_psnr_file',type = str)
    p.add_argument('--render_hash_img_mod',action = 'store_true', default = False)
    p.add_argument('--render_hash_img_path',type = str,default='')
    p.add_argument('--log_training_time',action = 'store_true',default = False)
    p.add_argument('--log_psnr',action = 'store_true',default = False)
    p.add_argument('--steps_til_summary',type = int)
    p.add_argument('--render_img_mod',action = 'store_true', default=False)
    p.add_argument('--render_video_mod',action = 'store_true',default=False)
    p.add_argument('--render_video_dir',type = str)
    p.add_argument('--render_volume_mod',action = 'store_true', default=False,
                    help = 'whether to render volume or not')
    p.add_argument('--restore_oneDim_resolution',type = int,
                    help = 'restore one dim data resolution')
    p.add_argument('--render_img_resolution',nargs = '+', type = int,
                    help = 'rendered image resolution')
    p.add_argument('--render_img_path', type = str,
                    help = 'path to the rendered image')
    p.add_argument('--render_1dim_path',type = str,
                    help = 'path to restore one dimension data')

    # dataset options
    p.add_argument('--video_path',type=str)
    p.add_argument('--sdf_path',type=str)
    p.add_argument('--remain_raw_resolution',action='store_true',default=False)
    p.add_argument('--color_type',type=str, default="RGB",
                    help= 'color type of image data')
    p.add_argument('--load_1dim_data',type=str,
                    help='path to one dimension data')
    p.add_argument('--data_length',type=int, default=1000,
                    help='length of one dimention data')
    p.add_argument('--data_distribution',type=str,
                    help='one dimention data distribution type: norm or uniform')  
    p.add_argument('--img_path', type=str, default = "pic/RGB_OR_1200x1200_045.png",
                    help='path to the source image')
    p.add_argument('--sidelength', nargs='+', type=int,
                    help='resized image resolutions')
    p.add_argument('--hash_table_resolution',nargs='+',type=int)
    p.add_argument('--grayscale', action='store_true', default=False,
                    help='whether to use grayscale')

    # model options
    p.add_argument('--n_hash',type = int, default = 1,
                    help='number of multihead hash')
    p.add_argument('--save_mod',action='store_true',default=False,
                    help='whether to save the model')
    p.add_argument('--save_mod_path',type=str)
    p.add_argument('--hash_mod',action='store_true',default = False,
                    help='whether to use hash table')
    p.add_argument('--model_type',type=str,
                    help='model type : siren or nerf or wavelet')
    p.add_argument('--input_dim', type=int, default=2, 
                    help='dimentions of input')             
    p.add_argument('--hidden_features', type=int,
                    help='hidden features in network')
    p.add_argument('--hidden_layers', type=int,
                    help='hidden layers in network')
    p.add_argument('--w0', type=int, default=30,
                    help='w0 for the siren model.')
    p.add_argument('--N_freqs', type=int, default=16,
                    help='position embedding frequency numbers')
    p.add_argument('--siren_hidden_features',type=int, default=128)
    p.add_argument('--siren_hidden_layers',type=int, default=4)

    opt = p.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    for k, v in opt.__dict__.items():
        print(k, v)

    if not opt.random_seed:
        setup_seed(opt.seed)

    return opt




