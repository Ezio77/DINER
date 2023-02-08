import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,NeRF,WaveNet,MLP,ComplexHashSiren,HashMLP,HashSiren,HashMLP_m
from dataio import ImageData,oneDimData
from skimage import io
from skimage.util import img_as_float
import skimage
import configargparse
from tensorboardX import SummaryWriter, writer
import time
import pdb
# from utils import loss2psnr
import utils
from sklearn.preprocessing import normalize
from PIL import Image
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from opt import HyperParameters


def train_img(opt):
    img_path                =       opt.img_path
    steps                   =       opt.steps
    lr                      =       opt.lr
    hidden_layers           =       opt.hidden_layers
    hidden_features         =       opt.hidden_features
    sidelength              =       opt.sidelength
    grayscale               =       opt.grayscale
    first_omega_0           =       opt.w0
    hidden_omega_0          =       opt.w0
    model_type              =       opt.model_type
    hash_mod                =       opt.hash_mod
    log_psnr                =       opt.log_psnr
    steps_til_summary       =       opt.steps_til_summary
    input_dim               =       opt.input_dim
    epochs                  =       opt.epochs
    remain_raw_resolution   =       opt.remain_raw_resolution
    experiment_name         =       opt.experiment_name

    """
    make directory
    """
    pre_dir = "experiment_results"
    utils.cond_mkdir(os.path.join(pre_dir,experiment_name))


    """
    check parameters
    """
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps!")

    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]

    model_input = model_input.to(device)
    gt = gt.to(device)

    hash_table_length = model_input.shape[0]

    if model_type == 'Siren':
        model = Siren(in_features = input_dim,
                          hidden_features = hidden_features,
                          hidden_layers = hidden_layers,
                          out_features = out_features,
                          ).to(device = device)

    
    elif model_type == 'MLP':
        model = MLP(in_features = input_dim,
                    out_features = out_features,
                    hidden_layers = hidden_layers,
                    hidden_features= hidden_features,
                    ).to(device = device)


    elif model_type == 'HashSiren':
        model = HashSiren(
                      hash_table_length = hash_table_length,
                      in_features = input_dim,
                      hidden_features = hidden_features,
                      hidden_layers = hidden_layers,
                      out_features = out_features,
                      outermost_linear = True,
                      first_omega_0 = first_omega_0,
                      hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'HashMLP':
        model = HashMLP(
                        hash_table_length = hash_table_length, 
                        in_features = input_dim, 
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")
        
    optimizer = optim.Adam(lr = lr,params = model.parameters())

    """
    training process
    """
    psnr_logger = np.zeros(int(epochs/steps_til_summary + 1))

    with tqdm(total=epochs) as pbar:
        max_psnr = 0  
        for epoch in range(epochs):
            
            loss = 0
            model_output = model(model_input)

            loss = criteon(model_output,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_logger[int((epoch+1) / steps_til_summary)] = utils.loss2psnr(loss)

            cur_psnr = utils.loss2psnr(loss)
            max_psnr = max(max_psnr,cur_psnr)

            if (epoch+1) % 100 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f" % (epoch+1, loss, cur_psnr))

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")

    with torch.no_grad():
        utils.render_raw_image(model,os.path.join(pre_dir,experiment_name,'recon_3000epoch','sort_3000epoch_recon.png'),[512,512])

        # gt = np.round(utils.to_numpy(((gt + 1) / 2 * 255))).astype(np.uint8)
        # table = utils.to_numpy(model.table)
        # data = np.concatenate([table,gt],axis=-1)

        # utils.save_data(data,os.path.join(pre_dir,experiment_name,'results','sort_table.mat'))

        # utils.save_data(psnr_logger,os.path.join(pre_dir,experiment_name,'results','sort_psnr.mat'))
        # utils.render_hash_3d_volume(model,[300,300,300],os.path.join(pre_dir,experiment_name,'results','sort_pcd.pcd'),os.path.join(pre_dir,experiment_name,'results','sort_hash_feature.mat'))


    
    """
    utils.save_data(data = (gt+1) / 2,save_path=os.path.join(pre_dir,experiment_name,"psnr.mat"))

    utils.render_raw_image(model,'12.27/hashMLP_recon.png',[600,600])
    utils.save_data(data = model.table,save_path='12.27/HashMLP_table.mat')

    utils.render_hash_image(model,[600,600],'12.27/hash_img.png')

    utils.render_hash_1d_line(model,1000000,'hash_images/channel_3_dim_1_linspace.mat')

    utils.render_hash_3d_volume(model,[100,100,100],f'3d_hash/channel_3_dim_3_{opt.epochs}epoch_pic29.pcd')

    data = torch.cat([model.table,model_output],dim=-1)
    utils.save_data(data,'hash_images/channel_3_dim_1.mat')

    utils.render_hash_image(model,render_img_resolution = [1200,1200],save_path='hash_images/channel_3_dim_2.png')
    """


    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()
    train_img(opt)

    """
    pre_dir = "experiment_results"
    psnr_log = np.zeros((30,10001))
    opt = HyperParameters()
    for i in range(1,31):
        opt.img_path = f'pic/RGB_OR_1200x1200_{i:03d}.png'
        psnr_log[i-1] = train_img(opt)
    
    utils.save_data(psnr_log,os.path.join(pre_dir,opt.experiment_name,f"diner_mlp_tableLength_{opt.input_dim:02d}_seed_{opt.seed:03d}.mat"))
    utils.save_data(psnr_log,os.path.join(pre_dir,opt.experiment_name,f"diner_mlp_tableLength_{opt.input_dim:02d}_seed_{opt.seed:03d}.npy"))
    """
