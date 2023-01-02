import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,HashSiren,HashMLP,MLP,HashMLP_idx,HashSiren_idx
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
import scipy.io




def train_img(opt):
    img_path                =       opt.img_path
    steps                   =       opt.steps
    lr                      =       opt.lr
    hidden_layers           =       opt.hidden_layers
    hidden_features         =       opt.hidden_features
    sidelength              =       opt.sidelength
    render_img_resolution   =       opt.render_img_resolution
    grayscale               =       opt.grayscale
    first_omega_0           =       opt.w0
    hidden_omega_0          =       opt.w0
    model_type              =       opt.model_type
    render_img_mod          =       opt.render_img_mod 
    log_psnr                =       opt.log_psnr
    steps_til_summary       =       opt.steps_til_summary
    log_training_time       =       opt.log_training_time
    save_mod                =       opt.save_mod
    input_dim               =       opt.input_dim
    render_hash_img_mod     =       opt.render_hash_img_mod
    log_psnr_file           =       opt.log_psnr_file
    epochs                  =       opt.epochs
    save_mod_path           =       opt.save_mod_path
    render_img_path         =       opt.render_img_path
    remain_raw_resolution   =       opt.remain_raw_resolution
    save_mod_prefix         =       opt.save_mod_prefix
    log_psnr_prefix         =       opt.log_psnr_prefix
    render_img_prefix       =       opt.render_img_prefix
    hash_table_resolution   =       opt.hash_table_resolution
    """
    check parameters
    """
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps,please set correct number!")


    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]


    model_input = model_input.to(device)

    gt = gt.to(device)

    hash_table_length = gt.shape[0]

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
        model = HashSiren_idx(
                      hash_table_length = hash_table_length,
                      in_features = input_dim,
                      hidden_features = hidden_features,
                      hidden_layers = hidden_layers,
                      out_features = out_features,
                      outermost_linear = True,
                      first_omega_0 = first_omega_0,
                      hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'HashMLP':
        model = HashMLP_idx(
                        hash_table_length = hash_table_length, 
                        in_features = input_dim, 
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")



    optimizer = optim.Adam(lr = lr,params = model.parameters())


    stripe_length = 800
    stripe_numbers = int(sidelength[0] / stripe_length)
    indexes = torch.linspace(0,stripe_numbers-1,stripe_numbers)

    loader = DataLoader(
        dataset = indexes,
        batch_size = 1,
        shuffle = True,
        num_workers = 8)




    """
    training process
    """
    psnr_logger = np.zeros((epochs+1))



    with tqdm(total=epochs) as pbar:
        max_psnr = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for step,idx in enumerate(loader):
                loss = 0
                model_output = model(int(idx * sidelength[1] * stripe_length) , int((idx+1) * sidelength[1] * stripe_length ))

                loss = criteon(model_output,gt[int(idx * sidelength[1] * stripe_length) : int((idx+1) * sidelength[1] * stripe_length ),:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            epoch_loss /= stripe_numbers



            cur_psnr = utils.loss2psnr(epoch_loss)

            if (epoch+1) % steps_til_summary == 0:
                psnr_logger[int((epoch+1) / steps_til_summary)] = cur_psnr

            max_psnr = max(max_psnr,cur_psnr)

            if (epoch+1) % 10 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f" % (epoch+1, epoch_loss, cur_psnr))

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")

    utils.save_data(psnr_logger,os.path.join('pluto',f'pluto_psnr_method{model_type}_epoch{epochs:05d}_tablelength{input_dim:02d}.mat'))

    utils.render_raw_image_batch(model,os.path.join('pluto',f'pluto_recon_method{model_type}_epoch{epochs:05d}_tablelength{input_dim:02d}.png'),[8000,8000])

    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()

    train_img(opt)


  