import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import HashSiren_interp,HashMLP_interp
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


def test_interp(opt,model,model_input):
    # device = torch.device('cuda')
    
    with torch.no_grad():
        rgb = utils.to_numpy(model(model_input))
        rgb = rgb.reshape(opt.sidelength[0],opt.sidelength[1],3)
        rgb = (((rgb+1.) / 2.) * 255).astype(np.uint8)

        io.imsave('interpolate_experiments_results/re.png',rgb)

    raw_img = skimage.io.imread('pic/RGB_OR_1200x1200_001.png')

    return skimage.metrics.peak_signal_noise_ratio(rgb,raw_img)



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


    """
    make directory
    """
    def makeDir():
        utils.cond_mkdir(save_mod_prefix)
        utils.cond_mkdir(log_psnr_prefix)
        utils.cond_mkdir(render_img_prefix)
    makeDir()


    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    raw_model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]


    model_input = raw_model_input.reshape(opt.sidelength[0],opt.sidelength[1],2)
    model_input = model_input[::2,::2,:] # 取奇数行和奇数列
    model_input = model_input.reshape(-1,2)
    model_input = model_input.to(device)

    gt = gt.reshape(opt.sidelength[0],opt.sidelength[1],3)
    gt = gt[::2,::2,:]
    gt = gt.reshape(-1,3)
    gt = gt.to(device)


    if model_type == 'HashSiren':
        model = HashSiren_interp(
                          hash_table_resolution = hash_table_resolution,
                          in_features = input_dim, # hash table width
                          hidden_features = hidden_features,
                          hidden_layers = hidden_layers,
                          out_features = out_features,
                          outermost_linear = True,
                          first_omega_0 = first_omega_0,
                          hidden_omega_0 = hidden_omega_0).to(device = device)


    elif model_type == 'HashMLP':
        model = HashMLP_interp(
                        hash_table_resolution = hash_table_resolution, 
                        in_features = input_dim, # hash table width
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")



    optimizer = optim.Adam(lr = lr,params = model.parameters())

    """
    training process
    """

    # iter_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
    # psnr_logger = np.zeros_like(iter_logger)
    # psnr_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
    psnr_logger = np.zeros((epochs+1))

    with tqdm(total=epochs) as pbar:

        max_psnr = 0

        for epoch in range(epochs):
            
            loss = 0
            model_output = model(model_input)

            loss = criteon(model_output,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % steps_til_summary == 0:
                psnr_logger[int((epoch+1) / steps_til_summary)] = utils.loss2psnr(loss)

            psnr_temp = utils.loss2psnr(loss)

            max_psnr = max(max_psnr,psnr_temp)

            if (epoch+1) % 100 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f" % (epoch+1, loss, psnr_temp))

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")
    

    print("*"*30)
    print(test_interp(opt,model,raw_model_input))
    print("*"*30)


    # scipy.io.savemat('interpolate_experiments_results/psnr/psnr_2_interp.mat',{"data":psnr_logger})

    # utils.render_raw_image(model,os.path.join('interpolate_experiments_results','render','result_2.png'),[1200,1200])

    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()

    train_img(opt)


