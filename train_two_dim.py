import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,NeRF,WaveNet,MLP,HashSiren,ComplexHashSiren,HashMLP
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


# def render_hash_img(model,render_img_resolution,save_path):
#     device = torch.device('cuda')
#     H = render_img_resolution[0]
#     W = render_img_resolution[1]
#     with torch.no_grad():
#     return 



def render_2_dim_image(model_output,render_img_resolution,save_path):
    H = render_img_resolution[0]
    W = render_img_resolution[1]
    pixels = (utils.to_numpy(model_output).reshape(H,W,-1) + 1.) / 2.
    padding = np.zeros((H,W,1))
    pixels = (np.concatenate([pixels,padding],axis=-1) * 255).astype(np.uint8)
    io.imsave(save_path,pixels)



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
    hash_mod                =       opt.hash_mod
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
    """
    check parameters
    """
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps,please set correct number!")


    """
    make directory
    """
    # def makeDir():
    #     cond_mkdir(save_mod_prefix)
    #     cond_mkdir(log_psnr_prefix)
    #     cond_mkdir(render_img_prefix)

    # makeDir()

    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 2

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]

    gt = gt[:,0:2]

    model_input = model_input.to(device)
    gt = gt.to(device)

    hash_table_length = model_input.shape[0]

    # xy,rgb = Dataset[0]
    # hash_table_length = len(ImageData(image_path=img_path,sidelength = sidelength,grayscale = grayscale,image_circle = image_circle))

    if model_type == 'Siren':
        model = Siren(in_features = input_dim,
                      hidden_features = hidden_features,
                      hidden_layers = hidden_layers,
                      out_features = out_features,
                      ).to(device = device)

    elif model_type == 'HashSiren':
        model = HashSiren(hash_table_length = hash_table_length,
                          in_features = input_dim, # hash table width
                          hidden_features = hidden_features,
                          hidden_layers = hidden_layers,
                          out_features = out_features,
                          outermost_linear = True,
                          first_omega_0 = first_omega_0,
                          hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'HashMLP':
        model = HashMLP(hash_table_length = hash_table_length, 
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

    # log psnr
    # scipy.io.savemat('two_dim/psnr_4.mat',{"data":psnr_logger})

    # data = torch.cat([model.table,model_output],dim = -1)
    # utils.save_data(data,'two_dim/points_4.mat')

    # render_2_dim_image(model_output,[1200,1200],'two_dim/pic_4.png')

    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()

    # for opt.imput_dim in [1,2,3,4,5]:
    for i in range(1,5):
        opt.input_dim = i
        for pic_idx in range(1,31):
            opt.img_path = f"pic/rg/RGB_OR_1200x1200_{pic_idx:03d}_rg.png"

            psnr = train_img(opt)

            if pic_idx == 1:
                psnr_logger = psnr[None,:]
            else:
                psnr_logger = np.concatenate([psnr_logger,psnr[None,:]],axis = 0)

        scipy.io.savemat(f'two_dim/{opt.model_type}_hashTableLength{opt.input_dim:02d}_epoch10000_results.mat',{"data":psnr_logger})