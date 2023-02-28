import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,NeRF,WaveNet,MLP,ComplexHashSiren,HashMLP,HashSiren
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
from utils import *
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
    """
    check parameters
    """
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps,please set correct number!")

    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 1

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = True,
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

    elif model_type == 'HashSiren':
        model = HashSiren(hash_mod = hash_mod,
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

    # iter_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
    # psnr_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
    psnr_logger = np.zeros(int(epochs/steps_til_summary + 1))

    Total_time = 0

    with tqdm(total=epochs) as pbar:
        # for step in range(steps):        
        
        max_psnr = 0
        start_time = time.time()   
        for epoch in range(epochs):
            
            loss = 0
            model_output = model(model_input)

            loss = criteon(model_output,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 等待GPU完成同步
            torch.cuda.synchronize()

            Total_time += (time.time() - start_time) / 1000.


            psnr_logger[int((epoch+1) / steps_til_summary)] = utils.loss2psnr(loss)

            cur_psnr = utils.loss2psnr(loss)
            max_psnr = max(max_psnr,cur_psnr)

            if (epoch+1) % 100 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f,total time %0.6fs" % (epoch+1, loss, cur_psnr,Total_time))

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")


    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()
    psnr_sum = np.zeros((5,30,10001))
    for i in range(1,6):
        opt.input_dim = i
        for pic_idx in range(1,31):
            opt.img_path = f'pic/RGB_OR_1200x1200_0{pic_idx:02d}.png'
            psnr_sum[i-1,pic_idx-1,:] = train_img(opt)

    scipy.io.savemat('img_exp/gray.mat', {"data":psnr_sum})