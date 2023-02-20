import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
from model import DinerSiren_interp,DinerMLP_interp
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
    grayscale               =       opt.grayscale
    first_omega_0           =       opt.w0
    hidden_omega_0          =       opt.w0
    model_type              =       opt.model_type
    steps_til_summary       =       opt.steps_til_summary
    input_dim               =       opt.input_dim
    epochs                  =       opt.epochs
    remain_raw_resolution   =       opt.remain_raw_resolution
    hash_table_resolution   =       opt.hash_table_resolution
    experiment_name         =       opt.experiment_name   

    # check parameters
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps,please set correct number!")

    # make directory
    log_dir = "log"
    utils.cond_mkdir(os.path.join(log_dir,experiment_name))


    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]

    model_input = model_input.to(device)

    gt = gt.to(device)

    if model_type == 'DinerSiren':
        model = DinerSiren_interp(opt = opt,
                          hash_table_resolution = hash_table_resolution,
                          in_features = input_dim, # hash table width
                          hidden_features = hidden_features,
                          hidden_layers = hidden_layers,
                          out_features = out_features,
                          outermost_linear = True,
                          first_omega_0 = first_omega_0,
                          hidden_omega_0 = hidden_omega_0).to(device = device)


    elif model_type == 'DinerMLP':
        model = DinerMLP_interp(hash_table_resolution = hash_table_resolution, 
                        in_features = input_dim, # hash table width
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")


    optimizer = optim.Adam(lr = lr,params = model.parameters())


    # training process
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

            if (epoch+1) % 1000 == 0:
                with torch.no_grad():
                    utils.render_raw_image(model,os.path.join(log_dir,experiment_name,f'recon_{(epoch+1):05d}.png'),[1200,1200])

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")

    # with torch.no_grad():
    #     utils.render_raw_image(model,os.path.join(log_dir,experiment_name,'recon.png'),[1200,1200])

    # scipy.io.savemat('interpolate_experiments_results/psnr/psnr_2_interp.mat',{"data":psnr_logger})
    # utils.render_raw_image(model,os.path.join('interpolate_experiments_results','render','result_2.png'),[1200,1200])

    return psnr_logger

if __name__ == "__main__":

    opt = HyperParameters()
    train_img(opt)

