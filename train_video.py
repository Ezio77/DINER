import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,NeRF,WaveNet,MLP,HashMLP,HashSiren,HashMLP_idx
from dataio import ImageData,oneDimData,VideoData
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
    steps                   =       opt.steps
    lr                      =       opt.lr
    hidden_layers           =       opt.hidden_layers
    hidden_features         =       opt.hidden_features
    render_img_resolution   =       opt.render_img_resolution
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
    save_mod_prefix         =       opt.save_mod_prefix
    log_psnr_prefix         =       opt.log_psnr_prefix
    render_img_prefix       =       opt.render_img_prefix
    video_path              =       opt.video_path
    """
    check parameters
    """
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps,please set correct number!")

    

    """
    make directory
    """
    # def makeDir():
    #     utils.cond_mkdir(save_mod_prefix)
    #     utils.cond_mkdir(log_psnr_prefix)
    #     utils.cond_mkdir(render_img_prefix)
    # makeDir()

    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    model_input,gt = VideoData(path = video_path)[0]


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
        model = HashMLP_idx(
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

    frame_number = 30
    H = 1080
    W = 1920

    frame_indexes = torch.linspace(0,29,frame_number)

    loader = DataLoader(
        dataset = frame_indexes,
        batch_size = 1,
        shuffle = True,
        num_workers = 8)

    if log_psnr == True:
        # iter_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
        psnr_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))

    # if log_training_time == True:
    #     start_time = time.time()

    Total_time = 0
    # dataloader = DataLoader(MyDataset,batch_size = 30000,shuffle=True,num_workers=8)

    with tqdm(total=epochs) as pbar:        
        max_psnr = 0 
        for epoch in range(epochs):
            epoch_loss = 0
            for step,frame_idx in enumerate(loader):
                loss = 0
                model_output = model(int(frame_idx*H*W),int((frame_idx+1)*H*W))
                loss = criteon(model_output,gt[int(frame_idx*H*W):int((frame_idx+1)*H*W)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # # 等待GPU完成同步
                # torch.cuda.synchronize()
                epoch_loss += loss

            epoch_loss /= frame_number
            if (epoch+1) % steps_til_summary == 0 and log_psnr == True:
                psnr_logger[int((epoch+1) / steps_til_summary)] = utils.loss2psnr(epoch_loss)

            cur_psnr = utils.loss2psnr(epoch_loss)
            max_psnr = max(max_psnr,cur_psnr)

            tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f" % (epoch+1,epoch_loss,cur_psnr))       
            pbar.update(1)

    print("Training process finished!")
    print(f"max psnr: {max_psnr}")

    if opt.render_video_mod:
        print("Start rendering video images.")
        cond_mkdir(opt.render_video_dir)
        utils.render_video_images(model,H,W,frame_number,opt.render_video_dir)

    if log_psnr == True:
        cond_mkdir(log_psnr_prefix)
        utils.save_data(psnr_logger,save_path = os.path.join(log_psnr_prefix,log_psnr_file))

    return max_psnr

if __name__ == "__main__":

    opt = HyperParameters()
    train_img(opt)
