import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
from model import Siren,MLP,DinerMLP,DinerSiren,MLP
from dataio import ImageData
from skimage import io
import skimage
import configargparse
from tensorboardX import SummaryWriter, writer
import time
import utils
from sklearn.preprocessing import normalize
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from opt import HyperParameters


class Logger:
    filename = None
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')
    
    @staticmethod
    def write_file(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')



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
    experiment_name         =       opt.experiment_name


    # make directory
    log_dir = "log"
    utils.cond_mkdir(os.path.join(log_dir,experiment_name))

    # check parameters
    if steps % steps_til_summary:
        raise ValueError("Steps_til_summary could not be devided by steps!")

    # logger
    Logger.filename = os.path.join(log_dir, experiment_name,'log.txt')

    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 1

    Dataset = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)
    model_input,gt = Dataset[0]

    model_input = model_input.to(device)
    gt = gt.to(device)


    hash_table_length = model_input.shape[0]

    model = MLP(in_features= input_dim,
                out_features=out_features,
                hidden_layers=hidden_layers,
                hidden_features=hidden_features
                ).to(device)

        
    optimizer = optim.Adam(lr = lr,params = model.parameters())


    # training process
    psnr_epoch = np.zeros(epochs + 1)

    with tqdm(total=epochs) as pbar:
        max_psnr = 0
        time_cost = 0
        for epoch in range(epochs):
            time_start = time.time()
            
            loss = 0
            model_output = model(model_input)

            loss = criteon(model_output,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            time_cost += time.time() - time_start

            psnr_epoch[epoch+1] = utils.loss2psnr(loss)

            cur_psnr = utils.loss2psnr(loss)
            max_psnr = max(max_psnr,cur_psnr)

            if (epoch+1) % 1000 == 0:
                with torch.no_grad():
                    utils.render_raw_image(model,os.path.join(log_dir,experiment_name,f'recon_{(epoch+1):05d}.png'),[1200,1200],gray = True)

            if (epochs + 1) % steps_til_summary == 0:
                log_str = f"[TRAIN] Epoch: {epoch+1} Loss: {loss.item()} PSNR: {cur_psnr} Time: {round(time_cost, 2)}"
                Logger.write(log_str)

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")


    return time_cost

if __name__ == "__main__":

    opt = HyperParameters()
    train_img(opt)

