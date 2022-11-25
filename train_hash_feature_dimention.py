import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import optim, nn
import math
from model import Siren,NeRF,WaveNet,MLP,Siren_interp,ComplexHashSiren,HashMLP,PureSiren,peMLP
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
from opt import *



def train_img():
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
    N_freqs                 =       opt.N_freqs
    image_circle            =       opt.image_circle
    hash_mod                =       opt.hash_mod
    render_img_mod          =       opt.render_img_mod 
    log_psnr                =       opt.log_psnr
    steps_til_summary       =       opt.steps_til_summary
    log_training_time       =       opt.log_training_time
    save_mod                =       opt.save_mod
    input_dim               =       opt.input_dim
    color_type              =       opt.color_type
    n_hash                  =       opt.n_hash
    render_hash_img_mod     =       opt.render_hash_img_mod
    log_psnr_file           =       opt.log_psnr_file
    epochs                  =       opt.epochs
    siren_hidden_features   =       opt.siren_hidden_features
    siren_hidden_layers     =       opt.siren_hidden_layers
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
    cond_mkdir(save_mod_prefix)
    cond_mkdir(log_psnr_prefix)
    cond_mkdir(render_img_prefix)



    device = torch.device('cuda')
    criteon = nn.MSELoss()

    if grayscale:
        out_features = 1
    else:
        out_features = 3

    model_input,gt = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)[0]


    model_input = model_input.to(device)
    gt = gt.to(device)

    hash_table_length = model_input.shape[0]

    # xy,rgb = Dataset[0]
    # hash_table_length = len(ImageData(image_path=img_path,sidelength = sidelength,grayscale = grayscale,image_circle = image_circle))




    if model_type == 'Siren':
        model = PureSiren(in_features = input_dim,
                          hidden_features = hidden_features,
                          hidden_layers = hidden_layers,
                          out_features = out_features,
                          ).to(device = device)

    elif model_type == 'HashSiren':
        model = Siren(hash_mod = hash_mod,
                      hash_table_length = hash_table_length,
                      in_features = input_dim,
                      hidden_features = hidden_features,
                      hidden_layers = hidden_layers,
                      out_features = out_features,
                      outermost_linear = True,
                      first_omega_0 = first_omega_0,
                      hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'instantNGP':
        model = HashMLP(hash_mod = hash_mod,
                        hash_table_length = hash_table_length, 
                        in_features = input_dim, 
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    elif model_type == 'ComplexHashSiren':
        model = ComplexHashSiren(hash_mod = hash_mod,
                                 hash_table_length = hash_table_length,
                                 in_features = input_dim,
                                 hidden_features = hidden_features,
                                 hidden_layers = hidden_layers,
                                 out_features = out_features,
                                 siren_hidden_features = siren_hidden_features,
                                 siren_hidden_layers = siren_hidden_layers,
                                 outermost_linear = True,
                                 first_omega_0 = first_omega_0,
                                 hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'peMLP':
        model = peMLP(in_features = input_dim,
                      out_features = out_features,
                      hidden_layers = hidden_layers,
                      hidden_features = hidden_features
                      ).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")
        
    optimizer = optim.Adam(lr = lr,params = model.parameters())


    """
    training process
    """

    if log_psnr == True:
        iter_logger = np.linspace(0,epochs,int(epochs/steps_til_summary + 1))
        psnr_logger = np.zeros_like(iter_logger)

    # if log_training_time == True:
    #     start_time = time.time()

    Total_time = 0

    # dataloader = DataLoader(MyDataset,batch_size = 30000,shuffle=True,num_workers=8)

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

            if (epoch+1) % steps_til_summary == 0 and log_psnr == True:
                psnr_logger[int((epoch+1) / steps_til_summary)] = utils.loss2psnr(loss)

            psnr_temp = utils.loss2psnr(loss)

            max_psnr = max(max_psnr,psnr_temp)


            if (epoch+1) % 100 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f,total time %0.6fs" % (epoch+1, loss, psnr_temp,Total_time))
                
            # if (epoch+1) % steps_til_summary == 0:
            #     pass
                # with torch.no_grad():
                    # new_dir = "pic45_res1000_HashSiren"
                    # cond_mkdir(f"results/{new_dir}/raw")
                    # cond_mkdir(f"results/{new_dir}/hash")
                    # cond_mkdir(f"results/{new_dir}/error")
                    # render_raw_image(model,save_path = f"results/{new_dir}/raw/raw_{epoch+1}.png")
                    # render_hash_image(model,render_img_resolution,save_path = f"results/{new_dir}/hash/hash_{epoch+1}.png")
                    # render_error_image(gt,model_output,save_path = f"results/{new_dir}/error/error_{epoch+1}.png")

            pbar.update(1)

    # if log_training_time == True:
    #     TotalTime = time.time() - start_time
    #     print(f"Total training time : {TotalTime}s.")

    print(f"MAX_PSNR : {max_psnr}")

    if render_img_mod:
        print("Start rendering image.")
        with torch.no_grad():
            render_raw_image(model,save_path=os.path.join(render_img_prefix,render_img_path))


    if render_hash_img_mod:
        print("Start rendering hash image.")
        with torch.no_grad():
            render_hash_image(model,render_img_resolution,False,0)

    if log_psnr == True:
        utils.cond_mkdir("log_psnr")
        # np.save(log_psnr_file,np.concatenate([iter_logger[None,:],psnr_logger[None,:]],axis = 0))
        np.save(os.path.join(log_psnr_prefix,log_psnr_file),np.concatenate([iter_logger[None,:],psnr_logger[None,:]],axis = 0))

    if save_mod == True:
        torch.save(model.state_dict(), os.path.join(save_mod_prefix,save_mod_path))

    return max_psnr,Total_time



if __name__ == "__main__":


    train()


    # if opt.training_data_type == 'oneDim':
    #     train_1dim()



