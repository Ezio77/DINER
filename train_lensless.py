import os
import torch
import numpy as np
import imageio.v2 as imageio
import configargparse
import time
import math
from scipy.io import loadmat
from model import HashSiren_Lessless
from utils import cond_mkdir
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

device = torch.device("cuda")
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log10(x)


def get_opts():
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')
    parser.add_argument('--root_dir', type=str, default= 'data/simulation_fox_baboon_400x400/', help='root directory of dataset')
    parser.add_argument('--pai_scale', type=float, default=0.5, help='phase scale')
    parser.add_argument('--exp_name', type=str, default='test_v1', help='experiment name')
    parser.add_argument('--num_epochs', type=int, default=10000, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay_step', nargs='+', type=int, default=[5000, 10000], help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1, help='learning rate decay amount')
    return parser.parse_args()


class Logger:
    filename = None
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            print(text, flush=True)
            log_file.write(text + '\n')

    @staticmethod
    def write_noprint(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')

def read_image(im_path):
    im = imageio.imread(im_path)
    im = np.array(im).astype(np.float32) / 255.
    return im

def write_image(im_path, im):
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    imageio.imwrite(im_path, im)

def load_lensless_dataset(hparams):

    # ctfs 
    CTF_path = os.path.join(hparams.root_dir, 'PropCTFSet.mat')
    CTF_set = loadmat(CTF_path)            
    CTFs = CTF_set['PropCTFSet']                                ## numpy.ndarray, complex128, [h, w, num]
    CTFs = np.transpose(CTFs, [2,0,1]).astype(np.complex64)     ## [num, h, w]
    
    # holograms
    images = []
    for imgid in range(CTFs.shape[0]):
        img_path = os.path.join(hparams.root_dir, 'RawImg_' + str(imgid + 1).zfill(2) + '.png')
        img = read_image(img_path)
        images.append(img[None, :])
    images = np.concatenate(images, 0) ## [num, h, w]

    # measurements_scale
    measurements_scale = 1.
    if os.path.exists(os.path.join(hparams.root_dir, 'measurements_scale.mat')):
        # just for simulation experiments
        measurements_scale = loadmat(os.path.join(hparams.root_dir, 'measurements_scale.mat'))
        measurements_scale = measurements_scale['measurements_scale'][0,0] 
    
    return images, CTFs, measurements_scale


def create_default_log_dir(root_dir, exp_name, cfg=None):
        # log/
        #   exp_name/
        #       image_out/
        #       log.txt
        log_dir_name = exp_name
        log_path = os.path.join(root_dir, 'log', log_dir_name)
        cond_mkdir(log_path)
        
        image_out_path = os.path.join(log_path, 'image_out')
        cond_mkdir(image_out_path)
        return log_path, image_out_path  

def train(hparams):
    cfg = {
        'log_interval': 100,
        'save_interval': 1000,
    }
    
    ## Create Logger
    log_path, image_out_path = create_default_log_dir(hparams.root_dir, hparams.exp_name)
    Logger.filename = log_path + '/log.txt'
    Logger.write('[START] ' + hparams.exp_name)
    
    ## Load training data 
    images, CTFs, measurements_scale = load_lensless_dataset(hparams)
    images = torch.tensor(images).to(device)
    CTFs = torch.tensor(CTFs).to(device)
    Logger.write('Load training data completed.')

    # local variables
    num_images, H, W = images.shape
    pai_scale = hparams.pai_scale
    
    ## Create model
    model = HashSiren_Lessless(
        hash_table_length = H * W, 
        input_dim = 2, 
        hidden_features = 64, 
        hidden_layers = 1, 
        out_features = 1).to(device)
    Logger.write('Create training model completed.')
    
    ## Create optimizer, scheduler
    optimizer = Adam(model.parameters(), lr=hparams.lr)
    scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, gamma=hparams.decay_gamma)
    
    ## Train
    Logger.write('Traing ...')
    N_iters = hparams.num_epochs + 1
    start = 1

    time_cost = 0
    for iter in trange(start, N_iters):
        time_start = time.time()

        # inference
        amp, phs = model()
        amp = amp.view(H, W)
        phs = phs.view(H, W)
        phs = phs * pai_scale * math.pi
        
        # object
        obj = amp * torch.exp(1j * phs)
        
        # batch processing 
        m_objFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(obj))) # [H, W]
        m_objFT_t = torch.mul(m_objFT, CTFs) # [num_images, H, W]   
        diffraction_pred_ = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(m_objFT_t, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
        intensities_pred = torch.pow(diffraction_pred_.real, 2) + torch.pow(diffraction_pred_.imag, 2)        
        intensities_pred = intensities_pred / measurements_scale
        
        loss = img2mse(intensities_pred, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        torch.cuda.synchronize()
        time_cost += time.time() - time_start

        ## rest is logging
        if iter % cfg['log_interval'] == 0 or iter == 1:
            with torch.no_grad():
                psnr = mse2psnr(loss)
            
            def get_learning_rate(optimizer):
                for param_group in optimizer.param_groups:
                    return param_group['lr']
                
            log_str = f"[TRAIN] Iter: {iter} Loss: {loss.item()} PSNR: {psnr.item()} lr: {get_learning_rate(optimizer)} Time: {round(time_cost, 2)}"
            Logger.write(log_str)
            # Logger.write_noprint(log_str)

        # save Amp & Phase image 
        if iter % cfg['save_interval'] == 0:    
            img_path = os.path.join(image_out_path, 'amp_'+ str(iter).zfill(6) + '.png')
            write_image(img_path, amp.cpu().detach().numpy())
            
            img_path = os.path.join(image_out_path, 'phase_'+ str(iter).zfill(6) + '.png')
            write_image(img_path, phs.cpu().detach().numpy() / (pai_scale * math.pi))
                        
    total_time = round(time_cost, 3)
    Logger.write('total_train_time: {}s'.format(total_time))



if __name__ == '__main__':
    hparams = get_opts()
    train(hparams)