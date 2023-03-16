import os
import torch
from torch import optim, nn
from model import Siren,MLP,DinerMLP_idx,DinerSiren_idx
from dataio import ImageData
import time
import utils
from sklearn.preprocessing import normalize
from tqdm.autonotebook import tqdm
from opt import HyperParameters
from loss import relative_l2_loss
from torch.utils.data import DataLoader

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
        raise ValueError("steps_til_summary could not be devided by steps!")
    
    # logger
    Logger.filename = os.path.join(log_dir, experiment_name,'log.txt')
    
    device = torch.device('cuda')
    criteon = nn.MSELoss()

    out_features = 3

    Dataset = ImageData(image_path = img_path,
                               sidelength = sidelength,
                               grayscale = grayscale,
                               remain_raw_resolution = remain_raw_resolution)
    model_input,gt = Dataset[0]

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


    elif model_type == 'DinerSiren':
        model = DinerSiren_idx(
                      hash_table_length = hash_table_length,
                      in_features = input_dim,
                      hidden_features = hidden_features,
                      hidden_layers = hidden_layers,
                      out_features = out_features,
                      outermost_linear = True,
                      first_omega_0 = first_omega_0,
                      hidden_omega_0 = hidden_omega_0).to(device = device)

    elif model_type == 'DinerMLP':
        model = DinerMLP_idx(
                        hash_table_length = hash_table_length, 
                        in_features = input_dim, 
                        hidden_features = hidden_features,
                        hidden_layers = hidden_layers,
                        out_features = out_features).to(device = device)

    else:
        raise NotImplementedError("Model_type not supported!")



    optimizer = optim.Adam(lr = lr,params = model.parameters())


    stripe_length = 2000
    stripe_numbers = int(sidelength[0] / stripe_length)
    indexes = torch.linspace(0,stripe_numbers-1,stripe_numbers)

    loader = DataLoader(
        dataset = indexes,
        batch_size = 1,
        shuffle = True,
        num_workers = 8)


    # training process
    with tqdm(total=epochs) as pbar:
        max_psnr = 0
        time_cost = 0
        for epoch in range(epochs):
            time_start = time.time()
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

            torch.cuda.synchronize()
            time_cost += time.time() - time_start
            cur_psnr = utils.loss2psnr(epoch_loss)

            if (epoch+1) % steps_til_summary == 0:
                log_str = f"[TRAIN] Epoch: {epoch+1} Loss: {epoch_loss.item()} PSNR: {cur_psnr} Time: {round(time_cost, 2)}"
                Logger.write(log_str)

            max_psnr = max(max_psnr,cur_psnr)

            if (epoch+1) % 100 == 0:
                tqdm.write("Step %d, Total loss %0.6f, PSNR %0.2f" % (epoch+1, epoch_loss, cur_psnr))

            pbar.update(1)

    print(f"MAX_PSNR : {max_psnr}")

    utils.render_raw_image_batch(model,os.path.join('experiment_results','pluto',f'pluto_recon_method{model_type}_epoch{epochs:05d}_tablelength{input_dim:02d}.png'),[8000,8000])

    recon_psnr = utils.calculate_psnr(os.path.join(log_dir,experiment_name,'recon.png'),img_path)

    print(f"Reconstruction PSNR: {recon_psnr:.2f}")

    return

if __name__ == "__main__":
    
    opt = HyperParameters()
    train_img(opt)

  