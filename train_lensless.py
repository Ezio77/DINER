


device = torch.device("cuda")


class Logger:
    filename = None
    
    @staticmethod
    def write(text):
        with open(Logger.filename, 'a') as log_file:
            log_file.write(text + '\n')


def load_lensless_dataset(hparams):
    
    # ctfs 
    CTF_path = os.path.join(hparams.root_dir, 'PropCTFSet.mat')
    CTF_set = loadmat(CTF_path)            
    CTFs = CTF_set['PropCTFSet'] 
    # CTFs = CTFs[:,:,0:hparams.img_num] ## numpy.ndarray, complex128, [h, w, num]
    CTFs = CTFs[:,:,hparams.imgs_selected_list] ## numpy.ndarray, complex128, [h, w, num]
    CTFs = np.transpose(CTFs, [2,0,1]) ## [num, h, w]
    CTFs = np.array(CTFs, dtype=np.complex64)
    
    # holograms
    images = []
    # for imgid in range(hparams.img_num):
    for imgid in hparams.imgs_selected_list:
        img_path = os.path.join(hparams.root_dir, 'RawImg_' + str(imgid + 1).zfill(2) + '.png')
        img = ((cv2.imread(img_path, -1)).astype(np.float32))/255. ## [h, w]
        images.append(img[None, :])
    images = np.concatenate(images, 0) ## [num, h, w]

    return images, CTFs


# def train(hparams):
#     cfg = {
#         'log_interval': 5,
#         'save_amp_phs_interval': 1000,
#     }

#     ## Create Logger
#     log_path, image_out_path = create_default_log_dir(hparams.root_dir, hparams.exp_name)
#     Logger.filename = log_path + '/log.txt'
#     Logger.write('[START] ' + hparams.exp_name)
    
#     ## Load training data 
#     images, CTFs = load_lensless_dataset(hparams)
#     # Move training data to GPU 
#     images = torch.Tensor(images).to(device)
#     CTFs = torch.tensor(CTFs).to(device)
#     print(images.shape, images.dtype)
#     print(CTFs.shape, CTFs.dtype)
#     Logger.write('Load training data completed.')

#     # local variables
#     num_images, H, W = images.shape
#     pai_scale = hparams.pai_scale
#     rawimg_max = hparams.rawimg_max
    
#     ## Create model
#     model = HashSiren_Lessless(imgH=H, imgW=W, hidden_layers=1, hidden_features=64, out_channels=1, init_scale=1.)
#     # model = Siren_Lessless(imgH=H, imgW=W, hidden_layers=1, hidden_features=64, out_channels=1)
#     model = model.to(device)
#     Logger.write('Create training model completed.')

#     ## Create ssim model 
#     ssim_model = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
#     ssim_model = ssim_model.to(device)
#     Logger.write('Create ssim model completed.')
    
#     ## Create optimizer, scheduler
#     optimizer = Adam(list(model.parameters()), lr=hparams.lr, eps=1e-8, weight_decay=hparams.weight_decay)
#     scheduler = get_scheduler(hparams, optimizer)
    
#     ## Train
#     Logger.write('Traing ...')
#     N_iters = hparams.num_epochs + 1
#     # N_iters = 1 + 1
#     start = 1

#     time_cost = 0
#     for iter in trange(start, N_iters):
#         time_start = time.time()

#         model.train()
#         # inference
#         amp, phs = model()

#         amp = amp.view(H, W)
#         phs = phs.view(H, W)
#         phs = phs * pai_scale * math.pi
        
#         # object
#         obj = amp * torch.exp(1j * phs)
        
#         # batch processing -----------------------------
#         m_objFT = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(obj))) # [H, W]
#         m_objFT_t = torch.mul(m_objFT, CTFs) # [num_images, H, W]   
#         diffraction_pred_ = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(m_objFT_t, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
#         intensities_pred = torch.pow(diffraction_pred_.real, 2) + torch.pow(diffraction_pred_.imag, 2)        
#         # -----------------------------

#         '''
#         # prop 
#         diffraction_pred = []
#         for i_image in range(num_images):
#             # ctf i
#             ctf_i = CTFs[i_image]                
#             # diffraction
#             objFT       = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(obj)))
#             objFT_t     = torch.mul(objFT, ctf_i)
#             diffraction = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(objFT_t)))
#             # save 
#             diffraction_pred.append(diffraction)
#         diffraction_pred = torch.stack(diffraction_pred, dim=0) # [num_image, h, w]    
#         intensities_pred = torch.pow(diffraction_pred.real, 2) + torch.pow(diffraction_pred.imag, 2)        
#         '''

#         intensities_pred = intensities_pred / rawimg_max
        
#         # mse_loss = img2mse(intensities_pred, images)
#         # ssim_loss = 1 - ssim_model(intensities_pred.unsqueeze(1), images.unsqueeze(1))
#         # loss = mse_loss * 0.99 + ssim_loss * 0.01
#         loss = img2mse(intensities_pred, images)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         torch.cuda.synchronize()
#         time_cost += time.time() - time_start

#         ## rest is logging
#         # if iter <= 10:
#         #     cfg['log_interval'] = 1
#         # else:
#         #     cfg['log_interval'] = 5
#         if iter % cfg['log_interval'] == 0 or iter == 1:
#             with torch.no_grad():
#                 psnr = mse2psnr(loss)
#                 # psnr = mse2psnr(img2mse(intensities_pred, images))
#                 ssim = ssim_model(intensities_pred.unsqueeze(1), images.unsqueeze(1)) # [num_images, 1, H, W]

#             log_str = f"[TRAIN] Iter: {iter} Loss: {loss.item()} SSIM: {ssim.item()} PSNR: {psnr.item()} lr: {get_learning_rate(optimizer)} Time: {round(time_cost, 2)}"
#             # Logger.write(log_str)
#             Logger.write_noprint(log_str)

#         # save Amp & Phase image 
#         if iter % cfg['save_amp_phs_interval'] == 0:    
#             amp_np = to8b(amp.squeeze(0).cpu().detach().numpy())
#             phs_np = to8b(phs.squeeze(0).cpu().detach().numpy() / (pai_scale * math.pi))
#             raw_np = to8b(intensities_pred[0].cpu().detach().numpy())
#             img_path = os.path.join(image_out_path, 'amp_'+ str(iter).zfill(6) + '.png')
#             cv2.imwrite(img_path, amp_np)
#             img_path = os.path.join(image_out_path, 'phase_'+ str(iter).zfill(6) + '.png')
#             cv2.imwrite(img_path, phs_np)
#             img_path = os.path.join(image_out_path, 'raw_image_0_'+ str(iter).zfill(6) + '.png')
#             cv2.imwrite(img_path, raw_np)
            
#         #     print(pattern)
            
#     total_time = round(time_cost, 3)
#     Logger.write('total_train_time: {}s'.format(total_time))


#     # torch.save(model.state_dict(), './exp_results/hashsiren_weights.pth')

#     # visualize_log_train_psnr_time(Logger.filename)
#     # visualize_log_train_psnr(Logger.filename)
#     # visualize_log_train_loss(Logger.filename)



if __name__ == '__main__':
    print('train lensless')
    