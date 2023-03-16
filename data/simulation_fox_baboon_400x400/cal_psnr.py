import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as SSIM
from cmath import log10

def mkdirs(path):
    if not os.path.exists(path):
            os.makedirs(path)


def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))


def psnr(y_true, y_pred):
    return -10. * np.log10(mse(y_true, y_pred))


# implementation: 'pytorch', 'pl', 'unet'
def calc_psnr(root_dir, exp_name, save=False, implementation='pl'):
    # ssim_model = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)

    img_path = os.path.join(root_dir, 'GT_amp.png')
    gt_amp = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
    img_path = os.path.join(root_dir, 'GT_phs.png')
    gt_phase = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
    
    if implementation == 'pytorch':
        image_dir = os.path.join(root_dir, 'Grid_1x1_' + exp_name, 'image_out')
    elif implementation == 'pl':
        image_dir = os.path.join(root_dir, 'image_out', exp_name)
    elif implementation == 'unet':
        image_dir = os.path.join(root_dir, exp_name, 'image_out')
    elif implementation == 'hashsiren':
        image_dir = os.path.join(root_dir, exp_name, 'image_out')
    elif implementation == 'tcnn':
        image_dir = os.path.join(root_dir, exp_name, 'image_out')

    image_list = os.listdir(image_dir)
    image_list = sorted(image_list)

    iters = []
    psnr_amp = []
    psnr_phs = []
    ssim_amp = []
    ssim_phs = []

    for image in image_list:
        if os.path.splitext(image)[1] == '.png' and image.find('psnr') == -1: # avoid repeating   
            # print(image)     

            if image.find('phase') != -1:
                img_path    = os.path.join(image_dir, image)
                pred_phase  = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
                # pred_phase  = pred_phase / np.max(pred_phase)
                psnr_       = psnr(pred_phase, gt_phase)
                psnr_phs.append(round(float(psnr_), 3))
                
                ssim_ = SSIM(pred_phase, gt_phase)
                # ssim_ = ssim_model(np.expand_dims(np.expand_dims(pred_phase, 0), 0), np.expand_dims(np.expand_dims(gt_phase, 0), 0))
                ssim_phs.append(ssim_)

                if save:
                    img_path = os.path.join(image_dir, os.path.splitext(image)[0] + '_psnr_' + str(round(float(psnr_), 3)) + '_ssim_' + str(round(float(ssim_), 3)) + '.png')
                    pred_phase = pred_phase * 255
                    pred_phase = pred_phase.astype(np.uint8)
                    cv2.imwrite(img_path, pred_phase)

            if image.find('amp') != -1:
                img_path    = os.path.join(image_dir, image)
                pred_amp    = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
                psnr_       = psnr(pred_amp, gt_amp)
                psnr_amp.append(round(float(psnr_), 3))
                
                ssim_ = SSIM(pred_amp, gt_amp)
                # ssim_ = ssim_model(np.expand_dims(np.expand_dims(pred_amp, 0), 0), np.expand_dims(np.expand_dims(gt_amp, 0), 0))
                ssim_amp.append(ssim_)

                if save:
                    img_path = os.path.join(image_dir, os.path.splitext(image)[0] + '_psnr_' + str(round(float(psnr_), 3)) + '_ssim_' + str(round(float(ssim_), 3)) + '.png')
                    pred_amp = pred_amp * 255
                    pred_amp = pred_amp.astype(np.uint8)
                    cv2.imwrite(img_path, pred_amp)

                # iters
                iter = int(image.split('.')[0][-6:])
                iters.append(iter)            
    
    # print(iters)
    # print(psnr_amp)
    return iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs


def calc_psnr_multi(root_dir, list_exp_name, save=False):
    m_iters = []
    m_psnr_amp = []
    m_psnr_phs = []
    m_ssim_amp = []
    m_ssim_phs = []

    for exp_name in list_exp_name:
        ## 
        if exp_name.lower().find('pytorch') != -1:
            iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs = calc_psnr(root_dir, exp_name, save, implementation='pytorch')
        elif exp_name.lower().find('unet') != -1 or exp_name.lower().find('untrained') != -1 or exp_name.lower().find('physen') != -1:
            iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs = calc_psnr(root_dir, exp_name, save, implementation='unet')
        elif exp_name.lower().find('hash') != -1 or exp_name.lower().find('dnf') != -1 or exp_name.lower().find('diner') != -1:
            # print('here')
            iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs = calc_psnr(root_dir, exp_name, save, implementation='hashsiren')
        elif exp_name.lower().find('tcnn') != -1 or exp_name.lower().find('ngp') != -1:
            iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs = calc_psnr(root_dir, exp_name, save, implementation='tcnn')
        else:
            iters, psnr_amp, psnr_phs, ssim_amp, ssim_phs = calc_psnr(root_dir, exp_name, save, implementation='pl')
        m_iters.append(iters)
        m_psnr_amp.append(psnr_amp)
        m_psnr_phs.append(psnr_phs)
        m_ssim_amp.append(ssim_amp)
        m_ssim_phs.append(ssim_phs)

    # print(m_iters)
    return m_iters, m_psnr_amp, m_psnr_phs, m_ssim_amp, m_ssim_phs


def parse_log(file_name):
    iters = []
    times = []
    psnrs = []
    offset_time = 0
    with open(file_name, 'r') as log_file:
        logs = log_file.readlines()
        for log in logs:
            log = log.split(' ')
            if log[0] == '[TRAIN]':
                iter_idx = log.index('Iter:') + 1
                time_idx = log.index('Time:') + 1
                psnr_idx = log.index('PSNR:') + 1
                # ssim_idx = log.index('SSIM:') + 1
                
                iter_ = int(log[iter_idx])
                time_ = round(float(log[time_idx]), 2)
                psnr_ = round(float(log[psnr_idx]), 2)

                iters.append(iter_)
                times.append(time_)
                psnrs.append(psnr_)
            
            if log[0] == '[OFFSET_TIME]':
                offset_time = round(float(log[1]), 2)
    
    new_times = []
    for time in times:
        if time + offset_time >= 0:
            new_times.append(time + offset_time)
        else:
            new_times.append(time)
    times = new_times
    return iters, times, psnrs
    

def parse_log_multi(list_exp_name):
    m_iters = []
    m_times = []
    m_psnrs = []

    for exp_name in list_exp_name:
        if exp_name.lower().find('pytorch') != -1:
            log_file = os.path.join('Grid_1x1_' + exp_name, 'log.txt')
            iters, times, psnrs = parse_log(log_file)
        elif exp_name.lower().find('unet') != -1 or exp_name.lower().find('untrained') != -1 or exp_name.lower().find('physen') != -1:
            log_file = os.path.join(exp_name, 'log', 'log.txt')
            iters, times, psnrs = parse_log(log_file)
        elif exp_name.lower().find('hash') != -1 or exp_name.lower().find('dnf') != -1 or exp_name.lower().find('diner') != -1:
            log_file = os.path.join(exp_name, 'log.txt')
            iters, times, psnrs = parse_log(log_file)
        elif exp_name.lower().find('tcnn') != -1 or exp_name.lower().find('ngp') != -1:
            log_file = os.path.join(exp_name, 'log.txt')
            iters, times, psnrs = parse_log(log_file)

        m_iters.append(iters)
        m_times.append(times)
        m_psnrs.append(psnrs)

    return m_iters, m_times, m_psnrs


## note: [m_val_iters] must be contained in [m_train_iters]
def align_iter(m_val_iters, m_train_iters, m_times):
    m_times_aligned = []
    for val_iters, train_iters, times in zip(m_val_iters, m_train_iters, m_times):
        times_aligned = []
        for v_iter in val_iters:
            idx = train_iters.index(v_iter)
            times_aligned.append(times[idx])
        m_times_aligned.append(times_aligned)
    return m_times_aligned
    

def visualize_val_psnr_ssim_iter(root_dir, list_exp_name, figs_dir):
    m_iters, m_psnr_amp, m_psnr_phs, m_ssim_amp, m_ssim_phs = calc_psnr_multi(root_dir, list_exp_name)
    
    ## psnr
    plt.figure()
    for x, y in zip(m_iters, m_psnr_amp):
        plt.plot(x, y) 
    plt.title('psnr amp')
    plt.xlabel('iters')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_amp.png')

    plt.figure()
    for x, y in zip(m_iters, m_psnr_phs):
        plt.plot(x, y) 
    plt.title('psnr phs')
    plt.xlabel('iters')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_phs.png')

    
    ## ssim
    plt.figure()
    for x, y in zip(m_iters, m_ssim_amp):
        plt.plot(x, y) 
    plt.title('ssim amp')
    plt.xlabel('iters')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_amp.png')

    
    plt.figure()
    for x, y in zip(m_iters, m_ssim_phs):
        plt.plot(x, y) 
    plt.title('ssim phs')
    plt.xlabel('iters')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_phs.png')


def visualize_train_psnr_time(list_exp_name, figs_dir, step=1):
    m_iters, m_times, m_psnrs = parse_log_multi(list_exp_name)
    
    ## train psnr -- iter
    plt.figure()
    for x, y in zip(m_iters, m_psnrs):
        plt.plot(x, y) 
    plt.title('train psnr - iteration')
    plt.xlabel('iters')
    plt.ylabel('train psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/train_psnr_iter.png')

    ## train psnr -- time
    plt.figure()
    for x, y in zip(m_times, m_psnrs):
        plt.plot(x, y) 
    plt.title('train psnr - time')
    plt.xlabel('time/s')
    plt.ylabel('train psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/train_psnr_time.png')
    
    ## train psnr -- log(time)
    plt.figure()
    for x, y in zip(m_times, m_psnrs):
        x = [log10(time).real for time in x] 
        # plt.plot(x, y, marker='.') 
        plt.plot(x, y) 
    plt.xticks(np.arange(5), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    # plt.title('train psnr - time')
    plt.xlabel('Time (sec.)')
    plt.ylabel('PSNR (dB)')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/train_psnr_log10_time.png')
    plt.savefig(figs_dir + '/train_psnr_log10_time.pdf')


def visualize_val_psnr_time(root_dir, list_exp_name, figs_dir):
    m_val_iters, m_psnr_amp, m_psnr_phs, m_ssim_amp, m_ssim_phs = calc_psnr_multi(root_dir, list_exp_name)
    m_train_iters, m_times, m_psnrs = parse_log_multi(list_exp_name)

    m_times_aligned = align_iter(m_val_iters, m_train_iters, m_times)

    ## psnr amp -- time
    plt.figure()
    for x, y in zip(m_times_aligned, m_psnr_amp):
        plt.plot(x, y) 
    plt.title('psnr amp - time')
    plt.xlabel('time/s')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_amp_time.png')

    plt.figure()
    for x, y in zip(m_times_aligned, m_psnr_amp):
        x = [log10(time).real for time in x] # start at 0
        plt.plot(x, y, marker='.') 
    plt.xticks(np.arange(5), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.title('psnr amp - time')
    plt.xlabel('time/s')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_amp_log10_time.png')

    ## psnr phs -- time
    plt.figure()
    for x, y in zip(m_times_aligned, m_psnr_phs):
        plt.plot(x, y) 
    plt.title('psnr phs - time')
    plt.xlabel('time/s')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_phs_time.png')

    plt.figure()
    for x, y in zip(m_times_aligned, m_psnr_phs):
        x = [log10(time).real for time in x] # start at 0
        plt.plot(x, y, marker='.') 
    plt.xticks(np.arange(5), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.title('psnr phs - time')
    plt.xlabel('time/s')
    plt.ylabel('psnr')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/psnr_phs_log10_time.png')

    ## ssim amp -- time
    plt.figure()
    for x, y in zip(m_times_aligned, m_ssim_amp):
        plt.plot(x, y) 
    plt.title('ssim amp - time')
    plt.xlabel('time/s')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_amp_time.png')

    plt.figure()
    for x, y in zip(m_times_aligned, m_ssim_amp):
        x = [log10(time).real for time in x] # start at 0
        plt.plot(x, y, marker='.') 
    plt.xticks(np.arange(5), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.title('ssim amp - time')
    plt.xlabel('time/s')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_amp_log10_time.png')

    ## ssim phs -- time
    plt.figure()
    for x, y in zip(m_times_aligned, m_ssim_phs):
        plt.plot(x, y) 
    plt.title('ssim phs')
    plt.xlabel('time/s')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_phs_time.png')

    plt.figure()
    for x, y in zip(m_times_aligned, m_ssim_phs):
        x = [log10(time).real for time in x] # start at 0
        plt.plot(x, y, marker='.') 
    plt.xticks(np.arange(5), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.title('ssim phs')
    plt.xlabel('time/s')
    plt.ylabel('ssim')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/ssim_phs_log10_time.png')


def visualize_train_psnr_time_3(list_exp_name, figs_dir, step=1):
    m_iters, m_times, m_psnrs = parse_log_multi(list_exp_name)
    
    ## train psnr -- log(time)
    log10_m_time = []
    for x in m_times:
        x = [log10(time).real for time in x] 
        log10_m_time.append(x)
    
    print(np.max(np.array(log10_m_time[0])))
    print(np.max(np.array(log10_m_time[1])))
    print(np.max(np.array(log10_m_time[2])))
    

    
    # hash siren
    hashsiren_time = []
    hashsiren_psnr = []
    for i in range(len(log10_m_time[0])):
        if log10_m_time[0][i] > 0.5: # 10^
            if i % 20 == 0: # x5
                hashsiren_time.append(log10_m_time[0][i])
                hashsiren_psnr.append(m_psnrs[0][i])
        else:
            hashsiren_time.append(log10_m_time[0][i])
            hashsiren_psnr.append(m_psnrs[0][i])      
            
    # dnf
    dnf_time = []
    dnf_psnr = []
    for i in range(len(log10_m_time[1])):
        if log10_m_time[1][i] > 2.7: # 10^
            if i % 50 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        elif log10_m_time[1][i] > 2.3: # 10^
            if i % 20 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        elif log10_m_time[1][i] > 1: # 10^
            if i % 5 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        else:
            dnf_time.append(log10_m_time[1][i])
            dnf_psnr.append(m_psnrs[1][i])   

    # untrained
    untrained_time = []
    untrained_psnr = []
    for i in range(len(log10_m_time[2])):
        if log10_m_time[2][i] > 0.5: # 10^
            if i % 20 == 0: # x5
                untrained_time.append(log10_m_time[2][i])
                untrained_psnr.append(m_psnrs[2][i])
        else:
            untrained_time.append(log10_m_time[2][i])
            untrained_psnr.append(m_psnrs[2][i])               


    plt.figure()
    # for x, y in zip(log10_m_time, m_psnrs):
    #     # plt.plot(x, y, marker='.') 
    #     plt.plot(x, y) 

    _linewidth = 2

    # plt.plot(log10_m_time[0], m_psnrs[0])
    # plt.plot(hashsiren_time, hashsiren_psnr, marker='.')
    plt.plot(hashsiren_time, hashsiren_psnr, linewidth=_linewidth)
    # plt.plot(log10_m_time[1], m_psnrs[1], marker='.')
    # plt.plot(dnf_time, dnf_psnr, marker='.')
    plt.plot(dnf_time, dnf_psnr, linewidth=_linewidth)
    # plt.plot(log10_m_time[2], m_psnrs[2], marker='.')
    # plt.plot(untrained_time, untrained_psnr, marker='.')
    plt.plot(untrained_time, untrained_psnr, linewidth=_linewidth)
    
    plt.grid()
    plt.xticks(np.arange(6)-1, labels=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    # plt.xticks(np.arange(6), labels=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.xlabel('Time (sec.)')
    plt.ylabel('PSNR (dB)')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/train_psnr_log10_time.png')
    plt.savefig(figs_dir + '/train_psnr_log10_time.pdf')



def visualize_train_psnr_time_4(list_exp_name, figs_dir, step=1):
    m_iters, m_times, m_psnrs = parse_log_multi(list_exp_name)
    
    ## train psnr -- log(time)
    log10_m_time = []
    for x in m_times:
        x = [log10(time).real for time in x] 
        log10_m_time.append(x)
    
    print(np.max(np.array(log10_m_time[0])))
    print(np.max(np.array(log10_m_time[1])))
    print(np.max(np.array(log10_m_time[2])))
    print(np.max(np.array(log10_m_time[3])))

    
    # hash siren
    hashsiren_time = []
    hashsiren_psnr = []
    for i in range(len(log10_m_time[0])):
        if log10_m_time[0][i] > 0.5: # 10^
            if i % 20 == 0: # x5
                hashsiren_time.append(log10_m_time[0][i])
                hashsiren_psnr.append(m_psnrs[0][i])
        else:
            hashsiren_time.append(log10_m_time[0][i])
            hashsiren_psnr.append(m_psnrs[0][i])      
            
    # dnf
    dnf_time = []
    dnf_psnr = []
    for i in range(len(log10_m_time[1])):
        if log10_m_time[1][i] > 2.7: # 10^
            if i % 50 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        elif log10_m_time[1][i] > 2.3: # 10^
            if i % 20 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        elif log10_m_time[1][i] > 1: # 10^
            if i % 5 == 0: # x5
                dnf_time.append(log10_m_time[1][i])
                dnf_psnr.append(m_psnrs[1][i])
        else:
            dnf_time.append(log10_m_time[1][i])
            dnf_psnr.append(m_psnrs[1][i])   

    # untrained
    untrained_time = []
    untrained_psnr = []
    for i in range(len(log10_m_time[2])):
        if log10_m_time[2][i] > 0.5: # 10^
            if i % 20 == 0: # x5
                untrained_time.append(log10_m_time[2][i])
                untrained_psnr.append(m_psnrs[2][i])
        else:
            untrained_time.append(log10_m_time[2][i])
            untrained_psnr.append(m_psnrs[2][i])               


    plt.figure()
    # for x, y in zip(log10_m_time, m_psnrs):
    #     # plt.plot(x, y, marker='.') 
    #     plt.plot(x, y) 

    _linewidth = 2

    # plt.plot(log10_m_time[0], m_psnrs[0])
    # plt.plot(hashsiren_time, hashsiren_psnr, marker='.')
    plt.plot(hashsiren_time, hashsiren_psnr, linewidth=_linewidth)
    # plt.plot(log10_m_time[1], m_psnrs[1], marker='.')
    # plt.plot(dnf_time, dnf_psnr, marker='.')
    plt.plot(dnf_time, dnf_psnr, linewidth=_linewidth)
    # plt.plot(log10_m_time[2], m_psnrs[2], marker='.')
    # plt.plot(untrained_time, untrained_psnr, marker='.')
    plt.plot(untrained_time, untrained_psnr, linewidth=_linewidth)
    plt.plot(log10_m_time[3], m_psnrs[3], linewidth=_linewidth)
    
    plt.grid()
    plt.xticks(np.arange(6)-1, labels=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    # plt.xticks(np.arange(6), labels=['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    plt.xlabel('Time (sec.)')
    plt.ylabel('PSNR (dB)')
    plt.legend(list_exp_name)
    plt.savefig(figs_dir + '/train_psnr_log10_time.png')
    plt.savefig(figs_dir + '/train_psnr_log10_time.pdf')


# conclusion: 2x64 is better than 3x64
if __name__ == '__main__':

    root_dir = './'
    figs_dir = './__result_figures__'
    mkdirs(figs_dir)
    list_exp_name = [
        # 'Hash+SIREN', # batchprocess_A100_hashsiren_hiddenlayer_1
        # 'DNF', # dnf_stochastic_v1
        # 'Untrained', # unet_sigmoid
        
        'DINER',        # 'batchprocess_A100_hashsiren_denselog05_v4',
        'DNF',               # 'batchprocess_A100_dnf_denselog05_v4',
        'PhysenNet',         # 'A100_unet_sigmoid_denselog5_v1',
        # 
        # 'InstantNGP',        # 'batchprocess_A100_tcnn_denselog05_v6',
        # 'batchprocess_A100_dnf_denselog05_v1',
        # 'batchprocess_A100_dnf_denselog05_v2',
        
        
        # 'batchprocess_A100_hashsiren_denselog10_v2',
        # 'batchprocess_A100_hashsiren_denselog05_v1',
        # 'batchprocess_A100_hashsiren_denselog05_v2',
        # 'batchprocess_A100_hashsiren_denselog05_v4',
        # 'batchprocess_A100_hashsiren_denselog_v3',
        
        # 'batchprocess_A100_hashsirenPlusSiren_v4',
        # 'batchprocess_A100_hashsirenPlusSiren_v5',
        # 'batchprocess_A100_hashsirenPlusMLP_v1',
    ]
    visualize_val_psnr_ssim_iter(root_dir, list_exp_name, figs_dir)
    visualize_val_psnr_time(root_dir, list_exp_name, figs_dir)
    # visualize_train_psnr_time(list_exp_name, figs_dir)
    visualize_train_psnr_time_3(list_exp_name, figs_dir)
        
    print('Finished!')
 



## better: hashsiren_phs_4_decay_0.5_milestones_[4000_6000]_sigmoid_v1