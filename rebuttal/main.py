import numpy as np
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
import skimage
from skimage import io
import torch
import scipy.io


def readImage(pic_path,gt_path):
    # read the images
    pic = io.imread(pic_path)
    gt = io.imread(gt_path)

    # normalize
    # pic = (pic / 255. * 2 - 1).astype(np.double)
    # gt = (gt / 255. * 2 - 1).astype(np.double)
    pic = (pic / 255. * 2 - 1)
    gt = (gt / 255. * 2 - 1)

    # numpy to tensor
    pic = torch.tensor(pic)
    gt = torch.tensor(gt)

    # reshape
    pic = pic.permute(2,0,1).unsqueeze(0).to(torch.float32)
    gt = gt.permute(2,0,1).unsqueeze(0).to(torch.float32)
    
    return pic,gt


if __name__ == "__main__":

    methods = ['hashMLP','HashSiren_L2H64','peMLP_L2H64','Siren_L2H64','instantNGP_L2H64']

    logger = np.zeros([30,5])

    for pic_id in range(1,31):
        for i,method in enumerate(methods):
            if method != 'instantNGP_L2H64':
                pic, gt = readImage(f"pic/pic{pic_id:02d}_{method}.png",f"gt/RGB_OR_1200x1200_{pic_id:03d}.png")
            else:
                pic, gt = readImage(f"pic/RGB_OR{pic_id:02d}_instantNGP_L2H64.png",f"gt/RGB_OR_1200x1200_{pic_id:03d}.png")
            d = loss_fn_alex(pic, gt)
            logger[pic_id-1][i] = d.detach().numpy()

    
    scipy.io.savemat('results/lpips.mat',{"data":logger})
    np.save('results/lpips.npy',logger)

    scipy.io.savemat('results/lpips_average.mat',{"data":np.mean(logger,axis = 0)})
    np.save('results/lpips_average.npy',np.mean(logger,axis = 0))
