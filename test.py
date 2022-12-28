import numpy as np
import torch
import skimage
import matplotlib.pyplot as plt
from skimage import data,io
from skimage.color import rgb2gray
import utils

# 读取原图片
image = io.imread("pic/RGB_OR_1200x1200_029.png")
image = skimage.transform.resize(image, [200,200,3])
io.imsave('pic/resized/029_200*200.png',image)


# image = image.reshape(-1,3)

# # [x, y] = torch.meshgrid(torch.linspace(0,1,W), torch.linspace(0,1,H))
# # x = x.contiguous().view(-1, 1)
# # y = y.contiguous().view(-1, 1)

# utils.save_data(image, "12.14/RGB.mat")
