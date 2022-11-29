import numpy as np
import torch
import skimage
from skimage import io

# input = torch.linspace(1,9,9).view(1,1,3,3)

# grid = torch.rand(1,2,2,2)
# grid[0][0][0][0] = 0
# grid[0][0][0][1] = 0

# res = torch.nn.functional.grid_sample(input,grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True)

img = io.imread("pic/RGB_OR_1200x1200_001.png")
img[:,:,2] = 0

io.imsave('pic/RGB_OR_1200x1200_001_2_dim.png',img)

breakpoint()