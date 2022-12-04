import numpy as np
import torch
import skimage
from skimage import io

# input = torch.linspace(1,9,9).view(1,1,3,3)

# grid = torch.rand(1,2,2,2)
# grid[0][0][0][0] = 0
# grid[0][0][0][1] = 0

# res = torch.nn.functional.grid_sample(input,grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True)



# for i in range(1,31):
#     img = io.imread(f"pic/RGB_OR_1200x1200_{i:03d}.png")
#     img[:,:,2] = 0
#     io.imsave(f'pic/rg/RGB_OR_1200x1200_{i:03d}_rg.png',img)

# for i in range(1,31):
#     img = io.imread(f"pic/RGB_OR_1200x1200_{i:03d}.png")
#     img[:,:,2] = (0.5 * img[:,:,0] + 0.5 * img[:,:,1]).astype(np.uint8)

#     io.imsave(f'pic/lin/RGB_OR_1200x1200_{i:03d}_lin.png',img)



# a = np.random.randint([10,10])
# b = a(2:4,3:6)

