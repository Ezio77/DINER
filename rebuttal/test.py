import numpy as np
import skimage
from skimage import io
import torch


# pic = io.imread("gt/RGB_OR_1200x1200_001.png")
# pic = pic / 255. * 2 - 1

# pic = torch.zeros([3,7,8])
# pic = pic.permute(2,0,1).unsqueeze(0)
# breakpoint()

a = np.random.random([2,3,4])
a = np.mean(a, axis = 0)
print(a.shape)
