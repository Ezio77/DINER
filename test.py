import numpy as np
import torch
import skimage
import matplotlib.pyplot as plt
from skimage import data,io
from skimage.color import rgb2gray
import utils
import os
from torch import nn

np_file = os.path.join('log','diner_mlp_a6000_time','time.npy')
# np_file = os.path.join('log','diner_siren_a6000_time','time.npy')
# np_file = os.path.join('log','time_diner_mlp','time.npy')
# np_file = os.path.join('log','time_diner_siren','time.npy')


time = np.load(np_file)


time = time[:,1:]


time = np.mean(time,axis=-1)

time_ratio = time / time[1]

diner_mlp_time = time_ratio * 58.48

diner_mlp_time = np.round(diner_mlp_time,2)
print(diner_mlp_time)

save_path = os.path.join("log","time","diner_mlp_a100_time.npy")
utils.save_data(diner_mlp_time,save_path)
