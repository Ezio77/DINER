import numpy as np
import torch
import skimage
import matplotlib.pyplot as plt
from skimage import data,io
from skimage.color import rgb2gray
import utils
import os
from torch import nn

# # 定义三个向量
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# c = torch.tensor([7, 8, 9])

# # 将三个向量堆叠成一个矩阵，其中每个向量是一行
# mat = torch.stack([a, b, c], dim=0)

# # 对矩阵进行外积运算
# result = torch.einsum('i,j,k->ijk', a, b, c)

# # 输出结果
# print(result)





# n = 10
# width = 3
# hash_table_resolution = [100,100]

# x = nn.parameter.Parameter(1e-4 * (torch.rand(hash_table_resolution[0],width,n)*2 -1),requires_grad = True)
# y = nn.parameter.Parameter(1e-4 * (torch.rand(hash_table_resolution[1],width,n)*2 -1),requires_grad = True)

# ans = torch.zeros((hash_table_resolution[0],hash_table_resolution[1],width))

# for w in range(width):
#     for i in range(n):
#         if i == 0:
#             ans[...,w] = torch.einsum('i,j -> ij',x[:,w,i],y[:,w,i])
#         else:
#             ans[...,w] += torch.einsum('i,j -> ij',x[:,w,i],y[:,w,i])


# breakpoint()

