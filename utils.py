import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import torch
from torch import optim, nn
import math
import skimage
from skimage import io
from skimage.util import img_as_float
import time
import pdb
import imageio
from opt import *
from sklearn.preprocessing import normalize
import open3d as o3d

def to_numpy(x):
    return x.detach().cpu().numpy()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def loss2psnr(loss):
    return 10*torch.log10(4 /loss)


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def tensor2grid(inputTensor):
    # input (N,2)

    N = inputTensor.shape[0]


    x = inputTensor[:,0].reshape(-1,1).repeat(1,N)
    y = inputTensor[:,1].reshape(1,-1).repeat(N,1)

    grid = torch.cat([x[...,None],y[...,None]],dim = -1)[None,...]

    return grid



def gifMaker():
    orgin = 'gif'         #首先设置图像文件路径
    files = os.listdir(orgin)       #获取图像序列
    files.sort()
    image_list = []
    for file in files:
        path = os.path.join(orgin, file)
        image_list.append(path)
    print(image_list)  
    gif_name = 'N0_L4_F64.gif'  #设置动态图的名字
    duration = 0.2
    create_gif(image_list, gif_name, duration)   #创建动态图




def interp2(x,y,img,xi,yi):
    """
    按照matlab interp2写的加速2d插值
    当矩阵规模很大的时候,numba就快,矩阵规模小,则启动numba有开销
    原图是规整矩阵才能这么做
    """
    # @nb.jit
    def _interpolation(x,y,m,n,mm,nn,zxi,zyi,alpha,beta,img,return_img):
        qsx = int(m/2)
        qsy = int(n/2)
        for i in range(mm):     # 行号
            for j in range(nn):
                zsx,zsy = int(zxi[i,j]+qsx),int(zyi[i,j]+qsy)  # 左上的列坐标和行坐标
                zxx,zxy = int(zxi[i,j]+qsx),int(zyi[i,j]+qsy+1) # 左下的列坐标和行坐标
                ysx,ysy = int(zxi[i,j]+qsx+1),int(zyi[i,j]+qsy)  # 右上的列坐标和行坐标
                yxx,yxy = int(zxi[i,j]+qsx+1),int(zyi[i,j]+qsy+1) # 右下的列坐标和行坐标
                fu0v = img[zsy,zsx]+alpha[i,j]*(img[ysy,ysx]-img[zsy,zsx])
                fu0v1 = img[zxy,zxx]+alpha[i,j]*(img[yxy,yxx]-img[zxy,zxx])
                fu0v0 = fu0v+beta[i,j]*(fu0v1-fu0v)
                return_img[i,j] = fu0v0 
        return return_img
    
    m,n = img.shape  # 原始大矩阵大小
    mm,nn = xi.shape # 小矩阵大小,mm为行,nn为列
    zxi = np.floor(xi)  # 用[u0]表示不超过S的最大整数
    zyi = np.floor(yi)
    alpha = xi-zxi   # u0-[u0]
    beta = yi-zyi
    return_img = np.zeros((mm,nn))
    return_img = _interpolation(x,y,m,n,mm,nn,zxi,zyi,alpha,beta,img,return_img)
    return return_img


def ImageResize(img_path,sidelength,resized_img_path):
    img = io.imread(img_path)
    image_resized = skimage.transform.resize(img,sidelength)
    io.imsave(resized_img_path,image_resized)




def render_raw_image(model,save_path,img_resolution = opt.sidelength,color_type = opt.color_type):
    if color_type == "RGB":
        device = torch.device('cuda')
        H,W = img_resolution
        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
        y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
        xy = torch.cat([x, y],dim = -1).to(device = device) # xy shape [H*W,2]
        rgb = (model(xy).view(H,W,3) + 1) / 2
        img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)

        io.imsave(save_path,img)
    
    elif color_type == "YCbCr":
        device = torch.device("cuda")
        H,W = img_resolution
        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
        y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
        xy = torch.cat([x, y],dim = -1).to(device = device) # xy shape [H*W,2]   
        rgb = (model(xy).view(H,W,3) + 1) / 2
        img = (rgb.detach().cpu().numpy() * 255).astype(np.float64)
        # print(np.max(img),np.min(img))
        img = skimage.color.convert_colorspace(img, "YCbCr", "RGB")
        print(np.max(img),np.min(img))
        print(type(img))
        io.imsave(opt.render_img_path,img)
    
    else:
        return NotImplementedError("This color type is not implemented!")
    

# 新实现的hash iamge渲染函数
def render_hash_image(model,render_img_resolution,save_path):
    device = torch.device('cuda')
    H = render_img_resolution[0]
    W = render_img_resolution[1]
    if opt.grayscale:
        C = 1
    else:
        C = 3
    x_min = min(model.table[0,:]).item()
    x_max = max(model.table[0,:]).item()
    y_min = min(model.table[1,:]).item()
    y_max = max(model.table[1,:]).item()
    [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, W), torch.linspace(y_min,y_max, H))
    x = x.contiguous().view(-1, 1)
    y = y.contiguous().view(-1, 1)

    xy = torch.cat([x,y],dim = -1).to(device = device)

    model.hash_mod = False
    rgb = (model(xy).view(H, W, C) + 1) / 2
    img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    model.hash_mod = True

    io.imsave(save_path,img)
    print(f"range from ({x_min},{y_min}) to ({x_max},{y_max})")


# img_raw [N,3]
# img_const [N,3]
def render_error_image(img_raw,img_const,save_path):
    img_error = abs(to_numpy(img_raw) - to_numpy(img_const))
    img_error = img_error.reshape((opt.sidelength[0],opt.sidelength[1],3))*255
    img_error = img_error.astype(np.uint8)

    io.imsave(save_path,img_error)


def render_volume(model,hash_table_length,render_volume_resolution = 255):
    device = torch.device('cuda')
    # pointCloud = np.zeros(hash_table_length,6) # x,y,z,r,g,b
    hash_table = model.table.detach().cpu().numpy()
    hash_table = normalize(hash_table,axis=0,norm="max") # 归一化
    hash_table *= render_volume_resolution # 将归一化的坐标映射到0-255之间
    xyz = hash_table.astype(int)
    placeHolder = torch.randn(1,2).to(device)
    rgb = (model(placeHolder) + 1) / 2
    rgb = rgb.detach().cpu().numpy().astype(float)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(f"pointCloud/res45_2_64_w1.ply", pcd)

    return pcd

if __name__ == '__main__':
    
    pass
