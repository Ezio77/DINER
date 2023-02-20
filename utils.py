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
from opt import HyperParameters
from sklearn.preprocessing import normalize
import open3d as o3d
import scipy.io
from opt import HyperParameters
from tqdm.autonotebook import tqdm
import pcl


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


def render_raw_image_batch(model,save_path,img_resolution):
    device = torch.device('cuda')
    H,W = img_resolution
    # [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    # x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
    # y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
    # xy = torch.cat([x, y],dim = -1).to(device = device) # xy shape [H*W,2]

    rgb = torch.zeros((H*W,3))

    stripe_length = 100
    stripe_numbers = int( H / stripe_length)
    
    with torch.no_grad():
        for idx in range(stripe_numbers):
            rgb[int(idx * W * stripe_length) : int((idx+1) * W * stripe_length )] =  model(int(idx * W * stripe_length) , int((idx+1) * W * stripe_length ))

    rgb = (rgb.view(H,W,3) + 1) / 2
    # rgb = (model(0,int(H*W-1)).view(H,W,3) + 1) / 2

    img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)

    io.imsave(save_path,img)




def render_raw_image(model,save_path,img_resolution,gray = False):
    device = torch.device('cuda')
    H,W = img_resolution
    [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
    y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
    xy = torch.cat([x, y],dim = -1).to(device = device) # xy shape [H*W,2]

    if not gray:
        rgb = (model(xy).view(H,W,3) + 1) / 2
    else:
        rgb = (model(xy).view(H,W,1) + 1) / 2

    img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)

    io.imsave(save_path,img)




def render_hash_1d_line(model,render_line_resolution,save_path):
    device = torch.device('cuda')
    L = render_line_resolution
    C = 3
    x_min,x_max = min(model.table[:,0]).item(),max(model.table[:,0]).item()

    x = torch.linspace(x_min,x_max,steps=render_line_resolution,device=device).view(render_line_resolution,1)

    model.hash_mod = False
    with torch.no_grad():
        rgb = (model(x) + 1) / 2
        rgb = np.round(to_numpy(rgb) * 255).astype(np.uint8)
    model.hash_mod = True

    x = to_numpy(x)

    data = np.concatenate([x,rgb],axis=-1)

    save_data(data,save_path)






def render_hash_3d_volume(model,render_volume_resolution,save_pcd_path,save_data_path):
    opt = HyperParameters()

    # save as point cloud
    device = torch.device('cuda')
    H = render_volume_resolution[0]
    W = render_volume_resolution[1]
    D = render_volume_resolution[2]
    C = 3

    x_min,x_max = min(model.table[:,0]).item(),max(model.table[:,0]).item()
    y_min,y_max = min(model.table[:,1]).item(),max(model.table[:,1]).item() 
    z_min,z_max = min(model.table[:,2]).item(),max(model.table[:,2]).item()

    print(f"range from ({x_min},{y_min},{z_min}) to ({x_max},{y_max},{z_max})")


    [x,y,z] = torch.meshgrid(torch.linspace(x_min, x_max, H), torch.linspace(y_min,y_max, W),\
                    torch.linspace(z_min, z_max, D))

    x = x.contiguous().view(-1, 1)
    y = y.contiguous().view(-1, 1)
    z = z.contiguous().view(-1, 1)


    xyz = torch.cat([x,y,z],dim = -1).to(device = device)

    with torch.no_grad():
        model.hash_mod = False
        rgb = (model(xyz) + 1) / 2
        rgb = rgb.detach().cpu().numpy()
        model.hash_mod = True

    xyz = to_numpy(xyz)

    # points = np.concatenate([x,y,z,rgb],axis = -1)

    pcd = o3d.geometry.PointCloud()
    pcd.points =  o3d.utility.Vector3dVector(xyz)
    pcd.colors =  o3d.utility.Vector3dVector(rgb)

    o3d.io.write_point_cloud(save_pcd_path,pcd)

    rgb = np.round(rgb * 255.).astype(np.uint8)

    ret = np.concatenate([xyz,rgb],axis = -1)

    save_data(ret,save_data_path)


# 新实现的hash iamge渲染函数
def render_hash_image(model,render_img_resolution,save_path):
    device = torch.device('cuda')
    H = render_img_resolution[0]
    W = render_img_resolution[1]
    C = 3

    x_min,x_max = min(model.table[:,0]).item(),max(model.table[:,0]).item()
    y_min,y_max = min(model.table[:,1]).item(),max(model.table[:,1]).item() 

    [x, y] = torch.meshgrid(torch.linspace(x_min, x_max, W), torch.linspace(y_min,y_max, H))
    x = x.contiguous().view(-1, 1)
    y = y.contiguous().view(-1, 1)

    xy = torch.cat([x,y],dim = -1).to(device = device)

    model.hash_mod = False
    rgb = (model(xy).view(H, W, C) + 1) / 2
    img = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    model.hash_mod = True

    # with torch.no_grad():
    #     rgb = (model.net(xy).view(H, W, C) + 1) / 2
    #     img = (to_numpy(rgb)*255).astype(np.uint8)

    io.imsave(save_path,img)
    print(f"range from ({x_min},{y_min}) to ({x_max},{y_max})")

    rgb =  np.round(to_numpy(rgb.view(-1,3)) * 255).astype(np.uint8)
    xy = to_numpy(xy)
    res = np.concatenate([xy,rgb],axis=-1)

    save_data(res,'hash_images/2d_hash_table.mat')



# img_raw [N,3]
# img_const [N,3]
def render_error_image(img_raw,img_const,sidelength,save_path):
    img_error = abs(to_numpy(img_raw) - to_numpy(img_const))
    img_error = img_error.reshape((sidelength[0],sidelength[1],3))*255
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
    rgb = np.round(rgb.detach().cpu().numpy()).astype(float)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(f"pointCloud/res45_2_64_w1.ply", pcd)

    return pcd

def save_data(data,save_path):
    # save_mod : 'mat', 'npy'
    if isinstance(data,torch.Tensor):
        data = to_numpy(data)
    if save_path[-3:] == 'mat':
        scipy.io.savemat(save_path,{"data":data})
    elif save_path[-3:] == 'npy':
        np.save(save_path,data)
    else:
        raise NotImplementedError("File format not supported!")

def render_video_images(model,H,W,N,path):
    with tqdm(total=N) as pbar:
        for i in range(N):
            with torch.no_grad():
                model_output = model(int(i*H*W),int((i+1)*H*W))
            img = to_numpy(model_output)
            img = img.reshape(H,W,-1)
            img_path = f'render_{i:02d}.png'
            img = np.round((img + 1.) / 2. * 255.).astype(np.uint8)
            skimage.io.imsave(os.path.join(path,img_path),img)
            pbar.update(1)

def remove_image_alpha(image_path,save_path):
    img = io.imread(image_path)
    img = img[:,:,:3]
    
    io.imsave(save_path,img)
    



if __name__ == '__main__':
    remove_image_alpha('12.25/img6.png','12.25/res.png')
