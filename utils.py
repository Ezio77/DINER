import numpy as np
import os
import torch
import skimage
from skimage import io
import imageio
from opt import HyperParameters
from sklearn.preprocessing import normalize
import open3d as o3d
import scipy.io
from opt import HyperParameters
from tqdm.autonotebook import tqdm
import torchvision.transforms.functional as F
from PIL import Image

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

def gifMaker(gif_name):
    orgin = 'gif'
    files = os.listdir(orgin)
    files.sort()
    image_list = []
    for file in files:
        path = os.path.join(orgin, file)
        image_list.append(path)
    print(image_list)
    duration = 0.2
    create_gif(image_list, gif_name, duration)


def ImageResize(img_path,sidelength,resized_img_path):
    img = io.imread(img_path)
    image_resized = skimage.transform.resize(img,sidelength)
    io.imsave(resized_img_path,image_resized)

@torch.no_grad()
def render_raw_image_batch(model,save_path,img_resolution):
    H,W = img_resolution
    rgb = torch.zeros((H*W,3))

    stripe_length = 100
    stripe_numbers = int( H / stripe_length)
    
    with torch.no_grad():
        for idx in range(stripe_numbers):
            rgb[int(idx * W * stripe_length) : int((idx+1) * W * stripe_length )] =  model(int(idx * W * stripe_length) , int((idx+1) * W * stripe_length ))

    rgb = (rgb.view(H,W,3) + 1) / 2
    img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    io.imsave(save_path,img)

@torch.no_grad()
def render_raw_image(model,save_path,img_resolution,gray = False,linear = False):
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

    if not linear:
        img =  np.round(rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        img = (linear_to_srgb(rgb.detach().cpu().numpy()) * 255).astype(np.uint8)

    io.imsave(save_path,img)

@torch.no_grad()
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

@torch.no_grad()
def render_hash_3d_volume(model,render_volume_resolution,save_pcd_path,save_data_path):
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

@torch.no_grad()
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

    io.imsave(save_path,img)
    print(f"range from ({x_min},{y_min}) to ({x_max},{y_max})")

@torch.no_grad()
def render_error_image(img_raw,img_const,sidelength,save_path):
    img_error = abs(to_numpy(img_raw) - to_numpy(img_const))
    img_error = img_error.reshape((sidelength[0],sidelength[1],3))*255
    img_error = img_error.astype(np.uint8)
    io.imsave(save_path,img_error)

@torch.no_grad()
def render_volume(model,pc_path,render_volume_resolution = 255):
    device = torch.device('cuda')
    # pointCloud = np.zeros(hash_table_length,6) # x,y,z,r,g,b
    hash_table = model.table.detach().cpu().numpy()
    hash_table = normalize(hash_table,axis=0,norm="max") 
    hash_table *= render_volume_resolution
    xyz = hash_table.astype(int)
    placeHolder = torch.randn(1,2).to(device)
    rgb = (model(placeHolder) + 1) / 2
    rgb = np.round(rgb.detach().cpu().numpy()).astype(float)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(pc_path, pcd)

    return pcd

@torch.no_grad()
def save_data(data,save_path):
    # file : 'mat', 'npy'
    if isinstance(data,torch.Tensor):
        data = to_numpy(data)
    if save_path[-3:] == 'mat':
        scipy.io.savemat(save_path,{"data":data})
    elif save_path[-3:] == 'npy':
        np.save(save_path,data)
    else:
        raise NotImplementedError("File format not supported!")

@torch.no_grad()
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

def srgb_to_linear(img):
    limit = 0.04045
    if isinstance(img,np.ndarray):
        image = np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
    elif isinstance(img,torch.Tensor):
        image = torch.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

    return image

def calculate_psnr(image_path1, image_path2):
    image1 = F.to_tensor(Image.open(image_path1))
    image2 = F.to_tensor(Image.open(image_path2))
    mse = torch.mean((image1 - image2) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def linear_to_srgb(img):
    limit = 0.0031308
    if isinstance(img,np.ndarray):
        image = np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    elif isinstance(img,torch.Tensor):
        image = torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    return image


