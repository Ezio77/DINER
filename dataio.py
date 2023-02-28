import math
import os
import errno
import matplotlib.colors as colors
import matplotlib as mpl
import skimage
import skimage.filters
from skimage import io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from skimage.color import rgb2gray
from pykdtree.kdtree import KDTree
from opt import HyperParameters
import utils

class ImageData(Dataset):
    def __init__(self,
                image_path,
                sidelength,
                grayscale,
                remain_raw_resolution):
        super().__init__()
        self.remain_raw_resolution = remain_raw_resolution
        self.image = io.imread(image_path)
        self.grayscale = grayscale

        if grayscale and len(self.image.shape) == 3:
            self.image = rgb2gray(self.image)

        self.image = self.normalize(self.image,sidelength)
        self.xy,self.rgb = self.img_process(self.image)
        self.shape = self.image.shape

    def normalize(self,image,sidelength):
        if self.remain_raw_resolution:
            transform = Compose([
                                    ToTensor(),
                                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        else:
            transform = Compose([
                                    Resize(sidelength),
                                    ToTensor(),
                                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])

        image = Image.fromarray(image)
        image = transform(image)
        image = image.permute(1, 2, 0)

        return image

    def img_process(self,img):
        H,W,C = img.shape
        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
        x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
        rgb = img.view(-1,C)
        xy = torch.cat([x,y],dim = -1)
        return xy,rgb

    def __len__(self):
        return self.image.shape[0] * self.image.shape[1]

    def __getitem__(self, idx):
        return self.xy, self.rgb

# convert image from RGB to linear
class ImageData_linear(Dataset):
    def __init__(self,
                image_path,
                sidelength,
                grayscale,
                remain_raw_resolution,
                linear = True):
        super().__init__()
        self.remain_raw_resolution = remain_raw_resolution
        self.image = io.imread(image_path)
        self.grayscale = grayscale
        self.linear = linear

        if grayscale and len(self.image.shape) == 3:
            self.image = rgb2gray(self.image)

        self.image = self.normalize(self.image,sidelength)
        self.xy,self.rgb = self.img_process(self.image)
        self.shape = self.image.shape

    def srgb_to_linear(self,img):
        limit = 0.04045
        if isinstance(img,np.ndarray):
            image = np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
        elif isinstance(img,torch.Tensor):
            image = torch.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

        return image

    def linear_to_srgb(self,img):
        limit = 0.0031308
        if isinstance(img,np.ndarray):
            image = np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        elif isinstance(img,torch.Tensor):
            image = torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        return image

    def normalize(self,image,sidelength):
        if self.remain_raw_resolution:
            transform = ToTensor()
        else:
            transform = Compose([Resize(sidelength),ToTensor()])

        image = Image.fromarray(image)
        image = transform(image)
        image = image.permute(1, 2, 0) # range (0,1)
        
        image = self.srgb_to_linear(image) # range (0,1)

        image = (image - 0.5) / 0.5 # range (-1,1)

        return image

    def img_process(self,img):
        H,W,C = img.shape
        [x, y] = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
        x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
        rgb = img.view(-1,C)
        xy = torch.cat([x,y],dim = -1)
        return xy,rgb

    def __len__(self):
        return self.image.shape[0] * self.image.shape[1]

    def __getitem__(self, idx):
        return self.xy, self.rgb

class oneDimData(Dataset):
    def __init__(self,data_length,data_distribution):
        super().__init__()
        self.data_length = data_length
        self.data_distribution = data_distribution
        if data_distribution == 'norm':
            self.data = torch.randn((data_length,1))
        if data_distribution == 'uniform':
            self.data = (torch.rand((data_length,1)) - 0.5) * 2

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.data

class LightFiedData(Dataset):
    def __init__(self,data_path,sidelength):
        super().__init__()
        self.data_path = data_path
        self.sidelength = sidelength
        self.grayscale = False

    def PreProcess(self,image,sidelength):
        image = Image.fromarray(image)
        transform = Compose([
                             Resize(sidelength),
                             ToTensor(),
                             Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
        
        image = transform(image)
        if self.grayscale == False:
            image = image.permute(1, 2, 0)
        return image

    def preprocessing(self,data_path):
        images_path = os.listdir(data_path)
        images_path.sort()
        image_list = []

        for i in range(len(images_path)):
            dir = os.path.join(data_path,images_path[i])
            image = io.imread(dir)
            image = self.PreProcess(image,self.sidelength).numpy()
            image_list.append(image)

        image_list = np.array(image_list)

        n_image = len(images_path)
        a = int(math.sqrt(n_image))
        if a*a != n_image:
            raise ValueError("The number of images is not a square number!")

        H,W = self.sidelength[0],self.sidelength[1]
        C = 3


        image_list = image_list.reshape(a,a,H,W,C)

        image_list = torch.tensor(image_list)
        rgb = image_list.view(-1,C)

        [u,v,x,y] = torch.meshgrid( torch.linspace(0,a-1,a) , torch.linspace(0,a-1,a) , torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        # [u,v] = torch.meshgrid(torch.linspace(0, a - 1, a), torch.linspace(0, a - 1, a))
        y = (y.contiguous().view(-1, 1) / H - 0.5) / 0.5
        x = (x.contiguous().view(-1, 1) / W - 0.5) / 0.5
        u = (u.contiguous().view(-1, 1) / a - 0.5) / 0.5
        v = (v.contiguous().view(-1, 1) / a - 0.5) / 0.5
        xy = torch.cat([u,v,x,y],dim = -1)


        return xy,rgb

    def __len__(self):
        return self.sidelength[0] * self.sidelength[1] * len(os.listdir(self.data_path))

    def __getitem__(self,idx):
        return self.preprocessing(self.data_path)

class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, 
                 pointcloud_path,
                 sidelen = 256,
                 num_samples=30**3,
                 coarse_scale=1e-1,
                 fine_scale=1e-3):
        super().__init__()
        self.sidelen = sidelen
        self.num_samples = num_samples
        self.pointcloud_path = pointcloud_path
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale

        self.load_mesh(pointcloud_path)

    def __len__(self):
        return 10000  # arbitrary

    def load_mesh(self, pointcloud_path):
        pointcloud = np.genfromtxt(pointcloud_path)
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize(self.v)
        self.kd_tree = KDTree(self.v)
        print('loaded pc')

    def normalize(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v[idx]

        points =utils.to_numpy(utils.get_mgrid(sidelen=self.sidelen,dim = 3,centered=True,include_end=True))

        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))


        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]

        return points, sdf

    def __getitem__(self, idx):
        coords, sdf = self.sample_surface()

        return {'coords': torch.from_numpy(coords).float()}, \
               {'sdf': torch.from_numpy(sdf).float()}

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        self.path_to_video = path_to_video
        self.file_list = os.listdir(path_to_video).sort()
        self.num_frames = len(os.listdir(path_to_video))
        self.H,self.W,self.C = skimage.io.imread(os.path.join(path_to_video,self.file_list[0])).shape

    def process(self):
        all_data = np.zeros((self.num_frames,self.H,self.W,self.C))
        for idx,file in enumerate(self.file_list):
            all_data[idx] = skimage.io.imread(os.path.join(self.path_to_video,file))
        
        return all_data.reshape(-1,3)

    def norm(self,data):
        data = (data / 255.0) * 2 - 1
        return data

    def __len__(self):
        return 1

    def __getitem__(self,idx):
        return torch.tensor((self.norm(self.process())),dtype=torch.float)

class VideoData(Dataset):
    def __init__(self,path):
        super().__init__()
        self.path = path
        self.file_list = sorted(os.listdir(path))
        self.num_frames = len(os.listdir(path))
        self.H,self.W,self.C = skimage.io.imread(os.path.join(path,self.file_list[0])).shape

    def get_mgrid(self,sidelen, dim=3, centered=True, include_end=False):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
        if isinstance(sidelen, int):
            sidelen = dim * (sidelen,)

        if include_end:
            denom = [s-1 for s in sidelen]
        else:
            denom = sidelen

        if dim == 2:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / denom[0]
            pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / denom[1]
        elif dim == 3:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[..., 0] = pixel_coords[..., 0] / denom[0]
            pixel_coords[..., 1] = pixel_coords[..., 1] / denom[1]
            pixel_coords[..., 2] = pixel_coords[..., 2] / denom[2]
        else:
            raise NotImplementedError('Not implemented for dim=%d' % dim)

        if centered:
            pixel_coords -= 0.5

        pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
        return pixel_coords


    def norm(self,data):
        data = (data / 255.0) * 2 - 1
        return data

    def process(self):
        all_data = np.zeros((self.num_frames,self.H,self.W,self.C))
        for idx,file in enumerate(self.file_list):
            all_data[idx] = skimage.io.imread(os.path.join(self.path,file))
        
        all_data = self.norm(all_data)
        all_data = torch.from_numpy(all_data).float()

        return all_data.reshape(-1,3)

    def __len__(self):
        return 1

    def __getitem__(self,idx):

        return self.get_mgrid(sidelen=[self.num_frames,self.H,self.W]),self.process()

class VideoIndex(Dataset):
    def __init__(self,N,H,W):
        super().__init__()
        self.N = N
        self.H = H
        self.W = W
        self.total_number = N*H*W
        self.index = torch.linspace(0,self.total_number - 1,self.total_number,dtype=int)

    def __len__(self):
        return self.total_number

    def __getitem__(self,idx):
        return self.index[idx]

class uniform_color_space_3D(Dataset):
    def __init__(self,R_len,G_len,B_len):
        super().__init__()
        self.R_len = R_len
        self.G_len = G_len
        self.B_len = B_len
    
    def __len__(self):
        return self.R_len * self.G_len * self.B_len

    def __getitem__(self, idx):
        # return {"coords":utils.get_mgrid(sidelen=[self.R_len, self.G_len, self.B_len],dim = 3),\
        #         "rgb":utils.get_mgrid(sidelen=[self.R_len, self.G_len, self.B_len],dim = 3)}
        return utils.get_mgrid(sidelen=[self.R_len, self.G_len, self.B_len],dim = 3),\
            utils.get_mgrid(sidelen=[self.R_len, self.G_len, self.B_len],dim = 3)
