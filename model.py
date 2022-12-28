import torch
from torch import nn,optim
from torch.utils.data import Dataset
import math
import numpy as np
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from math import pi
from numpy import mean, std, diag, dot
import pdb
import utils
from scipy import ndimage
import time
from opt import HyperParameters

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

                # self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) * np.pi / 2 * self.omega_0, 
                #                              np.sqrt(6 / self.in_features) * np.pi / 2 * self.omega_0)


        
    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out

"""
class Siren(nn.Module):
    def __init__(self,hash_mod,hash_table_length, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.0):
        super().__init__()

        self.hash_mod = hash_mod
        if hash_mod:
            self.table = nn.parameter.Parameter(torch.randn((hash_table_length,in_features)),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

                # final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) * np.pi / 2 * hidden_omega_0, 
                #                               np.sqrt(6 / hidden_features) * np.pi / 2 * hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
        else:
            output = self.net(coords)

        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output
"""

class ComplexHashSiren(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length,
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 siren_hidden_features,
                 siren_hidden_layers,
                 outermost_linear=True,
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.hash_mod = hash_mod
        

        # self.weight = nn.parameter.Parameter(0.5 * torch.ones(hash_table_length,3))


        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        # self.table_list = nn.parameter.Parameter(torch.zeros((n_hash,hash_table_length,in_features)),requires_grad = True)
        # self.table_list = nn.parameter.Parameter(torch.randn((n_hash,hash_table_length,in_features)),requires_grad = True)

        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.siren = []
        self.siren.append(SineLayer(in_features, siren_hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(siren_hidden_layers):
            self.siren.append(SineLayer(siren_hidden_features, siren_hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(siren_hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.siren.append(final_linear)
        else:
            self.siren.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.siren = nn.Sequential(*self.siren)

    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table[:,:])

            output += self.siren(coords)
            output = output / 2.0

            # output = output * self.weight + (1 - self.weight) * self.net(coords)

        else:
            output = self.net(coords)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        # output = torch.sigmoid(output)
        return output

class Siren(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class HashSiren(nn.Module):
    def __init__(self,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(self.table)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class HashMLP_interp(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True):
        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        self.hash_table_resolution = hash_table_resolution
    
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)


        self.net = []
        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
                
            self.net.append(final_linear)
        else:
            self.net.append(ReluLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)

    # coords [N,H*W,2]
    def forward(self, coords):
        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,1,2),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().view(-1,self.in_features)
        output = self.net(net_in)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class HashSiren_interp(nn.Module):
    def __init__(self,
                hash_table_resolution, # [H,W]
                in_features,
                hidden_features,
                hidden_layers,
                out_features,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30.0):


        super().__init__()
        self.opt = HyperParameters()
        self.in_features = in_features
        # self.table_list.append(nn.parameter.Parameter(1e-4 * (torch.rand((opt.input_sidelength[0]*opt.input_sidelength[1]*(4**i),2))*2 -1),requires_grad = True))

        self.hash_table_resolution = hash_table_resolution

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_resolution[0]*hash_table_resolution[1],in_features))*2 -1),requires_grad = True)


        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    

    # coords [N,H*W,2]
    def forward(self, coords):
        # coords.reshape(1,self.opt.sidelength[0],self.opt.sidelength[1],2)
        # coords = coords.permute(0,3,1,2)

        grid = coords[None,None,...].to(device="cuda:0") # [1,1,N,2] 输入的坐标

        # temp_input = self.table_list[1,:,:].reshape(1,self.opt.input_sidelength[0],self)

        # net_in [640000,4]
        net_in = nn.functional.grid_sample(self.table.reshape(1,self.hash_table_resolution[0],self.hash_table_resolution[1],self.in_features).permute(0,3,1,2),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().view(-1,self.in_features)

        # net_in = net_in.permute(0,2,3,1).reshape(self.opt.output_sidelength[0],self.opt.output_sidelength[1],2)
        # net_in = net_in.permute(0,2,3,1).squeeze().reshape(-1,2)

        output = self.net(net_in)
        
        output = torch.clamp(output, min = -1.0,max = 1.0) 
        # torch.clamp_(output, min = -1.0,max = 1.0)

        # output = torch.sigmoid(output)
        return output

class ReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return torch.relu(self.linear(input))

class Sigmoid_layer(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear = nn.Linear(in_features = in_features, out_features = out_features)

    def forward(self, input):
        return torch.sigmoid(self.linear(input))

class MLP(nn.Module):
    def __init__(self, in_features, out_features,hidden_layers,hidden_features, bias = True):
        super().__init__()
        self.net = []
        self.net.append(ReluLayer(in_features = in_features,out_features = hidden_features,bias = bias))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(in_features = hidden_features, out_features = hidden_features,bias = bias))

        self.net.append(nn.Linear(hidden_features, out_features))      

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs) #返回一个一维的tensor（张量），这个张量包含了从start到end（包括端点）的等距的steps个数据点。这里返回的就是 0,1,2,3...,N-1
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]
        return torch.cat(out, -1)

class HashMLP(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.hash_mod = True
        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        if self.hash_mod:
            output = self.net(self.table)
        else:
            output = self.net(coords)


        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class HashMLP_m(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        breakpoint()

        self.net = []

        self.net.append(ReluLayer(in_features+1, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        output = self.net(self.table)
        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class HashMLP_idx(nn.Module):
    def __init__(self,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,):
                
        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self,start,end):
        output = self.net(self.table[start:end])
        output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class NeRF(nn.Module):
    def __init__(self,
                hash_mod,
                hash_table_length, 
                in_features, 
                hidden_features,
                hidden_layers,
                out_features,
                N_freqs):

        super().__init__()
        self.N_freqs = N_freqs
        self.PEoutput_features = in_features*(2*self.N_freqs+1)
        # self.table = nn.parameter.Parameter(data=data.to("cuda:0"),requires_grad=True)

        self.hash_mod = hash_mod
        if not hash_mod:
            self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)

        self.net = []
        self.net.append(Embedding(in_features,self.N_freqs))
        self.net.append(ReluLayer(self.PEoutput_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        else:
            output = self.net(coords)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class WaveLayer(nn.Module):
    def __init__(self, in_features, out_features,bias=True,
                 is_first=False,omega_0 = 30):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        # 正态分布 N (0,1)
        self.bias = nn.parameter.Parameter(torch.randn(out_features) / 10,requires_grad = True)

        # self.bias = nn.parameter.Parameter(torch.randn(out_features),requires_grad = True)

        # 均匀分布 U (-1,1)
        # self.bias = nn.parameter.Parameter(torch.rand(out_features) * 2 - 1.0,requires_grad = True)

        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        x1 = self.linear(input)
        x2 = self.linear(input) + self.bias
        return torch.sin(self.omega_0 * x1) * torch.exp(- x2*x2 / 2.0)

class WaveNet(nn.Module):
    def __init__(self,hash_mod,hash_table_length, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,first_omega_0=30, hidden_omega_0=30.0):
        super().__init__()


        self.hash_mod = hash_mod
        if hash_mod:
            self.table = nn.parameter.Parameter(torch.randn((hash_table_length,in_features)),requires_grad = True)

        self.net = []
        self.net.append(WaveLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(WaveLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(WaveLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        else:
            output = self.net(coords)
            output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

"""
Activation Function
"""
def WaveletActivation(x):
    out = torch.sin(30*x) * torch.exp(- x*x / 2.0)
    return out

def SincActivation(x):
    return torch.sin(x) / torch.abs(x)
