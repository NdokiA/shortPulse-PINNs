from pinnPack import pinnUtils
from collections import OrderedDict
import torch
import torch.nn as nn
import math 
    
class LinBlock(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, drop_frac, act = 'tanh'):
        super(LinBlock, self).__init__()
        if type(act) == str:
            act = pinnUtils.find_activation(act)
        self.layer = torch.nn.Sequential(torch.nn.Linear(num_inputs, num_outputs),
                                         act(),
                                         torch.nn.Dropout(drop_frac))
    def forward(self, x):
        return self.layer(x)

class DNN(torch.nn.Module):
    def __init__(self, sizes, drop_frac = 0, act = 'tanh', encode_dim = None):
        super(DNN, self).__init__()
        self.m = 10
        self.fourierLength = encode_dim
        if encode_dim:
            sizes = [2+2*self.m] + sizes[1:] 
        
        layers = []
        self.depth = len(sizes) - 1 
        for i in range(len(sizes) - 2):
            block = LinBlock(sizes[i], sizes[i+1], drop_frac=drop_frac, act=act)
            layers.append((f'block_{i}', block))
        layers.append(('output', torch.nn.Linear(sizes[-2], sizes[-1])))
        
        layerDict = OrderedDict(layers) 
        self.layers = torch.nn.Sequential(layerDict) 
        
    def ff_encod(self, x, t): 
        w = torch.tensor(2.0*math.pi/self.fourierLength, device = x.device, dtype = torch.float32) 
        k = torch.arange(1, self.m+1, device = x.device) 
        H = torch.hstack([t, torch.ones_like(x), torch.cos(k*w*x),
                          torch.sin(k*w*x)])
        return H
    
    def forward(self, x):
        if self.fourierLength:
            x = self.ff_encod(*x)
        elif isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)
        
        return self.layers(x)
        