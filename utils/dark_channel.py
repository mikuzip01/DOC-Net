import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.morphology import Erosion2d


class Dark_Channel(object):
    def __init__(self, args=None, kernel_size = 5, device='cpu'):
        self.args = args
        self.kernel_size = kernel_size
        self.step = int(self.kernel_size/2)
        self.device = device
        # self.erosion2d = Erosion2d(1,1,kernel_size, soft_max=False)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def dc_map(self, x):
        bs, _, len_x, len_y = x.shape
        r ,g, b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        minx = torch.where(r<g, r, g)
        minx = torch.where(minx<b, minx, b)

        dc = -self.maxpool(-minx)
        return dc.to(self.device)

if __name__ == '__main__':
    x = torch.ones(2, 3, 10, 10)
    x[:,0,0,0] = 0
    x[:, 0, 9, 9] = 0.5
    net = Dark_Channel(kernel_size=5)
    print(x.shape)
    out = net.dc_map(x)
    print(x)
    print(x.shape)
    print(out)
    print(out.shape)