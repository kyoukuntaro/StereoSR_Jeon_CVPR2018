import torch
from torch import nn
from torch.nn import functional as F

class StereoSRLuminance(nn.Module):
    def __init__(self):
        super(StereoSRLuminance, self).__init__()
        self.conv1 = nn.Conv2d(65,64,3,1,1)
        self.convrelu16 = nn.Sequential()
        for i in range(16):
            self.convrelu16.add_module('convrelu{}'.format(i+1), ConvRelu())
        self.conv2 = nn.Conv2d(64,1,3,1,1)
        self.weight_initialize()

    def forward(self, x):
        h = self.conv1(x)
        h = self.convrelu16(h)
        h = self.conv2(h)+x[:,[64],:,:]
        return h

    def weight_initialize(self):
        nn.init.xavier_normal_(self.conv1.weight)
        self.convrelu16.apply(self.init_seq_weights)
        nn.init.xavier_normal_(self.conv2.weight)

    def init_seq_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

class StereoSRChrominance(nn.Module):
    def __init__(self):
        super(StereoSRChrominance, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.convrelu14 = nn.Sequential()
        for i in range(14):
            self.convrelu14.add_module('convrelu{}'.format(i + 1), ConvRelu())
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 3, 3, 1, 1)
        self.weight_initialize()

    def forward(self, x, lum):
        h = torch.cat([x, lum], 1)
        h = self.conv1(h)
        h = self.convrelu14(h)
        h = self.conv2(h)
        h = torch.cat([h, x, lum], 1)
        h = self.conv3(h)
        return h

    def weight_initialize(self):
        nn.init.xavier_normal_(self.conv1.weight)
        self.convrelu14.apply(self.init_seq_weights)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)

    def init_seq_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)


class StereoSRInfer(nn.Module):
    def __init__(self):
        super(StereoSRInfer, self).__init__()
        self.conv1 = nn.Conv2d(65,64,3,1,1)
        self.convrelu16 = nn.Sequential()
        for i in range(16):
            self.convrelu16.add_module('convrelu{}'.format(i+1), ConvRelu())
        self.conv2 = nn.Conv2d(64,1,3,1,1)
        self.conv3 = nn.Conv2d(3,64,3,1,1)
        self.convrelu14 = nn.Sequential()
        for i in range(14):
            self.convrelu14.add_module('convrelu{}'.format(i+1), ConvRelu())
        self.conv4 = nn.Conv2d(64,3,3,1,1)
        self.conv5 = nn.Conv2d(6,3,3,1,1)

    def forward(self, xl, xr):
        h = torch.cat([xl[:,[0],:,:], xr],1)
        h = self.conv1(h)
        h = self.convrelu16(h)
        h = self.conv2(h)+xl[:,[0],:,:]
        h = torch.cat([h, xl[:,1:3,:,:]],1)
        h = self.conv3(h)
        h = self.convrelu14(h)
        h = self.conv4(h)
        h = torch.cat([h,xl],1)
        h = self.conv5(h)
        return h

class ConvRelu(nn.Module):
    def __init__(self):
        super(ConvRelu, self).__init__()
        self.conv1 = nn.Conv2d(64,64,3,1,1)
    
    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        return h
