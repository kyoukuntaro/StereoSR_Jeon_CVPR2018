from math import log10
import os
import argparse

import torch
from torch import nn, optim
import torch.cuda

from dataset import Dataset
from net import StereoSRLuminance, StereoSRChrominance
from visualize import lum_visualize

import matplotlib.pyplot as plt
from PIL import Image


def main():

    parser = argparse.ArgumentParser(description='Enhancing the Spatial Resolution of Stereo Images using a Parallax Prior')
    parser.add_argument('--img_folder', type=str, help = 'image folder')
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--epoch', type=int, default=20, help='epoch_size')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
    parser.add_argument('--train_data_size', type=int, default=20000, help='train_data_size')
    opt = parser.parse_args()

    epoch_size = opt.epoch
    lr = opt.lr


    dataset = Dataset(opt.img_folder, opt.scale_factor, opt.train_data_size)
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L = StereoSRLuminance().to(device, dtype=torch.float)
    criterion = nn.MSELoss()
    optimizer_lum = optim.Adam(L.parameters(), lr=lr, betas=(0.9,0.999))
    print('Training Start')
    for epoch in range(epoch_size):
        epoch_loss = 0
        for i, data in enumerate(dataloder, 0):
            input, target = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            optimizer_lum.zero_grad()
            output = L(input[:,:65,:,:])
            loss = criterion(output, target[:,[0],:,:])
            loss.backward()
            optimizer_lum.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloder)
        print(epoch, 10*log10(1/epoch_loss))
        if not 'result' in os.listdir('./'):
            os.mkdir('result')
        torch.save(L, 'result/model_lum{0}.pth'.format(str(epoch).zfill(2)))

        #lum_visualize(input[0, 64, :, :], 'input{0}.png'.format(epoch))
        #lum_visualize(output[0, 0, :, :], 'output{0}.png'.format(epoch))
        #lum_visualize(target[0, 0, :, :], 'target{0}.png'.format(epoch))

    C = StereoSRChrominance().to(device, dtype=torch.float)
    optimizer_chr = optim.Adam(C.parameters(), lr=lr, betas=(0.9,0.999))
    for epoch in range(epoch_size):
        epoch_loss = 0
        for i, data in enumerate(dataloder, 0):
            input, target = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            lum = L(input[:,:65,:,:])
            optimizer_chr.zero_grad()
            output = C(input[:,65:,:,:], lum)
            loss = criterion(output, target)
            loss.backward()
            optimizer_chr.step()
            epoch_loss += loss
        epoch_loss /= len(dataloder)
        print(epoch, 10 * log10(1 / epoch_loss))
        torch.save(C, 'result/model_chr{0}.pth'.format(str(epoch).zfill(2)))

    print('Finish!!')



if __name__ == '__main__':
    main()