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


    epoch_size = 20
    lr = 0.0005


    dataset = Dataset('D:/dataset/sutereo/Train/')
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=96, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L = StereoSRLuminance().to(device, dtype=torch.float)
    criterion = nn.MSELoss()
    optimizer_lum = optim.Adam(L.parameters(), lr=lr, betas=(0.9,0.999))
    print('Training Start')
    if 'model_lum19.pth' in os.listdir('./result') and 1:
        L = torch.load('./result/model_lum19.pth')
    else:
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

            lum_visualize(input[0, 64, :, :], 'input{0}.png'.format(epoch))
            lum_visualize(output[0, 0, :, :], 'output{0}.png'.format(epoch))
            lum_visualize(target[0, 0, :, :], 'target{0}.png'.format(epoch))

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

    print('End!!')
    return 0



if __name__ == '__main__':
    main()