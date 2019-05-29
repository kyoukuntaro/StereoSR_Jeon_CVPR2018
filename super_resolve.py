import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.cuda

from net import StereoSRChrominance, StereoSRLuminance

def main():
    img_dir = 'D:/dataset/sutereo/Test/'
    L = 33
    S = 2
    SH = 64

    device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))

    img_L = np.array(Image.open(img_dir+'008_L.png').convert('YCbCr'),dtype='uint8')
    img_R = np.array(Image.open(img_dir+'008_R.png').convert('YCbCr'),dtype='uint8')

    img_size = int(img_L.shape[0]/(L*S))*int((img_L.shape[1]-SH*S)/(L*S))
    input = np.zeros([img_size,67,L*S,L*S],dtype='float')
    cnt = 0
    for y in range(int(img_L.shape[0]/(L*S))):
        for x in range(int((img_L.shape[1]-SH*S)/(L*S))):
            top = y*L*S
            left = x*L*S+SH*S

            img_L_HR = img_L[top:top + L * S, left:left + L * S, :]
            img_L_LR = (Image.fromarray(img_L_HR)).resize([L, L], resample=Image.BICUBIC)
            img_R_HR = img_R[top:top + L * S, left - (SH - 1) * S:left + L * S, :]
            img_R_LR = (Image.fromarray(img_R_HR)).resize([L + SH - 1, L], resample=Image.BICUBIC)
            img_R_LR = np.array(img_R_LR.resize([img_R_LR.size[0] * S, img_R_LR.size[1] * S]))

            img_Input = np.zeros([L * S, L * S, SH + 3], dtype='uint8')
            img_Input[:, :, SH:SH + 3] = np.array(img_L_LR.resize([L * S, L * S], resample=Image.BICUBIC))

            for i in range(SH):
                # img_R_LR = img_R_LR.resize([L*S, L*S], resample=Image.BICUBIC)
                if i == 0:
                    img = img_R_LR[:, -33 * S:, :]
                else:
                    img = img_R_LR[:, -33 * S - i * S:-i * S, :]
                img = np.array((Image.fromarray(img)).convert('YCbCr'))
                img_Input[:, :, i] = img[:, :, 0]
            input[cnt,:,:,:] = img_Input.transpose([2,0,1])
            cnt += 1
    input = torch.Tensor(input.astype('float') / 255).to(device)

    #Lum = StereoSRLuminance().to(device)
    Lum = torch.load('result/model_lum19.pth').to(device)
    #Chr = StereoSRChrominance().to(device)
    Chr = torch.load('result/model_chr19.pth').to(device)

    lum = Lum(input[:,:SH+1,:,:])
    output = Chr(input[:, 65:, :, :], lum)
    input = np.clip(input[:,64:,:,:].to('cpu').detach().numpy() * 255, 0, 255)
    output = np.clip(output.to('cpu').detach().numpy()*255, 0, 255)

    super_img = np.zeros([3,int(img_L.shape[0]/(L*S))*L*S,int((img_L.shape[1]-SH*S)/(L*S))*L*S], dtype='uint8')
    input_img = np.zeros([3, int(img_L.shape[0] / (L * S)) * L * S, int((img_L.shape[1] - SH * S) / (L * S)) * L * S],dtype='uint8')
    cnt = 0
    for y in range(int(img_L.shape[0]/(L*S))):
        for x in range(int((img_L.shape[1]-SH*S)/(L*S))):
            super_img[:,y*L*S:(y+1)*L*S,x*L*S:(x+1)*L*S] = output[cnt,:,:,:]
            input_img[:, y * L * S:(y + 1) * L * S, x * L * S:(x + 1) * L * S] = input[cnt, :, :, :]
            cnt += 1
    input_img = Image.fromarray(input_img.transpose([1, 2, 0]), 'YCbCr')
    super_img = Image.fromarray(super_img.transpose([1,2,0]), 'YCbCr')
    plt.imshow(input_img)
    plt.show()
    plt.imshow(super_img)
    plt.show()


if __name__ == '__main__':
    main()


