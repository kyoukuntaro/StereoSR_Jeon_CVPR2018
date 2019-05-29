import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, scale_factor, train_data_size):
        super(Dataset, self).__init__()
        self.N = 800     #ファイル数
        self.L = 33      #入力サイズ
        self.S = scale_factor       #倍率
        self.D = train_data_size   #生成するデータ数
        self.SH = 64     #シフトする画像の数
        np.random.seed(0)
        #ステレオ画像のファイル名にルールあり
        if not 'data' in os.listdir('./'):
            make_input(img_folder,self.N, self.L, self.S, self.D, self.SH)

        self.input_all = np.zeros([self.D,self.SH+3,self.L*self.S,self.L*self.S],dtype=np.uint8)
        self.target_all = np.zeros([self.D,3,self.L*self.S,self.L*self.S],dtype=np.uint8)
        for i in range(self.D):
            input = np.load('./data/input/input{0}.npy'.format(i))
            target = np.load('./data/target/target{0}.npy'.format(i))
            self.input_all[i,:,:,:] = input
            self.target_all[i,:,:,:] = target



    def __len__(self):
        return self.D

    def __getitem__(self, idx):
        return self.input_all[idx,:,:,:].astype('float')/255, self.target_all[idx,:,:,:].astype('float')/255

def make_input(image_folder, N, L, S, D, SH):
    print("Start Making Dataset!!")
    os.mkdir('data')
    if not 'input' in os.listdir('./data'):
        os.mkdir('data/input')
    if not 'target' in os.listdir('./data'):
        os.mkdir('data/target')
    for d in tqdm(range(D)):
        img_id = np.random.randint(800)+1
        img_L = np.array(Image.open(image_folder+str(img_id).zfill(3)+'_L.png'))
        img_R = np.array(Image.open(image_folder+str(img_id).zfill(3)+'_R.png'))
        top = np.random.randint(0,img_L.shape[0]-L*S)
        left = np.random.randint(SH*S, img_L.shape[1]-L*S)
        img_L_HR = img_L[top:top+L*S,left:left+L*S,:]
        img_L_LR = (Image.fromarray(img_L_HR)).resize([L,L],resample=Image.BICUBIC)
        img_L_LR = img_L_LR.convert('YCbCr')
        img_R_HR = img_R[top:top+L*S,left-(SH-1)*S:left+L*S,:]
        img_R_LR = (Image.fromarray(img_R_HR)).resize([L+SH-1,L], resample=Image.BICUBIC)
        img_R_LR = np.array(img_R_LR.resize([img_R_LR.size[0]*S,img_R_LR.size[1]*S]))

        img_Input = np.zeros([L*S,L*S,SH+3],dtype='uint8')
        img_Input[:,:,SH:SH+3] = np.array(img_L_LR.resize([L*S,L*S], resample=Image.BICUBIC))

        for i in range(SH):
            #img_R_LR = img_R_LR.resize([L*S, L*S], resample=Image.BICUBIC)
            if i==0:
                img = img_R_LR[:,-33*S:,:]
            else:
                img = img_R_LR[:,-33*S-i*S:-i*S,:]
            img = np.array((Image.fromarray(img)).convert('YCbCr'))
            img_Input[:,:,i] = img[:,:,0]
        np.save('data/input/input{0}.npy'.format(d),img_Input.transpose([2,0,1]))
        img_L_HR = Image.fromarray(img_L_HR).convert('YCbCr')
        img_L_HR = np.array(img_L_HR)
        np.save('data/target/target{0}.npy'.format(d),img_L_HR.transpose([2,0,1]))
