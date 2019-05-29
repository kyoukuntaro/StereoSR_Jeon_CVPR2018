import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def lum_visualize(mat, file_nm):
    img = np.clip(mat.to('cpu').detach().numpy() * 255, 0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    if file_nm == '':
        plt.imshow(img)
        plt.show()
    else:
        if not 'visualize' in os.listdir('./'):
            os.mkdir('visualize')
        img.save('visualize/'+file_nm)

    return
    if not 'visualize' in os.listdir('./'):
        os.mkdir('visualize')
    if not foder_nm in os.listdir('./visualize'):
        os.mkdir('visualize/'+foder_nm)
