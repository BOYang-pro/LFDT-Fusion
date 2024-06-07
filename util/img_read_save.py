import numpy as np
import cv2
import os
#from skimage.io import imsave
from imageio import imsave
import torch.nn.functional as F
def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)
    
def check_image_size(x,window_size):
    _, _, h, w = x.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x
