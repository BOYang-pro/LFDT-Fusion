from __future__ import print_function
import inspect, re
import os
import collections
import torch
import numpy as np
from PIL import Image



def randrot(img):
    mode = np.random.randint(0, 4)
    return rot(img, mode)

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img


def randfilp(img):
    mode = np.random.randint(0, 3)
    return flip(img, mode)


def flip(img, flip_mode):
   
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img


def save_image(image_numpy, image_path):

    image_numpy = Image.fromarray(image_numpy.astype('uint8'))
    image_numpy.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
    
def fuse_cb_cr(Cb1, Cr1, Cb2, Cr2):
    B, C, H, W = Cb1.shape
    Cb = torch.ones((B, C, H, W)).cuda()
    Cr = torch.ones((B, C, H, W)).cuda()

    for b in range(B):
        for k in range(H):
            for n in range(W):
                if abs(Cb1[b, 0, k, n] - 0) < 1e-6 and abs(Cb2[b, 0, k, n] -0) < 1e-6:
                    Cb[b, 0, k, n] = 0
                else:
                    middle_1 = Cb1[b, 0, k, n] * abs(Cb1[b, 0, k, n] - 0) + Cb2[b, 0, k, n] * abs(Cb2[b, 0, k, n] - 0)
                    middle_2 = abs(Cb1[b, 0, k, n] - 0) + abs(Cb2[b, 0, k, n] - 0)
                    Cb[b, 0, k, n] = middle_1 / middle_2

                if abs(Cr1[b, 0, k, n] - 0) < 1e-6 and abs(Cr2[b, 0, k, n] - 0) < 1e-6:
                    Cr[b, 0, k, n] = 0
                else:
                    middle_3 = Cr1[b, 0, k, n] * abs(Cr1[b, 0, k, n] - 0) + Cr2[b, 0, k, n] * abs(Cr2[b, 0, k, n] - 0)
                    middle_4 = abs(Cr1[b, 0, k, n] - 0) + abs(Cr2[b, 0, k, n] - 0)
                    Cr[b, 0, k, n] = middle_3 / middle_4
    return Cb, Cr
