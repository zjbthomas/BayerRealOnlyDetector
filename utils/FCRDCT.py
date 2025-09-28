import math

from albumentations.core.transforms_interface import ImageOnlyTransform

import torch

import sys
sys.path.append("..")
from utils.dct import *

# zigzag
def zigzag(block_size = 32):
    # initializing the variables
    h = 0
    v = 0

    out = []

    while ((v < block_size) and (h < block_size)):
        if ((h + v) % 2) == 0:                 # going up
            if (v == 0):
                out.append(v * block_size + h)

                if (h == block_size - 1):
                    v += 1
                else:
                    h += 1                        

            elif ((h == block_size - 1) and (v < block_size)):   # if we got to the last column
                out.append(v * block_size + h)
                v += 1

            elif ((v > 0) and (h < block_size - 1)):    # all other cases
                out.append(v * block_size + h)
                v -= 1
                h += 1

        else:                                    # going down
            if ((v == block_size - 1) and (h <= block_size - 1)):       # if we got to the last line
                out.append(v * block_size + h)
                h += 1

            elif (h == 0):                  # if we got to the first column
                out.append(v * block_size + h)

                if (v == block_size - 1):
                    h += 1
                else:
                    v += 1

            elif ((v < block_size - 1) and (h > 0)):     # all other cases
                out.append(v * block_size + h)
                v += 1
                h -= 1

        if ((v == block_size - 1) and (h == block_size - 1)):          # bottom right element
            out.append(v * block_size + h)
            break

    return out

# split and combine
def split(img, block_size = 32):
    # img is of size CHW
    assert img.size(1) % block_size == 0 and img.size(2) % block_size == 0 and img.size(0) == 3

    n_patch_y = img.size(1) // block_size
    n_patch_x = img.size(2) // block_size
    n_patch = n_patch_y * n_patch_x

    reshaped = torch.reshape(img, (3, n_patch_y, block_size, n_patch_x, block_size))
    permuted = torch.permute(reshaped, (0, 1, 3, 2, 4))
    blocks = torch.reshape(permuted, (3, n_patch, block_size, block_size))

    return blocks

def combine(blocks, H, W):
    assert blocks.size(0) == 3
    assert blocks.size(2) == blocks.size(3)

    n_patch = blocks.size(1)
    block_size = blocks.size(2)

    assert H % block_size == 0 and W % block_size == 0

    n_patch_y = H // block_size
    n_patch_x = W // block_size

    assert n_patch_y * n_patch_x == n_patch

    reshaped = torch.reshape(blocks, (3, n_patch_y, n_patch_x, block_size, block_size))
    permuted = torch.permute(reshaped, (0, 1, 3, 2, 4))
    img = torch.reshape(permuted, (3, H, W))

    return img

# dct wrapper
def dctw(blocks, zigzagf = True):
    dd = torch.stack([torch.stack([dct_2d(b, norm='ortho') for b in c.unbind()]) for c in blocks.unbind()])

    # zigzag
    if (zigzagf):
        n_patch = blocks.size(1)
        block_size = blocks.size(2)

        stripes = torch.reshape(dd, (3 * n_patch, block_size * block_size))
        zigzag_indices = zigzag(block_size)
        zigzaged = torch.stack([s[zigzag_indices] for s in stripes.unbind()])

        dd = torch.reshape(zigzaged, (3, n_patch, block_size, block_size))

    return dd

def idctw(dd, zigzagf = True):
    # izigzag
    if (zigzagf):
        n_patch = dd.size(1)
        block_size = dd.size(2)

        zigzag_indices = zigzag(block_size)
        reverse = np.zeros_like(zigzag_indices)
        reverse[zigzag_indices] = np.arange(block_size * block_size)

        stripes = torch.reshape(dd, (3 * n_patch, block_size * block_size))
        izigzaged = torch.stack([s[reverse] for s in stripes.unbind()])
        dd = torch.reshape(izigzaged, (3, n_patch, block_size, block_size))

    blocks = torch.stack([torch.stack([idct_2d(d, norm='ortho') for d in c.unbind()]) for c in dd.unbind()])

    return blocks

# fcr and ifcr
def fcr(blocks):
    assert blocks.size(0) == 3
    assert blocks.size(2) == blocks.size(3)

    n_patch = blocks.size(1)
    block_size = blocks.size(2)
    fcr_size = int(math.sqrt(n_patch))

    reshaped = torch.reshape(blocks, (3, n_patch, block_size * block_size, 1))
    permuted = torch.permute(reshaped, (0, 2, 1, 3))
    fcred = torch.reshape(permuted, (3 * block_size * block_size, fcr_size, fcr_size))

    return fcred

def ifcr(fcred):
    assert fcred.size(0) % 3 == 0

    block_size = int(math.sqrt(fcred.size(0) // 3))
    fcr_size = fcred.size(1)
    n_patch = fcr_size * fcr_size

    reshaped = torch.reshape(fcred, (3, block_size * block_size, n_patch, 1))
    permuted = torch.permute(reshaped, (0, 2, 1, 3))
    blocks = torch.reshape(permuted, (3, n_patch, block_size, block_size))

    return blocks

# rearrange color channels
def rearrangec(fcred):
    assert fcred.size(0) % 3 == 0

    rearrange_indices = np.arange(fcred.size(0)).reshape(3, -1).transpose().reshape(1, -1)[0]
    fcred = fcred[rearrange_indices]

    return fcred

def irearrangec(fcred):
    assert fcred.size(0) % 3 == 0

    rearrange_indices = np.arange(fcred.size(0)).reshape(3, -1).transpose().reshape(1, -1)[0]
    reverse = np.zeros_like(rearrange_indices)
    reverse[rearrange_indices] = np.arange(fcred.size(0))
    fcred = fcred[reverse]

    return fcred

# normalization
def normalize(fcred):
    # range of DCT: [-block_size * block_size, block_size * block_size] when input image in range [0, 1]

    assert fcred.size(0) % 3 == 0
    block_size = int(math.sqrt(fcred.size(0) // 3))

    limit =  2 * block_size

    fcred_norm = (fcred + limit) / limit / 2.0

    # theorectically we do not need the following
    fcred_norm[fcred_norm > 1.0] = 1.0
    fcred_norm[fcred_norm < 0.0] = 0.0

    return fcred_norm

def denormalize(fcred_norm):
    assert fcred_norm.size(0) % 3 == 0
    block_size = int(math.sqrt(fcred_norm.size(0) // 3))

    limit = block_size * block_size

    fcred = fcred_norm * limit * 2.0 - limit
    return fcred

# apply and inverse
def iapply(fcred, H, W, zigzag = True, rearrange = True):
    fcred = denormalize(fcred)

    if (rearrange): fcred= irearrangec(fcred)

    dd = ifcr(fcred)

    blocks = idctw(dd, zigzag)

    img = combine(blocks, H, W) 

    return img

class FCRDCT(ImageOnlyTransform):
    def __init__(self, block_size=32, zigzag = True, rearrange = True, always_apply=False, p=1.0):
        super(FCRDCT, self).__init__(always_apply, p)
        self.block_size = block_size
        self.zigzag = zigzag
        self.rearrange = rearrange

    def apply(self, img, **params):
        #img = img / 255.0

        blocks = split(img, self.block_size)

        dd = dctw(blocks, self.zigzag)

        fcred = fcr(dd)

        if (self.rearrange): fcred= rearrangec(fcred)

        fcred = normalize(fcred)

        return fcred


class DCT(ImageOnlyTransform):
    def __init__(self, convert = False, log = False, factor = 1, always_apply=False, p=1.0):
        super(DCT, self).__init__(always_apply, p)
        self.convert = convert
        self.log = log
        self.factor = factor

    def apply(self, img, **params):
        if (self.convert):
            img = img / 255.0

        dd = torch.stack([dct_2d(c, norm='ortho') for c in img.unbind()])

        if (self.log):
            dd = self.factor * torch.log(torch.abs(dd) + 1) #1e-12)

        # normalization
        #limit = 256 # 2 * 512 / 4
        #dd_norm = (dd + limit) / limit / 2.0

        # theorectically we do not need the following
        #dd_norm[dd_norm > 1.0] = 1.0
        #dd_norm[dd_norm < 0.0] = 0.0

        return dd