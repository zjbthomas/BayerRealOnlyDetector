from albumentations.core.transforms_interface import ImageOnlyTransform

import torch

import numpy as np

class ChannelNorm(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ChannelNorm, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # standardization
        img = torch.stack([(i - torch.mean(i)) / (torch.std(i) + np.finfo(np.float64).eps) for i in img.unbind()])
        # normalization
        img = torch.stack([(i - torch.min(i)) / (torch.max(i) - torch.min(i) + np.finfo(np.float64).eps) for i in img.unbind()])

        return img