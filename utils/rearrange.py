import random

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

import torch

from PIL import Image
import numpy as np

class Crop(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Crop, self).__init__(always_apply, p)

    def apply(self, img, **params):
        C, H, W = img.shape
        assert H == W

        crop_min = H // 8 * 3

        cropped = img[:, crop_min:, crop_min:]

        return cropped