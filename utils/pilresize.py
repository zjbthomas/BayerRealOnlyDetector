import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

import torch

from PIL import Image
import numpy as np

def convert2DNADet(img):
    transform = A.Compose([
        PILResize((128, 128)),
        PILResize((512, 512)),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # no need to divide by 255 after normalization
        ToTensorV2()
    ])
    
    img = img.detach()
    img = torch.stack([transform(image = (i.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))['image'] for i in img.unbind()]) # permute: convert back to RGB

    return img

class PILResize(ImageOnlyTransform):
    def __init__(self, size, always_apply=False, p=1.0):
        super(PILResize, self).__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize(self.size)
        img = np.asarray(img_pil)

        return img