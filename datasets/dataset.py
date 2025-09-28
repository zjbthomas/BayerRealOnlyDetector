import os
import random
import numpy as np
import cv2
from pathlib import Path

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class AttributorDataset(Dataset):

    def __init__(self, global_rank, iut_paths_file, id, crop_size, quality, n_c_samples = None, val = False, test = False):

        if (val):
            set_name = 'val'
        elif (test):
            set_name = 'test'
        else:
            set_name = 'train'

        self.crop_size = crop_size

        self.n_c_samples = n_c_samples
        
        self.train = not val and not test
        self.val = val
        self.test = test

        self.save_path = 'cond_paths_file_' + str(id) + '_' + set_name + '.txt'

        self.paths_mls = []

        # get parent directory (add it to iut_path later)
        prefix = Path(iut_paths_file).parent.absolute()

        n_max = 0

        with open(iut_paths_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                parts = l.rstrip().split(' ')
                iut_path = parts[0]
                ml = int(parts[1])

                if (iut_path[0] != '/'):
                    self.paths_mls.append((os.path.join(prefix,iut_path), ml))
                else:
                    self.paths_mls.append((iut_path, ml))

        # ----------
        #  TODO: Transforms for data augmentation (more augmentations should be added)
        # ----------  
        assert quality[0] <= quality[1]

        self.transform_post = A.Compose([
            A.ImageCompression(quality_lower=quality[0], quality_upper=quality[1], always_apply=True, p=1.0)
        ])

        self.transform_core = A.Compose(
            transforms = [
                # mandatory - pre
                A.Normalize(mean=0.0, std=1.0),
                # basic
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # mandatory - post
                ToTensorV2()
            ],
            additional_targets={'extra': 'image'}
        )

        self.transforms_post_val = []
        for q in [quality[0], (quality[0] + quality[1]) // 2, quality[1]]:
            t = A.Compose([
                A.ImageCompression(quality_lower=q, quality_upper=q, always_apply=True, p=1.0)
            ])

            self.transforms_post_val.append(t)

        self.transforms_post_test = []
        self.transforms_post_test.append(A.Compose([
            A.ImageCompression(quality_lower=100, quality_upper=100, always_apply=True, p=1.0)
        ]))

    def __getitem__(self, item):
        # ----------
        # Read images
        # ----------
        (iut_filename, ml) = self.paths_mls[item]

        try:
            iut = cv2.cvtColor(cv2.imread(iut_filename), cv2.COLOR_BGR2RGB)
        except:
            print('Failed to load image {}'.format(iut_filename))
            return None
        
        if (iut is None):
            print('Failed to load image {}'.format(iut_filename))
            return None

        # ----------
        # Apply transform
        # ----------
        iut_clean_set = []
        iut_noisy_set = []
        ml_set = []

        if (self.train):
            # post transforms
            iut_noisy = self.transform_post(image=iut)['image']

            # core transforms
            transformed = self.transform_core(image = iut, extra = iut_noisy)

            iut_clean = transformed['image']
            iut_noisy = transformed['extra']
            
            # [0, 1] to [-1, 1] (according to VAE design)
            iut_clean = (iut_clean * 2.0) - 1.0
            iut_noisy = (iut_noisy * 2.0) - 1.0

            # cropping
            _, h, w = iut_clean.shape

            h_range = h - self.crop_size
            w_range = w - self.crop_size

            h_rand = random.randint(0, h_range // 2)
            w_rand = random.randint(0, w_range // 2)

            iut_clean = iut_clean[:, (h_rand * 2):(h_rand * 2 + self.crop_size), (w_rand * 2):(w_rand * 2 + self.crop_size)]
            iut_noisy = iut_noisy[:, (h_rand * 2):(h_rand * 2 + self.crop_size), (w_rand * 2):(w_rand * 2 + self.crop_size)]

            iut_clean_set.append(iut_clean)
            iut_noisy_set.append(iut_noisy)
            ml_set.append(ml)
        else:
            if self.val:
                ts = self.transforms_post_val
            elif self.test:
                ts = self.transforms_post_test

            for t in ts:
                # post transforms
                iut_noisy = t(image=iut)['image']

                # core transforms
                transformed = self.transform_core(image = iut, extra = iut_noisy)

                iut_clean = transformed['image']
                iut_noisy = transformed['extra']
                
                # [0, 1] to [-1, 1] (according to VAE design)
                iut_clean = (iut_clean * 2.0) - 1.0
                iut_noisy = (iut_noisy * 2.0) - 1.0

                # cropping
                iut_clean = iut_clean[:, :self.crop_size, :self.crop_size]
                iut_noisy = iut_noisy[:, :self.crop_size, :self.crop_size]

                iut_clean_set.append(iut_clean)
                iut_noisy_set.append(iut_noisy)
                ml_set.append(ml)

        # unshuffle is done in model

        return torch.stack(iut_clean_set), torch.stack(iut_noisy_set), torch.LongTensor(ml_set)

    def __len__(self):
        return len(self.paths_mls)
