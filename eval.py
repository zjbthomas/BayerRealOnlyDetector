import os
import cv2
import sys
import argparse
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import random
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.VAEDetector import *
from models.ResNetDetector import *

def read_paths(iut_paths_file, undersampling, subset):
    distribution = dict()
    n_min = None

    # get parent directory (add it to iut_path later)
    prefix = Path(iut_paths_file).parent.absolute()

    with open(iut_paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split(' ')
            iut_path = parts[0]
            label = int(parts[1])

            # add to distribution
            if (label not in distribution):
                distribution[label] = [os.path.join(prefix,iut_path)]
            else:
                distribution[label].append(os.path.join(prefix,iut_path))

    for label in distribution:
        if (n_min is None or len(distribution[label]) < n_min):
            n_min = len(distribution[label])

    # undersampling
    iut_paths_labels = []

    for label in distribution:
        ll = distribution[label]

        if (undersampling == 'all'):
            for i in ll:
                iut_paths_labels.append((i, label))
        elif (undersampling == 'min'):
            picked = random.sample(ll, n_min)
            
            for p in picked:
                iut_paths_labels.append((p, label))
        else:
            print('Unsupported undersampling method {}!'.format(undersampling))
            sys.exit()

    return iut_paths_labels

def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def save_hist(data, save_path):
    plt.figure()
    plt.hist(data, bins=50)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes

    parser.add_argument("--subset", type=str, help="evaluation on certain subset")
    parser.add_argument("--undersampling", type=str, default='min', choices=['all', 'min'])

    parser.add_argument("--crop_size", type=int, default=128, help="size of cropped images")
    parser.add_argument("--quality", type=int, default=100, help="quality")

    parser.add_argument('--out_dir', type=str, default='out')
    
    parser.add_argument("--model", choices=['vae', 'resnet', 'dvae'], default='vae', help="model type")
    parser.add_argument("--highpass", choices=['yes', 'no'], default='yes', help="Highpass filter")
    parser.add_argument("--latent_dim", type=int, default=2048, help="dimension of latent")

    parser.add_argument('--load_path', type=str, help='path to the pretrained model', default="checkpoints/model.pth")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.model == 'vae' or args.model == 'dvae'):
        model = VAEDetector(args.model == 'vae', args.highpass == 'yes', args.crop_size, args.latent_dim).cuda()
    elif (args.model == 'resnet'):
        model = ResNetDetector(args.highpass == 'yes', args.crop_size, args.latent_dim).cuda()
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

    if args.load_path != None and os.path.exists(args.load_path):
        print('Load pretrained model: {}'.format(args.load_path))
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    else:
        print("%s not exist" % args.load_path)
        sys.exit()

    # no training
    model.eval()

    # read paths for data
    if not os.path.exists(args.iut_paths_file):
        print("%s not exists, quit" % args.iut_paths_file)
        sys.exit()

    if (args.subset):
        print("Evaluation on subset {}".format(args.subset))

    iut_paths_labels = read_paths(args.iut_paths_file, args.undersampling, args.subset)

    print("Eval set size is {}!".format(len(iut_paths_labels)))

    # create/reset output folder
    print("Predicted maps will be saved in :%s" % args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if (args.subset is None):
        os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

    # save paths
    if (args.undersampling == 'min'):
        save_path = os.path.join(args.out_dir, 'paths_file_eval.txt')
        with open(save_path, 'w') as f:
            for (iut_path, label) in iut_paths_labels:
                f.write(iut_path + '\t' + str(label) + '\n')

        print('Eval paths file saved to %s' % (save_path))

    # csv
    if (args.subset is None):
        f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w', newline='')
        writer = csv.writer(f_csv)

        header = ['Image', 'Label', 'RMSE']
        writer.writerow(header)

    # transforms
    quality = int(args.quality)

    print('JPEG compression quality is %d' % (quality))

    transform = A.Compose([
        A.ImageCompression(quality_lower=quality, quality_upper=quality, always_apply=True, p=1.0),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])
        
    y_pred = []
    y_true = []

    det_pred = []
    det_true = []

    ## prediction
    for ix, (iut_path, lab) in enumerate(tqdm(iut_paths_labels, mininterval = 600)):
        try:
            img = cv2.cvtColor(cv2.imread(iut_path), cv2.COLOR_BGR2RGB)
        except:
            print('Failed to load image {}'.format(iut_path))
            continue
        if (img is None):
            print('Failed to load image {}'.format(iut_path))
            continue

        iut = transform(image = img)['image'].to(device)

        # [0, 1] to [-1, 1] (according to VAE design)
        iut = (iut * 2.0) - 1.0

        # reshape into patches
        _, h, w = iut.shape

        if (h < args.crop_size or w < args.crop_size):
            iut = F.interpolate(iut.unsqueeze(0), size=(args.crop_size, args.crop_size), mode='bicubic', align_corners=False).squeeze(0)
            _, h, w = iut.shape

        n_patches = (h // args.crop_size) * (w // args.crop_size)

        iut = iut.unfold(1, args.crop_size, args.crop_size).unfold(2, args.crop_size, args.crop_size) # 3, h // CROP_SIZE, w // CROP_SIZE, CROP_SIZE, CROP_SIZE

        iut = torch.reshape(iut, (3, n_patches, args.crop_size, args.crop_size))

        iut = torch.permute(iut, (1, 0, 2, 3))

        # prediction
        with torch.no_grad():
            if (args.model == 'vae' or args.model == 'dvae'):
                d_outs, s_outs, \
                    oris, recs, \
                    mus_r, mus_g, mus_b, \
                    feas_r, feas_g, feas_b = model(iut, iut)
            elif (args.model == 'resnet'):
                d_outs, s_outs, \
                    feas_r, feas_g, feas_b = model(iut, iut)
            else:
                print("Unrecognized model %s" % args.model)
                sys.exit()
        
        y = torch.cat(tuple(d_outs), dim = 1)
        y = torch.mean(torch.sigmoid(y), dim = 1)

        # avg over batch (0-dim) to get a score for the whole image
        y = torch.mean(y, dim = 0)

        y = not (y >= 0.5)
        y = 1 if y else 0

        y_pred.append(y)
        y_true.append(lab)

        det_pred.append(0 if y == 0 else 1)
        det_true.append(0 if lab == 0 else 1)

        # write to csv
        if (args.subset is None):
            row = [iut_path, y, lab, y == lab]
            writer.writerow(row)

    ## accuracy
    print("acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy_score(y_true, y_pred)))

    ## accuracy (det)
    print("det acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy_score(det_true, det_pred)))

    if (args.subset is None): f_csv.close()