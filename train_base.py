import argparse
import os
import sys
from shutil import move
from datetime import datetime
from contextlib import nullcontext
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# tensorboard
from torch.utils.tensorboard import SummaryWriter

import timm

from datasets.dataset import *
from utils.losses import *
from models.VAEDetector import *
from models.ResNetDetector import *

# for multiprocessing
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

# for removing damaged images
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def parse_args():
    parser = argparse.ArgumentParser()
    
    ## job
    parser.add_argument("--id", type=int, help="unique ID from Slurm")
    parser.add_argument("--run_name", type=str, default="freq", help="run name")

    parser.add_argument("--seed", type=int, default=3721, help="seed")

    ## multiprocessing
    parser.add_argument('--dist_backend', default='nccl', choices=['gloo', 'nccl'], help='multiprocessing backend')
    parser.add_argument('--port', type=int, default=3721, help='port')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    
    ## dataset
    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes
    parser.add_argument("--val_paths_file", type=str, help="path to the validation set")
    parser.add_argument("--test_paths_file", type=str, help="path to the test set")
    parser.add_argument("--n_c_samples", type=int, help="samples per classes (None for non-controlled)")
    parser.add_argument("--val_n_c_samples", type=int, help="samples per classes for validation set (None for non-controlled)")
    parser.add_argument("--test_n_c_samples", type=int, help="samples per classes for test set (None for non-controlled)")

    parser.add_argument("--quality", type=int, nargs=2, default=[70, 100], help="JPEG quality")
    parser.add_argument("--crop_size", type=int, default=128, help="size of cropped images")

    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")

    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches") # note that batch size will be multiplied by n_classes in the end

    ## model
    parser.add_argument("--model", choices=['vae', 'resnet', 'dvae'], default='vae', help="model type")
    parser.add_argument("--highpass", choices=['yes', 'no'], default='yes', help="Highpass filter")
    parser.add_argument("--latent_dim", type=int, default=2048, help="dimension of latent")

    parser.add_argument('--load_path', type=str, help='pretrained checkpoint for continued training (A)')
    
    ## awl
    parser.add_argument('--load_path_awl', type=str, help='pretrained checkpoint for continued training (awl)')

    ## optimizer and scheduler
    parser.add_argument("--optim", choices=['adam', 'adamw'], default='adamw', help="optimizer")

    parser.add_argument('--factor', type=float, default=0.1, help='factor of decay')

    parser.add_argument('--patience', type=int, default=5, help='numbers of epochs to decay for ReduceLROnPlateau scheduler (None to disable)')

    parser.add_argument('--decay_epoch', type=int, help='numbers of epochs to decay for StepLR scheduler (low priority, None to disable)')

    ## training
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

    parser.add_argument("--cond_epoch", type=int, default=0, help="epoch to start training from")
    
    parser.add_argument("--n_early", type=int, default=1000, help="number of epochs for early stopping")

    parser.add_argument("--cond_state", type=str, help="state file for continued training")

    ## losses
    parser.add_argument("--weight_mode", type=str, default='manual', choices=['manual', 'awl'], help="mode for loss weights")

    parser.add_argument("--con_mode", type=str, default='supcon', choices=['supcon', 'triplet', 'none'], help="mode for contrastive learning")

    parser.add_argument("--lambda_kl", type=float, default=0.0001, help="att loss weight")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="rec loss weight")
    parser.add_argument("--lambda_bce", type=float, default=1.0, help="detection loss weight")
    parser.add_argument("--lambda_con", type=float, default=1.0, help="contrastive loss weight")

    ## log
    parser.add_argument('--save_dir', type=str, default='.', help='dir to save checkpoints and logs')
    parser.add_argument("--log_interval", type=int, default=0, help="interval between saving image samples")
    
    args = parser.parse_args()

    return args

def init_env(args, local_rank, global_rank):
    # for debug only
    #torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(local_rank)

    args, checkpoint_dir = init_env_multi(args, global_rank)

    return args, checkpoint_dir

def init_env_multi(args, global_rank):
    setup_for_distributed(global_rank == 0)

    if (args.id is None):
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    # set random number
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # checkpoint dir
    checkpoint_dir = args.save_dir + "/checkpoints/" + str(args.id) + "_" + args.run_name
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if (args.weight_mode == 'awl'):
        args.lambda_att = 0.0
        args.lambda_con = 0.0
        args.lambda_cls = 0.0

    # finalizing args, print here
    print(args)

    return args, checkpoint_dir

def init_models(args):
    if (args.model == 'vae' or args.model == 'dvae'):
        A = VAEDetector(args.model == 'vae', args.highpass == 'yes', args.crop_size, args.latent_dim).cuda()
    elif (args.model == 'resnet'):
        A = ResNetDetector(args.highpass == 'yes', args.crop_size, args.latent_dim).cuda()
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

    print(A)

    return A

def init_awl(args):
    if (args.con_mode == 'none'):
        awl = AutomaticWeightedLoss(1).cuda()
    else:
        awl = AutomaticWeightedLoss(2).cuda()
    return awl

def init_dataset(args, global_rank, world_size, val = False, test = False):
    assert not(val and test) # val and test cannot be both True

    # return None if no validation set provided
    if (val and args.val_paths_file is None):
        print('No val set!')
        return None, None
    
    # return None if no test set provided
    if (test and args.test_paths_file is None):
        print('No test set!')
        return None, None
    
    # switch between train/val/test
    if (val):
        paths_file = args.val_paths_file
        n_c_samples = args.val_n_c_samples

        set_name = 'Val'

    elif (test):
        paths_file = args.test_paths_file
        n_c_samples = args.test_n_c_samples

        set_name = 'Test'

    else:
        paths_file = args.iut_paths_file
        n_c_samples = args.n_c_samples

        set_name = 'Train'

    dataset = AttributorDataset(global_rank,
                                paths_file,
                                args.id,
                                args.crop_size,
                                args.quality,
                                n_c_samples,
                                val, test)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    local_batch_size = args.batch_size // world_size

    if (not val and not test):
        print('Local batch size is {} ({}//{})!'.format(local_batch_size, args.batch_size, world_size))

    dataloader = DataLoader(dataset=dataset, batch_size=local_batch_size, num_workers=args.workers, pin_memory=True, drop_last=True, sampler=sampler, collate_fn=collate_fn)

    n_drop = len(dataloader.dataset) - len(dataloader) * args.batch_size
    print('{} set size is {} (drop_last {})!'.format(set_name, len(dataloader) * args.batch_size, n_drop))

    return sampler, dataloader

def init_optims(args, world_size,
                A,
                awl):
    
    # Optimizers
    local_lr = args.lr / world_size

    print('Local learning rate is %.3e (%.3e/%d)!' % (local_lr, args.lr, world_size))

    if (args.optim == 'adam'):
        optimizer = torch.optim.Adam([
            {'params':A.parameters(), 'lr':local_lr},
            {'params': awl.parameters(), 'lr': 0.01}
        ])
    elif (args.optim == 'adamw'):
        optimizer = torch.optim.AdamW([
            {'params':A.parameters(), 'lr':local_lr},
            {'params': awl.parameters(), 'lr': 0.01}
        ])
    else:
        print("Unrecognized optimizer %s" % args.optim)
        sys.exit()

    print("Using optimizer {}".format(args.optim))

    return optimizer

def init_schedulers(args, optimizer):
    lr_scheduler = None

    # high priority for ReduceLROnPlateau (validation set required)
    if (args.val_paths_file and args.patience):
        print("Using scheduler ReduceLROnPlateau")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                    factor = args.factor,
                                                    patience = args.patience)
    # low priority StepLR
    elif (args.decay_epoch):
        print("Using scheduler StepLR")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,
                                                    step_size = args.decay_epoch,
                                                    gamma = args.factor)
    
    else:
        print("No scheduler used")

    return lr_scheduler

def load_dicts(args,
                A,
                awl):
    # Load pretrained models
    if args.load_path != None and args.load_path != 'timm':
        print('Load pretrained model: {}'.format(args.load_path))

        A.load_state_dict(torch.load(args.load_path))

    if args.weight_mode == 'awl' and args.load_path_awl != None:
        print('Load pretrained model: {}'.format(args.load_path_awl))

        awl.load_state_dict(torch.load(args.load_path_awl))

    return A, awl

# for saving checkpoints
def save_checkpoints(args, checkpoint_dir, id, epoch, save_best, last_best, get_module,
                    A,
                    awl):
    if (get_module):
        net_A = A.module
        net_awl = awl.module
    else:
        net_A = A
        net_awl = awl.module

    # always remove save last and remove previous
    torch.save(net_A.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch) + '.pth'))

    last_pth = os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch - 1) + '.pth')
    if (os.path.exists(last_pth)):
        os.remove(last_pth)

    # save best
    if (save_best):
        save_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(epoch) + '.pth')
        torch.save(net_A.state_dict(), save_best_pth)

        # last_best_pth = os.path.join(checkpoint_dir, str(id) + "_best_" + str(last_best) + '.pth')
        # if (os.path.exists(last_best_pth)):
        #     os.remove(last_best_pth)

        print('Checkpoint saved to %s.' % (save_best_pth))

    if args.weight_mode == 'awl':
        torch.save(net_A.state_dict(),
                os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch) + '_awl.pth'))

        last_pth = os.path.join(checkpoint_dir, str(id) + "_last_" + str(epoch - 1) + '_awl.pth')
        if (os.path.exists(last_pth)):
            os.remove(last_pth)

def predict_loss_train(args, data, A,
                criterion_BCE, criterion_rec, criterion_L2,
                awl):
    # load data
    in_clean, in_noisy, in_m_ls = data

    in_clean = in_clean.to('cuda', non_blocking=True) # B, C, H, W
    in_noisy = in_noisy.to('cuda', non_blocking=True) # B, N, C, H, W
    in_m_ls = in_m_ls.to('cuda', non_blocking=True) # B, 1

    factor = in_clean.shape[1]

    in_clean = torch.reshape(in_clean, (in_clean.shape[0] * in_clean.shape[1], in_clean.shape[2], in_clean.shape[3], in_clean.shape[4]))
    in_noisy = torch.reshape(in_noisy, (in_noisy.shape[0] * in_noisy.shape[1], in_noisy.shape[2], in_noisy.shape[3], in_noisy.shape[4]))
    in_m_ls = torch.reshape(in_m_ls, (1, in_m_ls.shape[0] * in_m_ls.shape[1])).squeeze(0)

    # calculate loss
    if (args.model == 'vae' or args.model == 'dvae'):
        d_outs, s_outs, \
            oris, recs, \
            mus_r, mus_g, mus_b, \
            feas_r, feas_g, feas_b = A(in_clean, in_noisy)
    elif (args.model == 'resnet'):
        d_outs, s_outs, \
            feas_r, feas_g, feas_b = A(in_clean, in_noisy)
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

    if (args.model == 'vae' or args.model == 'dvae'):
        # VAE loss
        loss_kl = 0.0
        for i in range(len(feas_r)):
            loss_kl += torch.mean(-0.5 * torch.sum(1 + feas_r[i] - mus_r[i] ** 2 - feas_r[i].exp(), dim = 1), dim = 0)
        for i in range(len(feas_g)):
            loss_kl += torch.mean(-0.5 * torch.sum(1 + feas_g[i] - mus_g[i] ** 2 - feas_g[i].exp(), dim = 1), dim = 0)
        for i in range(len(feas_b)):
            loss_kl += torch.mean(-0.5 * torch.sum(1 + feas_b[i] - mus_b[i] ** 2 - feas_b[i].exp(), dim = 1), dim = 0)
        loss_kl *= args.lambda_kl

        loss_rec = 0.0
        for i in range(len(recs)):
            loss_rec += criterion_rec(recs[i], oris[i])
        loss_rec *= args.lambda_rec
    else:
        loss_kl = 0.0
        loss_rec = 0.0

    # detection loss
    factor_d = 1.0 / len(d_outs)
    factor_s = 1.0 / len(s_outs)

    loss_bce = 0.0
    for i in range(len(d_outs)):
        loss_bce += factor_d * criterion_BCE(d_outs[i], torch.ones_like(d_outs[i]))
    for i in range(len(s_outs)):
        loss_bce += factor_s * criterion_BCE(s_outs[i], torch.zeros_like(s_outs[i]))
    loss_bce *= args.lambda_bce
    
    # contrastive loss
    MARGIN = 1.0

    g_i_indices = [0, 3] # diag
    g_o_indices = [1, 2] # anti-diag

    rb1_indices = [0, 1, 2, 3] # diag
    rb2_indices = [4, 5, 6, 7] # anti-diag
    rb3_indices = [8, 9, 10, 11] # anti-diag
    rb4_indices = [12, 13, 14, 15] # diag

    loss_con = 0.0

    # different variances
    for ii in g_i_indices:
        for oi in g_o_indices:
            loss_con += factor_d * torch.clamp(MARGIN - criterion_L2(feas_g[ii], feas_g[oi]), min=0.0)

    for di in rb1_indices + rb4_indices:
        for adi in rb2_indices + rb3_indices:
            loss_con += factor_d * torch.clamp(MARGIN - criterion_L2(feas_r[di], feas_r[adi]), min=0.0)
            loss_con += factor_d * torch.clamp(MARGIN - criterion_L2(feas_b[di], feas_b[adi]), min=0.0)

    # similar variances
    for i1 in range(len(g_i_indices)):
        for i2 in range(i1 + 1, len(g_i_indices)):
            loss_con += factor_s * criterion_L2(feas_g[g_i_indices[i1]], feas_g[g_i_indices[i2]])
    for o1 in range(len(g_o_indices)):
        for o2 in range(o1 + 1, len(g_o_indices)):
            loss_con += factor_s * criterion_L2(feas_g[g_o_indices[o1]], feas_g[g_o_indices[o2]])

    for l in [rb1_indices, rb2_indices, rb3_indices, rb4_indices]:
        for i1 in range(len(l)):
            for i2 in range(i1 + 1, len(l)):
                loss_con += factor_s * criterion_L2(feas_r[l[i1]], feas_r[l[i2]])
                loss_con += factor_s * criterion_L2(feas_b[l[i1]], feas_b[l[i2]])

    loss_con *= args.lambda_con

    loss = loss_kl + loss_rec + loss_bce + loss_con

    # culculate accuracy
    ds = torch.cat(tuple(d_outs), dim = 1)
    ds = torch.mean(torch.sigmoid(ds), dim = 1, keepdim=True)
    n_correct_d = (ds >= 0.5).sum().item()

    ss = torch.cat(tuple(s_outs), dim = 1)
    ss = torch.mean(torch.sigmoid(ss), dim = 1, keepdim=True)
    n_correct_s = (ss < 0.5).sum().item()

    if (args.model == 'vae' or args.model == 'dvae'):
        return loss, loss_kl, loss_rec, loss_bce, loss_con, n_correct_d, n_correct_s
    elif (args.model == 'resnet'):
        return loss, loss_bce, loss_con, n_correct_d, n_correct_s
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

def predict_loss_val(args, data, A,
                criterion_BCE, criterion_rec, criterion_L2,
                awl):
    # load data
    in_clean, in_noisy, in_m_ls = data

    in_clean = in_clean.to('cuda', non_blocking=True) # B, C, H, W
    in_noisy = in_noisy.to('cuda', non_blocking=True) # B, N, C, H, W
    in_m_ls = in_m_ls.to('cuda', non_blocking=True) # B, 1

    factor = in_clean.shape[1]

    in_clean = torch.reshape(in_clean, (in_clean.shape[0] * in_clean.shape[1], in_clean.shape[2], in_clean.shape[3], in_clean.shape[4]))
    in_noisy = torch.reshape(in_noisy, (in_noisy.shape[0] * in_noisy.shape[1], in_noisy.shape[2], in_noisy.shape[3], in_noisy.shape[4]))
    in_m_ls = torch.reshape(in_m_ls, (1, in_m_ls.shape[0] * in_m_ls.shape[1])).squeeze(0)

    # calculate loss
    if (args.model == 'vae' or args.model == 'dvae'):
        d_outs, s_outs, \
            oris, recs, \
            mus_r, mus_g, mus_b, \
            feas_r, feas_g, feas_b = A(in_clean, in_noisy)
    elif (args.model == 'resnet'):
        d_outs, s_outs, \
            feas_r, feas_g, feas_b = A(in_clean, in_noisy)
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()

    # culculate accuracy
    ds = torch.cat(tuple(d_outs), dim = 1)
    ds = torch.mean(torch.sigmoid(ds), dim = 1, keepdim=True)
    n_correct_d = (ds >= 0.5).sum().item() / factor

    ss = torch.cat(tuple(s_outs), dim = 1)
    ss = torch.mean(torch.sigmoid(ss), dim = 1, keepdim=True)
    n_correct_s = (ss < 0.5).sum().item() / factor

    # use -accuracy as loss
    loss = - (n_correct_d + n_correct_s)

    return loss, n_correct_d, n_correct_s

def predict_loss_test(args, data, A):
    # load data
    in_clean, in_noisy, in_m_ls = data

    in_clean = in_clean.to('cuda', non_blocking=True) # B, C, H, W
    in_noisy = in_noisy.to('cuda', non_blocking=True) # B, N, C, H, W
    in_m_ls = in_m_ls.to('cuda', non_blocking=True) # B, 1

    factor = in_clean.shape[1]

    in_clean = torch.reshape(in_clean, (in_clean.shape[0] * in_clean.shape[1], in_clean.shape[2], in_clean.shape[3], in_clean.shape[4]))
    in_noisy = torch.reshape(in_noisy, (in_noisy.shape[0] * in_noisy.shape[1], in_noisy.shape[2], in_noisy.shape[3], in_noisy.shape[4]))
    in_m_ls = torch.reshape(in_m_ls, (1, in_m_ls.shape[0] * in_m_ls.shape[1])).squeeze(0)

    # calculate loss
    if (args.model == 'vae' or args.model == 'dvae'):
        d_outs, _, _, _, _, _, _, _, _, _ = A(in_clean, in_noisy)
    elif (args.model == 'resnet'):
        d_outs, _, _, _, _ = A(in_clean, in_noisy)
    else:
        print("Unrecognized model %s" % args.model)
        sys.exit()  
    
    # culculate accuracy
    ds = torch.cat(tuple(d_outs), dim = 1)
    ds = torch.mean(torch.sigmoid(ds), dim = 1)
    n_correct = ((ds >= 0.5) != in_m_ls).sum().item() / factor

    return n_correct

def reset_optim_lr(optimizer, lr, world_size):
    local_lr = lr / world_size

    for g in optimizer.param_groups:
        g['lr'] = local_lr

    return optimizer

def save_state(checkpoint_dir, id, epoch, last_best,
                optimizer, lr_scheduler,
                best_val_loss, n_last_epochs):
    
    state = {'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
             'best_val_loss': best_val_loss, 'n_last_epochs': n_last_epochs}

    save_best_pth = os.path.join(checkpoint_dir, 'state_' + str(id) + "_best_" + str(epoch) + '.pth')
    torch.save(state, save_best_pth)

    # last_best_pth = os.path.join(checkpoint_dir, 'state_' + str(id) + "_best_" + str(last_best) + '.pth')
    # if (os.path.exists(last_best_pth)):
    #     os.remove(last_best_pth)

    print('State saved to %s with best val loss %.3e @%d.' % (save_best_pth, best_val_loss, n_last_epochs))

def train(args, global_rank, world_size, sync, get_module,
            checkpoint_dir,
            A,
            train_sampler, dataloader, val_sampler, val_dataloader, test_sampler, test_dataloader,
            optimizer,
            lr_scheduler,
            awl,
            prev_best_val_loss, prev_n_last_epochs):

    #criterion_KL = nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()
    criterion_BCE = nn.BCEWithLogitsLoss().cuda()
    criterion_rec = nn.L1Loss().cuda()
    criterion_L2 = nn.MSELoss().cuda()

    print("Using contrastive loss {}".format(args.con_mode))

    # tensorboard
    if global_rank == 0:
        os.makedirs(args.save_dir + "/logs/", exist_ok=True)
        writer = SummaryWriter(args.save_dir + "/logs/" + str(args.id) + "_" + args.run_name)

    last_best = -1
    new_best = -1
    n_last_epochs = prev_n_last_epochs

    lr_last_best = -1

    if (prev_best_val_loss is not None):
        best_val_loss = prev_best_val_loss

        print("Best val loss set to %.3e" % (prev_best_val_loss))
    else:
        best_val_loss = float('inf')

    # for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = args.cond_epoch
    for epoch in range(start_epoch, args.n_epochs):

        train_sampler.set_epoch(epoch)
        
        print('Starting Epoch {}'.format(epoch))

        # loss sum for epoch
        epoch_loss = 0
        epoch_loss_kl = 0
        epoch_loss_rec = 0
        epoch_loss_bce = 0
        epoch_loss_con = 0

        epoch_val_loss = 0

        # for accuracy
        train_n_correct_d = 0
        train_n_correct_s = 0

        # ------------------
        #  Train step
        # ------------------
        with A.join() if get_module else nullcontext(), awl.join() if get_module else nullcontext(): # get_module indicates using DDP         
            for step, data in enumerate(dataloader):
                curr_steps = epoch * len(dataloader) + step

                A.train()
                awl.train()

                if (sync): optimizer.synchronize()
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    if (args.model == 'vae' or args.model == 'dvae'):
                        loss, loss_kl, loss_rec, loss_bce, loss_con, n_correct_d, n_correct_s = predict_loss_train(args, data, A, criterion_BCE, criterion_rec, criterion_L2, awl)
                    elif (args.model == 'resnet'):
                        loss, loss_bce, loss_con, n_correct_d, n_correct_s = predict_loss_train(args, data, A, criterion_BCE, criterion_rec, criterion_L2, awl)
                    else:
                        print("Unrecognized model %s" % args.model)
                        sys.exit()

                # backward prop
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # log losses for epoch
                epoch_loss += loss.item()

                if (args.model == 'vae' or args.model == 'dvae'):
                    epoch_loss_kl += loss_kl.item()
                    epoch_loss_rec += loss_rec.item()
                else:
                    epoch_loss_kl += 0.0
                    epoch_loss_rec += 0.0

                epoch_loss_bce += loss_bce.item()
                epoch_loss_con += loss_con.item()

                # log accuracy
                train_n_correct_d += n_correct_d
                train_n_correct_s += n_correct_s

        # ------------------
        #  Validation
        # ------------------
        if (val_sampler and val_dataloader):

            val_sampler.set_epoch(epoch)

            # for accuracy
            val_n_correct_d = 0
            val_n_correct_s = 0

            A.eval()
            awl.eval()

            for step, data in enumerate(val_dataloader):
                with torch.no_grad():
                    loss, n_correct_d, n_correct_s = predict_loss_val(args, data, A, criterion_BCE, criterion_rec, criterion_L2, awl)

                    epoch_val_loss += loss

                    val_n_correct_d += n_correct_d
                    val_n_correct_s += n_correct_s

            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
                n_last_epochs = 0
                new_best = epoch
            else:
                n_last_epochs += 1
        else:
            new_best = epoch

        # ------------------
        #  Test
        # ------------------
        if (test_sampler and test_dataloader):

            test_sampler.set_epoch(epoch)

            # for accuracy
            test_n_correct = 0

            A.eval()
            awl.eval()

            for step, data in enumerate(test_dataloader):
                with torch.no_grad():
                    n_correct = predict_loss_test(args, data, A)

                    test_n_correct += n_correct

        # ------------------
        #  Step
        # ------------------
        lr_before_step = optimizer.param_groups[0]['lr']

        if (lr_scheduler):
            if (args.val_paths_file and args.patience):
                lr_scheduler.step(epoch_val_loss) # ReduceLROnPlateau
            elif (args.decay_epoch):
                lr_scheduler.step() # StepLR
            else:
                print("Error in scheduler step")
                sys.exit()

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (global_rank == 0):
            epoch_loss_avg = epoch_loss / len(dataloader)
            epoch_loss_kl_avg = epoch_loss_kl / len(dataloader)
            epoch_loss_rec_avg = epoch_loss_rec / len(dataloader)
            epoch_loss_bce_avg = epoch_loss_bce / len(dataloader)
            epoch_loss_con_avg = epoch_loss_con / len(dataloader)

            # accuracy
            local_batch_size = args.batch_size // world_size

            train_acc_d = train_n_correct_d / local_batch_size / len(dataloader)
            train_acc_s = train_n_correct_s / local_batch_size / len(dataloader)

            if (val_dataloader):
                epoch_val_loss_avg = epoch_val_loss / len(val_dataloader)
                best_val_loss_avg = best_val_loss / len(val_dataloader)
                lr_best_val_loss_avg = lr_scheduler.best / len(val_dataloader)

                val_acc_d = val_n_correct_d / local_batch_size / len(val_dataloader)
                val_acc_s = val_n_correct_s / local_batch_size / len(val_dataloader)
            else:
                epoch_val_loss_avg = 0
                best_val_loss_avg = 0
                lr_best_val_loss_avg = 0

                val_acc_d = 0
                val_acc_s = 0

            if (test_dataloader):
                test_acc = test_n_correct / local_batch_size / len(test_dataloader)
            else:
                test_acc = 0

            # global lr (use before-step lr)
            global_lr = lr_before_step * world_size

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                  f"[Epoch {epoch}/{args.n_epochs - 1}]"
                  f"[Loss {epoch_loss_avg:.3e}]"
                  f"[Loss KL {epoch_loss_kl_avg:.3e}]"
                  f"[Loss L2 {epoch_loss_rec_avg:.3e}]"
                  f"[Loss BCE {epoch_loss_bce_avg:.3e}]"
                  f"[Loss Con {epoch_loss_con_avg:.3e}]"
                  f"[Val Loss {epoch_val_loss_avg:.3e} (Best {best_val_loss_avg:.3e} @{n_last_epochs:d}, LR Best {lr_best_val_loss_avg:.3e} @{lr_scheduler.num_bad_epochs:d})]"
                  f"[Acc D {train_acc_d:.3f}]"
                  f"[Val Acc D {val_acc_d:.3f}]"
                  f"[Acc S {train_acc_s:.3f}]"
                  f"[Val Acc S {val_acc_s:.3f}]"
                  f"[Test Acc {test_acc:.3f}]"
                  f"[LR {global_lr:.3e}]")

            writer.add_scalar("Epoch LearningRate", global_lr, epoch)

            writer.add_scalar("Epoch Loss/Train", epoch_loss_avg, epoch)
            writer.add_scalar("Epoch Loss KL/Train", epoch_loss_kl_avg, epoch)
            writer.add_scalar("Epoch Loss L2/Train", epoch_loss_rec_avg, epoch)
            writer.add_scalar("Epoch Loss BCE/Train", epoch_loss_bce_avg, epoch)
            writer.add_scalar("Epoch Loss Con/Train", epoch_loss_con_avg, epoch)

            writer.add_scalar("Epoch Loss/Val", epoch_val_loss_avg, epoch)

            writer.add_scalar("Epoch Acc/Train D", train_acc_d, epoch)
            writer.add_scalar("Epoch Acc/Val D", val_acc_d, epoch)

            writer.add_scalar("Epoch Acc/Train S", train_acc_s, epoch)
            writer.add_scalar("Epoch Acc/Val S", val_acc_s, epoch)
            
            writer.add_scalar("Epoch Acc/Test", test_acc, epoch)

            # save model parameters
            if global_rank == 0:
                save_checkpoints(args, checkpoint_dir, args.id, epoch,
                                 new_best == epoch, last_best,
                                 get_module,
                                 A, awl)

                if (new_best == epoch):
                    last_best = epoch

            # save state
            if global_rank == 0 and lr_scheduler.num_bad_epochs == 0:
                save_state(checkpoint_dir, args.id, epoch, lr_last_best,
                 optimizer, lr_scheduler,
                 best_val_loss, n_last_epochs)

                lr_last_best = epoch
                
        # reset early stopping when learning rate changed
        lr_after_step = optimizer.param_groups[0]['lr']
        if (lr_after_step != lr_before_step):
            print("LR changed to %.3e" % (lr_after_step * world_size))

        # check early_stopping
        if (n_last_epochs > args.n_early or lr_scheduler.num_bad_epochs > args.n_early):
            print('Early stopping')
            break

    print('Finished training')

    if global_rank == 0:
        writer.close()

    pass