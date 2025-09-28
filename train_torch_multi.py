import os
from datetime import timedelta

import torch
import torch.nn as nn

# multiprocessing
from utils.dist import *

from train_base import *

# constants
SYNC = False
GET_MODULE = True

def main():
    args = parse_args()

    # Init dist
    init_dist('slurm', args.port)

    global_rank, world_size = get_dist_info()

    args, checkpoint_dir = init_env_multi(args, global_rank)

    # models
    A = init_models(args)

    awl = init_awl(args)

    A, awl = load_dicts(args, A, awl)
    
    # Wrap the model
    A = nn.parallel.DistributedDataParallel(A, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
    awl = nn.parallel.DistributedDataParallel(awl, device_ids=[torch.cuda.current_device()])

    optimizer = init_optims(args, world_size, A, awl)

    lr_scheduler = init_schedulers(args, optimizer)

    if (args.cond_state is not None):
        saved_state = torch.load(args.cond_state)

        optimizer.load_state_dict(saved_state['optimizer'])
        lr_scheduler.load_state_dict(saved_state['lr_scheduler'])

        prev_best_val_loss = saved_state['best_val_loss']
        prev_n_last_epochs = saved_state['n_last_epochs']

        # move to device
        device = torch.device('cuda:' + str(torch.cuda.current_device()))

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        prev_best_val_loss = None
        prev_n_last_epochs = 0

    # dataset
    train_sampler, dataloader = init_dataset(args, global_rank, world_size, False, False)
    val_sampler, val_dataloader = init_dataset(args, global_rank, world_size, True, False)
    test_sampler, test_dataloader = init_dataset(args, global_rank, world_size, False, True)

    train(args, global_rank, world_size, SYNC, GET_MODULE,
            checkpoint_dir,
            A,
            train_sampler, dataloader, val_sampler, val_dataloader, test_sampler, test_dataloader,
            optimizer,
            lr_scheduler,
            awl,
            prev_best_val_loss, prev_n_last_epochs)

if __name__ == '__main__':
    main()