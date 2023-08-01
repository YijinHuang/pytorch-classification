import os
import sys
import random
import builtins

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model


def main():
    args = parse_config()
    cfg = load_config(args.config)
    if cfg['base']['HPO']:
        hyperparameter_tuning(cfg)
    cfg = munchify(cfg)

    # print configuration
    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # create folder
    save_path = cfg.base.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg('Save path {} exists and will be overwrited.'.format(save_path), warning=True)
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.base.save_path = new_save_path
            warning = 'Save path {} exists. New save path is set to be {}.'.format(save_path, new_save_path)
            print_msg(warning, warning=True)

    os.makedirs(cfg.base.save_path, exist_ok=True)
    copy_config(args.config, cfg.base.save_path)

    if cfg.dist.distributed:
        print_msg('DISTRIBUTED GPU MODE')
    else:
        print_msg('SINGLE GPU MODE')

    n_gpus = cfg.dist.n_gpus if cfg.dist.n_gpus else torch.cuda.device_count()
    if cfg.dist.distributed:
        cfg.dist.world_size = n_gpus * cfg.dist.nodes
        os.environ['MASTER_ADDR'] = cfg.dist.addr
        os.environ['MASTER_PORT'] = cfg.dist.port
        mp.spawn(worker, nprocs=n_gpus, args=(n_gpus, cfg))
    else:
        worker(0, 1, cfg)


def worker(gpu, n_gpus, cfg):
    if cfg.dist.distributed:
        torch.cuda.set_device(gpu)
        cfg.dist.gpu = gpu
        cfg.dist.rank = cfg.dist.rank * n_gpus + gpu
        dist.init_process_group(
            backend=cfg.dist.backend,
            init_method='env://',
            world_size=cfg.dist.world_size,
            rank=cfg.dist.rank
        )
        torch.distributed.barrier()

        cfg.train.batch_size = int(cfg.train.batch_size / cfg.dist.world_size)
        cfg.train.num_workers = int((cfg.train.num_workers + n_gpus - 1) / n_gpus)

        # suppress printing
        if cfg.dist.gpu != 0 or cfg.dist.rank != 0:
            cfg.base.progress = False
            def print_pass(*args):
                pass
            builtins.print = print_pass

    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed + cfg.dist.rank # different seed for different process if distributed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    log_path = os.path.join(cfg.base.save_path, 'log')
    logger = SummaryWriter(log_path) if is_main(cfg) else None

    # train
    model = generate_model(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )
    if cfg.dist.distributed:
        torch.distributed.barrier()

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.base.save_path, 'final_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def hyperparameter_tuning(cfg):
    import nni
    params = nni.get_next_parameter()
    config_update(cfg, params)
    print_msg('Hyper-parameters optimization mode.', appendixs=params.keys())


if __name__ == '__main__':
    main()
