import os
import sys

import yaml
import torch
import shutil
import argparse
import torch.nn as nn

from tqdm import tqdm
from operator import getitem
from functools import reduce
from torch.utils.data import DataLoader

from utils.const import regression_loss


def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    args = parser.parse_args()
    return args


def load_config(path):
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg


def copy_config(src, dst):
    if os.path.split(src)[0] != dst:
        shutil.copy(src, dst)


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    return mean, std


def save_weights(model, save_path):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)


def print_msg(msg, appendixs=[], warning=False):
    color = '\033[93m'
    end = '\033[0m'
    print_fn = lambda x: print(color + x + end) if warning else print

    max_len = len(max([msg, *appendixs], key=len))
    max_len = min(max_len, get_terminal_col())
    print_fn('=' * max_len)
    print_fn(msg)
    for appendix in appendixs:
        print_fn(appendix)
    print_fn('=' * max_len)


def print_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        _print_config(config)
        print('=' * (len(name) + 10))
        print()


def _print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            _print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


# unnormalize image for visualization
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# convert labels to onehot
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]


# convert type of target according to criterion
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:
        y = y.long()
    elif criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


# convert output dimension of network according to criterion
def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features


def exit_with_error(msg):
    print(msg)
    sys.exit(1)


def is_main(cfg):
    return (not cfg.dist.distributed) or cfg.dist.rank == 0


def config_check(cfg):
    warning = None
    if cfg.scheduler_args.cosine.T_max != cfg.train.epochs & cfg.config_check.cosine_decay_epochs:
        cfg.scheduler_args.cosine.T_max = cfg.train.epochs - cfg.train.warmup_epochs
        warning = 'The max epoch of cosine decay scheduler is set to be the number of training epochs after warmup, because they are commonly the same. ' \
                  'If you want to set the max epoch of cosine decay scheduler manually, please set config_check.cosine_decay_epochs in the config file to False.'
        print_msg(warning, warning=True)


def config_update(cfg, params):
    keys = get_all_keys(cfg)
    name2key = {key[-1]: key for key in keys}
    names = list(name2key.keys())
    for key, value in params.items():
        if key not in names:
            raise KeyError('Invalid key: {}'.format(key))
        if names.count(key) > 1:
            raise KeyError('Key {} appears more than once, can not be updated'.format(key))
        ks = name2key[key]
        get_by_path(cfg, ks[:-1])[ks[-1]] = value


def get_by_path(d, path):
    return reduce(getitem, path, d)


def get_all_keys(cfg):
    keys = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            keys += [[key] + subkey for subkey in get_all_keys(value)]
        else:
            keys.append([key])
    return keys


def get_terminal_col():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def add_path_suffix(path):
    suffix = 0
    new_path = path
    while os.path.exists(new_path):
        suffix += 1
        new_path = path + '_{}'.format(suffix)
    return new_path
