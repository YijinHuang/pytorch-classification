import os
import pickle

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data import DatasetFromDict


def auto_statistics(data_path, data_index, batch_size, num_workers):
    print('Calculating mean and std of training set for data normalization.')
    if data_index:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transforms.ToTensor())
    else:
        train_path = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(train_path, transform=transforms.ToTensor())

    return mean_and_std(train_dataset, batch_size, num_workers)


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
        channel_std += X.std((2, 3)).sum(0)
        num_samples += X.size(0)

    channel_mean /= num_samples
    channel_std /= num_samples

    return channel_mean.tolist(), channel_std.tolist()

