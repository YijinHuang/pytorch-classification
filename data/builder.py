import os
import pickle

from torchvision import datasets

from .loader import pil_loader
from .transforms import data_transforms, simple_transform
from .dataset import DatasetFromDict, CustomizedImageFolder
from utils.func import mean_and_std, print_dataset_info


def generate_dataset(cfg):
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        mean, std = auto_statistics(
            cfg.base.data_path,
            cfg.base.data_index,
            cfg.data.input_size,
            cfg.train.batch_size,
            cfg.train.num_workers
        )
        cfg.data.mean = mean
        cfg.data.std = std

    train_transform, test_transform = data_transforms(cfg)
    if cfg.base.data_index:
        datasets = generate_dataset_from_pickle(
            cfg.base.data_index,
            train_transform,
            test_transform
        )
    else:
        datasets = generate_dataset_from_folder(
            cfg.base.data_path,
            train_transform,
            test_transform
        )

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path, data_index, input_size, batch_size, num_workers):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    if data_index not in [None, 'None']:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transform)
    else:
        train_path = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(train_path, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_dataset_from_folder(data_path, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = CustomizedImageFolder(train_path, train_transform, loader=pil_loader)
    test_dataset = CustomizedImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = CustomizedImageFolder(val_path, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset


def generate_dataset_from_pickle(pkl, train_transform, test_transform):
    data = pickle.load(open(pkl, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_dataset = DatasetFromDict(train_set, train_transform, loader=pil_loader)
    test_dataset = DatasetFromDict(test_set, test_transform, loader=pil_loader)
    val_dataset = DatasetFromDict(val_set, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset
