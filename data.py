import os
import random
import pickle

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets


def generate_dataset(data_config, data_path, pkl=None):
    if pkl:
        datasets = generate_dataset_from_pickle(pkl, data_config)
    else:
        datasets = generate_dataset_from_folder(data_path, data_config)

    train_dataset, test_dataset, val_dataset = datasets
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    return datasets


def generate_dataset_from_folder(data_path, data_config):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_preprocess, test_preprocess = data_augmentation(data_config)

    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, test_preprocess)
    val_dataset = datasets.ImageFolder(val_path, test_preprocess)

    return train_dataset, test_dataset, val_dataset


def generate_dataset_from_pickle(pkl, data_config):
    data = pickle.load(open(pkl, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_preprocess, test_preprocess = data_augmentation(data_config)

    train_dataset = DatasetFromDict(train_set, train_preprocess)
    test_dataset = DatasetFromDict(test_set, test_preprocess)
    val_dataset = DatasetFromDict(val_set, test_preprocess)

    return train_dataset, test_dataset, val_dataset


def data_augmentation(data_config):
    data_aug = data_config['data_augmentation']
    input_size = data_config['input_size']
    mean, std = data_config['mean'], data_config['std']

    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=data_aug['brightness'],
            contrast=data_aug['contrast']
        ),
        transforms.RandomResizedCrop(
            size=(input_size, input_size),
            scale=data_aug['scale'],
            ratio=data_aug['ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['degrees'],
            translate=data_aug['translate']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_preprocess, test_preprocess


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.pil_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')