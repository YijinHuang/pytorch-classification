import os

import torch
import random
import pickle
from PIL import Image
from config import DATA_CONFIG, DATA_AUGMENTATION
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets


def generate_dataset(data_path, input_size, pkl=None):
    if pkl:
        datasets = generate_dataset_from_pickle(pkl, input_size)
    else:
        datasets = generate_dataset_from_folder(data_path, input_size)

    train_dataset, test_dataset, val_dataset = datasets
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    return datasets


def generate_dataset_from_folder(data_path, input_size):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_preprocess, test_preprocess = data_augmentation(input_size)

    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, test_preprocess)
    val_dataset = datasets.ImageFolder(val_path, test_preprocess)

    return train_dataset, test_dataset, val_dataset


def generate_dataset_from_pickle(pkl, input_size):
    data = pickle.load(open(pkl, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_preprocess, test_preprocess = data_augmentation(input_size)

    train_dataset = DatasetFromDict(train_set, train_preprocess)
    test_dataset = DatasetFromDict(test_set, test_preprocess)
    val_dataset = DatasetFromDict(val_set, test_preprocess)

    return train_dataset, test_dataset, val_dataset


def data_augmentation(input_size):
    data_aug = DATA_AUGMENTATION
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(input_size, input_size),
            scale=data_aug['scale'],
            ratio=data_aug['stretch_ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['ratation'],
            translate=data_aug['translation_ratio'],
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_CONFIG['MEAN'], DATA_CONFIG['STD']),
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_CONFIG['MEAN'], DATA_CONFIG['STD']),
    ])
    return train_preprocess, test_preprocess


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.transform = transform

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


class ScheduledWeightedSampler(Sampler):
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights(self.targets)

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.double)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.double)
        self.sample_weight = torch.zeros(self.num_samples, dtype=torch.double)
        for i, _class in enumerate(self.targets):
            self.sample_weight[i] = self.w0[_class]

    def step(self):
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            self.weights = factor * self.w0 + (1 - factor) * self.wf
            for i, _class in enumerate(self.targets):
                self.sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.sample_weight, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self, train_targets):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights


class LossWeightsScheduler():
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights(self.targets)

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.float32)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.float32)

    def step(self):
        weights = self.w0
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            weights = factor * self.w0 + (1 - factor) * self.wf
        return weights

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self, train_targets):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights
