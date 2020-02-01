import os

import torch
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from config import DATA_CONFIG


def generate_data(data_path, input_size):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    data_aug = DATA_CONFIG['DATA_AUGMENTATION']
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

    train_dataset = datasets.ImageFolder(train_path, train_preprocess)
    test_dataset = datasets.ImageFolder(test_path, test_preprocess)
    val_dataset = datasets.ImageFolder(val_path, test_preprocess)

    return train_dataset, test_dataset, val_dataset


class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.decay_rate = DATA_CONFIG['DECAY_RATE']
        self.w0 = torch.as_tensor(DATA_CONFIG['INITIAL_SAMPLING_WEIGHTS'], dtype=torch.double)
        self.wf = torch.as_tensor(DATA_CONFIG['FINAL_SAMPLING_WEIGHTS'], dtype=torch.double)
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)

    def step(self):
        self.epoch += 1
        factor = self.decay_rate**(self.epoch - 1)
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img):
        sigma = self.sigma
        if not sigma > 0.0:
            color_vec = torch.zeros(3, dtype=torch.float32)
        else:
            color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))

        color_vec = color_vec.squeeze()
        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)
