import os
import pickle

from torch.utils.data import Dataset
from torchvision import transforms, datasets

from utils import mean_and_std, pil_loader, print_dataset_info


def generate_dataset(data_config, data_path, data_index=None, batch_size=16, num_workers=8):
    if data_config['mean'] == 'auto' or data_config['std'] == 'auto':
        mean, std = auto_statistics(data_path, data_index, batch_size, num_workers, data_config['input_size'])
        data_config['mean'] = mean
        data_config['std'] = std

    train_tf, test_tf = data_transforms(data_config)
    if data_index:
        datasets = generate_dataset_from_pickle(data_index, train_tf, test_tf)
    else:
        datasets = generate_dataset_from_folder(data_path, train_tf, test_tf)

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path, data_index, batch_size, num_workers, input_size):
    print('Calculating mean and std of training set for data normalization.')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    if data_index:
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


def data_transforms(data_config):
    data_aug = data_config['data_augmentation']
    input_size = data_config['input_size']
    mean, std = data_config['mean'], data_config['std']

    train_preprocess = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=data_aug['brightness'],
            contrast=data_aug['contrast'],
            saturation=data_aug['saturation'],
            hue=data_aug['hue']
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


class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None, loader=pil_loader):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label
