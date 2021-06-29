from re import L
from torchvision import transforms


def data_transforms(cfg):
    data_aug = cfg.data.data_augmentation
    aug_args = cfg.data_augmentation_args

    operations = {
        'random_crop': transforms.RandomResizedCrop(
            size=(cfg.data.input_size, cfg.data.input_size),
            scale=aug_args.random_crop.scale,
            ratio=aug_args.random_crop.ratio
        ),
        'horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_args.horizontal_flip.prob
        ),
        'vertical_flip': transforms.RandomVerticalFlip(
            p=aug_args.vertical_flip.prob
        ),
        'color_distortion': transforms.ColorJitter(
            brightness=aug_args.color_distortion.brightness,
            contrast=aug_args.color_distortion.contrast,
            saturation=aug_args.color_distortion.saturation,
            hue=aug_args.color_distortion.hue
        ),
        'rotation': transforms.RandomRotation(
            degrees=aug_args.rotation.degrees,
            fill=aug_args.value_fill
        ),
        'translation': transforms.RandomAffine(
            degrees=0,
            translate=aug_args.translation.range,
            fillcolor=aug_args.value_fill
        ),
        'grayscale': transforms.RandomGrayscale(
            p=aug_args.grayscale.prob
        ),
        'gaussian_blur': transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=aug_args.gaussian_blur.kernel_size,
                sigma=aug_args.gaussian_blur.sigma
            )],
            p=aug_args.gaussian_blur.prob
        )
    }

    augmentations = []
    for op in data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    train_preprocess = transforms.Compose([
        *augmentations,
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])

    return train_preprocess, test_preprocess


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
