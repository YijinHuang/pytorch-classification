import torch
from torchvision import transforms
from packaging import version


def data_transforms(cfg):
    data_aug = cfg.data.data_augmentation
    aug_args = cfg.data_augmentation_args

    operations = {
        'random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(cfg.data.input_size, cfg.data.input_size),
                scale=aug_args.random_crop.scale,
                ratio=aug_args.random_crop.ratio
            ),
            p=aug_args.random_crop.prob
        ),
        'horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_args.horizontal_flip.prob
        ),
        'vertical_flip': transforms.RandomVerticalFlip(
            p=aug_args.vertical_flip.prob
        ),
        'color_distortion': random_apply(
            transforms.ColorJitter(
                brightness=aug_args.color_distortion.brightness,
                contrast=aug_args.color_distortion.contrast,
                saturation=aug_args.color_distortion.saturation,
                hue=aug_args.color_distortion.hue
            ),
            p=aug_args.color_distortion.prob
        ),
        'rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_args.rotation.degrees,
                fill=aug_args.value_fill
            ),
            p=aug_args.rotation.prob
        ),
        'translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_args.translation.range,
                fillcolor=aug_args.value_fill
            ),
            p=aug_args.translation.prob
        ),
        'grayscale': transforms.RandomGrayscale(
            p=aug_args.grayscale.prob
        )
    }

    if version.parse(torch.__version__) >= version.parse('1.7.1'):
        operations['gaussian_blur'] = random_apply(
            transforms.GaussianBlur(
                kernel_size=aug_args.gaussian_blur.kernel_size,
                sigma=aug_args.gaussian_blur.sigma
            ),
            p=aug_args.gaussian_blur.prob
        )

    augmentations = []
    for op in data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    test_preprocess = transforms.Compose(normalization)

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
