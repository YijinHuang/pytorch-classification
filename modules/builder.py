import timm
import torch
import torch.nn as nn
from torchvision import models

from utils.func import print_msg, select_out_features


def generate_model(cfg):
    model = build_model(cfg)

    if cfg.train.checkpoint:
        if cfg.dist.distributed:
            loc = 'cuda:{}'.format(cfg.dist.gpu)
            weights = torch.load(cfg.train.checkpoint, map_location=loc, weights_only=True)          
        else:
            weights = torch.load(cfg.train.checkpoint, weights_only=True)

        incompatible_keys = model.load_state_dict(weights, strict=False)
        if set(incompatible_keys.missing_keys) == {'fc.weight', 'fc.bias'}:
            print_msg('Pre-trained weights are loaded, but the classification layer weights are missing and will be randomly initialized for the new task.')
        elif incompatible_keys.missing_keys:
            print_msg('Pre-trained weights are loaded, but the following keys are missing: {}'.format(incompatible_keys.missing_keys))
        if incompatible_keys.unexpected_keys:
            print_msg('Pre-trained weights are loaded, but the following keys are not used: {}'.format(incompatible_keys.unexpected_keys))

        print_msg('Load weights form {}'.format(cfg.train.checkpoint))

    if cfg.dist.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(cfg.dist.gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.dist.gpu])
    else:
        model = model.to(cfg.base.device)

    return model


def build_model(cfg):
    network = cfg.train.network
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    if cfg.train.backend == 'timm':
        if 'vit' in network or 'swin' in network:
            model = timm.create_model(
                network,
                img_size=cfg.data.input_size,
                in_chans=cfg.data.in_channels,
                num_classes=out_features,
                pretrained=cfg.train.pretrained,
            )
        else:
            model = timm.create_model(
                network,
                in_chans=cfg.data.in_channels,
                num_classes=out_features,
                pretrained=cfg.train.pretrained,
            )
    elif cfg.train.backend == 'torchvision':
        model = build_torchvision_model(
            cfg.train.network,
            out_features,
            cfg.train.pretrained
        )
    return model


def build_torchvision_model(network, num_classes, pretrained=False):
    model = BUILDER[network](pretrained=pretrained)
    if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'densenet' in network:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'vgg' in network:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif 'mobilenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'squeezenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
}