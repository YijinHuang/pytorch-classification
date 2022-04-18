import timm
import torch
import torch.nn as nn

from utils.func import print_msg, select_out_features


def generate_model(cfg):
    model = build_model(cfg)

    if cfg.train.checkpoint:
        weights = torch.load(cfg.train.checkpoint)
        model.load_state_dict(weights, strict=True)
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))

    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(cfg.base.device)
    return model


def build_model(cfg):
    network = cfg.train.network
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

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
    return model
