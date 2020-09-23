import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import NET_CONFIG


def generate_model(network, num_classes, checkpoint, pretrained):
    if checkpoint:
        model = torch.load(checkpoint)
        print('Load weights form {}'.format(checkpoint))
    else:
        if network not in NET_CONFIG.keys():
            raise Exception('NETWORK name should be one of NET_CONFIG keys in config.py.')
        net_config = NET_CONFIG[network]

    model = MyModel(
        net_config['MODEL'],
        net_config['BOTTLENECK_SIZE'],
        num_classes,
        pretrained,
        **net_config['OPTIONAL']
    ).cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


class MyModel(nn.Module):
    def __init__(self, backbone, bottleneck_size, num_classes, pretrained=False, **kwargs):
        super(MyModel, self).__init__()

        self.net = backbone(pretrained=pretrained, **kwargs)
        self.net.fc = nn.Linear(bottleneck_size, num_classes)

    def forward(self, x):
        pred = self.net(x)
        return pred
