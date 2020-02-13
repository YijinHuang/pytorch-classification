import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from config import NET_CONFIG


def generate_model(network, num_classes, checkpoint, pretrained):
    if checkpoint:
        model = torch.load(checkpoint)
        print('Load weights form {}'.format(checkpoint))
    else:
        if network not in NET_CONFIG.keys():
            raise Exception('NETWORK name should be one of NET_CONFIG keys in config.py.')
        net_config = NET_CONFIG[network]

        if 'efficientnet' in network:
            if pretrained:
                model = EfficientNet.from_pretrained(network, num_classes=num_classes).cuda()
            else:
                model = EfficientNet.from_name(network, num_classes=num_classes).cuda()
        else:
            model = MyModel(
                net_config['MODEL'],
                net_config['BOTTLENECK_SIZE'],
                num_classes,
                pretrained,
                net_config['DROPOUT'],
                **net_config['OPTIONAL']
            ).cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model

class MyModel(nn.Module):
    def __init__(self, backbone, bottleneck_size, num_classes, pretrained=False, dropout=0.2, **kwargs):
        super(MyModel, self).__init__()

        self.net = backbone(pretrained=pretrained, num_classes=1000, **kwargs)
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bottleneck_size, num_classes)
        )

    def forward(self, x):
        pred = self.net(x)
        return pred


class AngularModel(nn.Module):
    def __init__(self, backbone, bottleneck_size, num_classes, pretrained=False, **kwargs):
        super(AngularModel, self).__init__()

        self.net = backbone(pretrained=pretrained, num_classes=1000, **kwargs)
        self.net.fc = AngularFCLayer(bottleneck_size, num_classes)

    def forward(self, x):
        pred = self.net(x)
        return pred


class AngularFCLayer(nn.Module):
    def __init__(self, in_channels, num_class):
        super(AngularFCLayer, self).__init__()

        self.fc = nn.Linear(in_channels, num_class, bias=False)

    def forward(self, x):
        # normalize W and x
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        return self.fc(x)
