import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class MyModel(nn.Module):
    def __init__(self, backbone, bottleneck_size, num_classes, pretrained=False, **kwargs):
        super(MyModel, self).__init__()

        self.net = backbone(pretrained=pretrained, num_classes=1000, **kwargs)
        self.net.fc = nn.Sequential(
            nn.Dropout(0.2),
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


class MyEfficientNet(nn.Module):
    def __init__(self, net_size, bottleneck_size, num_classes, pretrained=False):
        super(MyEfficientNet, self).__init__()

        if pretrained:
            self.net = EfficientNet.from_pretrained('efficientnet-b{}'.format(net_size))
        else:
            self.net = EfficientNet.from_name('efficientnet-b{}'.format(net_size))

        self.net._fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(bottleneck_size, num_classes)
        )

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
