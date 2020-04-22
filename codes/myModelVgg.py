import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, cfgs, in_channels=3, num_classes=1000, out_channel=512, init_weights=True):
        super(VGG, self).__init__()

        layers = []
        for v in cfgs:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, num_classes, kernel_size=1, stride=1))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x.squeeze(3).squeeze(2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def myVGG(**kwargs):
    cfgs = [64, 64, 128, 'M', 128, 128, 256, 'M', 256, 256, 512]
    return VGG(cfgs, **kwargs)
