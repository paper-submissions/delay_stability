import torch
import torch.nn as nn

__all__ = ['cifar10_shallow', 'cifar100_shallow']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, regime_momentum=0.9, regime_lr=1e-2, regime_dampening=0, regime='normal'):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 384, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(384, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )
        if regime == 'normal':
            self.regime = [
                    {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
                     'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
                    {'epoch': 60, 'lr': regime_lr * (0.1 ** 1)},
                    {'epoch': 120, 'lr': regime_lr * (0.1 ** 2)},
                    {'epoch': 180, 'lr': regime_lr * (0.1 ** 3)}
                ]
        if regime == 'long':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
                 'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
                {'epoch': 60*3, 'lr': regime_lr * (0.1 ** 1)},
                {'epoch': 120*3, 'lr': regime_lr * (0.1 ** 2)},
                {'epoch': 180*3, 'lr': regime_lr * (0.1 ** 3)}
            ]

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x


def cifar10_shallow(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 10)
    return AlexNet(num_classes)


def cifar100_shallow(**kwargs):
    dataset = kwargs.pop('dataset', 'cifar100')
    num_classes = getattr(kwargs, 'num_classes', 100)
    return AlexNet(num_classes, **kwargs)
