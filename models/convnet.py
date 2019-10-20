# Version ICLR 11/09/2019
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['convnet']


class conv_model(nn.Module):

    def __init__(self, regime='normal', regime_lr=0.1, regime_momentum=0.9, regime_dampening=0, **kwargs):
        super(conv_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr, 'dampening': regime_dampening,
                 'weight_decay': 0, 'momentum': regime_momentum},
                {'epoch': 81, 'lr': regime_lr * 1e-1},
                {'epoch': 122, 'lr': regime_lr * 1e-2, 'weight_decay': 0},
                {'epoch': 164, 'lr': regime_lr * 1e-3}
            ]
        elif regime == 'fixed':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr, 'dampening': regime_dampening,
                 'weight_decay': 0, 'momentum': regime_momentum}
            ]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def convnet(**kwargs):
    return conv_model(**kwargs)
