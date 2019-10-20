# Version ICLR 11/09/2019
import torch
import torch.nn as nn
from models.modules.fixed_proj import LinearFixed

__all__ = ['fully_conn']


class fully_conn_model(nn.Module):

    def __init__(self, depth=3, width=1024, regime='normal', regime_lr=0.1, regime_momentum=0.9, regime_dampening=0,
                 fixed_linear=False, **kwargs):
        super(fully_conn_model, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(784, width), nn.LeakyReLU())
        self.layers = [nn.Sequential(nn.Linear(width, width), nn.LeakyReLU()) for _ in range(0, depth - 1)]
        self.layers = [*self.first_layer, *self.layers]
        self.layers = nn.Sequential(*self.layers)
        self.classifier = LinearFixed(width, 10) if fixed_linear else nn.Linear(width, 10)

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

    def forward(self, inputs):
        out = self.layers(inputs.view(-1, 784))
        out = self.classifier(out)
        out = out.view(-1, 10)
        return out


def fully_conn(**kwargs):
    return fully_conn_model(**kwargs)
