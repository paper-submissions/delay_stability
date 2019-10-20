import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.gbn import GhostBatchNorm
from numpy import sqrt

batch_norm = nn.BatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.bn1 = batch_norm(in_planes, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = batch_norm(out_planes, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, bn_momentum=0.1):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      bn_momentum=bn_momentum)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, bn_momentum=0.1):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet_cifar(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, gbn=False, bn_momentum=0.1, regime='normal',
                 scale_lr=1, regime_lr=0.1, regime_momentum=0.9, regime_dampening=0.9, workers_num=1):
        super(WideResNet_cifar, self).__init__()
        if gbn is True:
            global batch_norm
            batch_norm = GhostBatchNorm
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, bn_momentum=bn_momentum)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, bn_momentum=bn_momentum)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, bn_momentum=bn_momentum)
        # global average pooling and classifier
        self.bn1 = batch_norm(nChannels[3], momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        # self.fc = fixed_proj.Proj(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, batch_norm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)

        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': regime_momentum, 'dampening': regime_dampening,
                 'lr': regime_lr, 'weight_decay': 5e-4},
                {'epoch': 60, 'lr': regime_lr * 2e-1},
                {'epoch': 120, 'lr': regime_lr * 2e-2},
                {'epoch': 160, 'lr': regime_lr * 2e-3}
            ]
        elif regime == 'fixed':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr, 'weight_decay': 0, 'momentum': regime_momentum}
            ]
        elif regime == 'long':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': regime_momentum, 'dampening': regime_dampening,
                 'lr': regime_lr, 'weight_decay': 5e-4},
                {'epoch': 60 * workers_num, 'lr': regime_lr * 2e-1},
                {'epoch': 120 * workers_num, 'lr': regime_lr * 2e-2},
                {'epoch': 160 * workers_num, 'lr': regime_lr * 2e-3}
            ]
        elif regime == 'async2':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
                 'step_lambda': ramp_up_lr(0.1, 0.1 * sqrt(scale_lr), 390 * 5)},
                {'epoch': 5, 'lr': sqrt(scale_lr) * 1e-1},
                {'epoch': 60, 'lr': sqrt(scale_lr) * 2e-2},
                {'epoch': 120, 'lr': sqrt(scale_lr) * 2e-3},
                {'epoch': 160, 'lr': sqrt(scale_lr) * 2e-4}
            ]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wideresnet(**config):
    dataset = config.pop('dataset', 'imagenet')
    if config.pop('quantize', False):
        from .modules.quantize import QConv2d, QLinear, RangeBN
        torch.nn.Linear = QLinear
        torch.nn.Conv2d = QConv2d
        torch.nn.BatchNorm2d = RangeBN

    bn_norm = config.pop('bn_norm', None)
    if bn_norm is not None:
        from .modules.lp_norm import L1BatchNorm2d, TopkBatchNorm2d
        if bn_norm == 'L1':
            torch.nn.BatchNorm2d = L1BatchNorm2d
        if bn_norm == 'TopK':
            torch.nn.BatchNorm2d = TopkBatchNorm2d

    if dataset == 'imagenet':
        config.setdefault('num_classes', 1000)
        depth = config.pop('depth', 28)
        assert False

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 28)
        config.setdefault('widen_factor', 10)
        return WideResNet_cifar(**config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 28)
        config.setdefault('widen_factor', 10)
        return WideResNet_cifar(**config)
