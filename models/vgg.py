# Version ICLR 11/09/2019
import torch.nn as nn

__all__ = ['vgg']
batch_norm = nn.BatchNorm2d

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, regime='normal_imnet', regime_momentum=0.9,
                 regime_lr=1e-2, regime_dampening=0, scale_lr=1):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) if num_classes == 1000 else nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512 * 7 * 7, num_classes) if num_classes == 1000 else nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)

        if regime == 'warm-up':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': regime_momentum, 'weight_decay': 5e-4,
                 'step_lambda': ramp_up_lr(regime_lr, regime_lr * scale_lr, 5004 * 5)},
                {'epoch': 5, 'momentum': regime_momentum, 'lr': regime_lr * scale_lr},
                {'epoch': 30, 'momentum': regime_momentum, 'lr': regime_lr * scale_lr * 1e-1},
                {'epoch': 60, 'momentum': regime_momentum, 'lr': regime_lr * scale_lr * 1e-2},
            ]
        if regime == 'normal_cifar':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
                 'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
                {'epoch': 80, 'lr': regime_lr * (0.2 ** 1)},
                {'epoch': 120, 'lr': regime_lr * (0.2 ** 2)},
                {'epoch': 160, 'lr': regime_lr * (0.2 ** 3)}
            ]
        if regime == 'normal_cifar_long':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
                 'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
                {'epoch': 80*3, 'lr': regime_lr * (0.2 ** 1)},
                {'epoch': 120*3, 'lr': regime_lr * (0.2 ** 2)},
                {'epoch': 160*3, 'lr': regime_lr * (0.2 ** 3)}
            ]
        elif regime == 'normal_imnet':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
                 'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
                {'epoch': 30, 'lr': regime_lr * 1e-1},
                {'epoch': 60, 'lr': regime_lr * 1e-2}
            ]
        elif regime == 'fixed':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr, 'dampening': regime_dampening,
                 'weight_decay': 0, 'momentum': regime_momentum}
            ]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # active only for ImageNet
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, batch_norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, bn=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if bn:
                layers += [conv2d, batch_norm(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, bn, **kwargs):
    model = VGG(make_layers(cfgs[cfg], bn=bn), **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return _vgg('vgg16_bn', 'D', True, **kwargs)


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return _vgg('vgg19_bn', 'E', True, **kwargs)


def vgg(**config):
    dataset = config.pop('dataset', 'imagenet')
    depth = config.pop('depth', 11)
    bn = config.pop('bn', False)
    global batch_norm
    if config.pop('gbn', False):
        from models.modules.gbn import GhostBatchNorm
        batch_norm = GhostBatchNorm

    if dataset == 'imagenet':
        config.setdefault('num_classes', 1000)
    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
    if depth == 11:
        if bn is False:
            return vgg11(**config)
        else:
            return vgg11_bn(**config)
    if depth == 13:
        if bn is False:
            return vgg13(**config)
        else:
            return vgg13_bn(**config)
    if depth == 16:
        if bn is False:
            return vgg16(**config)
        else:
            return vgg16_bn(**config)
    if depth == 19:
        if bn is False:
            return vgg19(**config)
        else:
            return vgg19_bn(**config)
