import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet']
batch_norm = nn.BatchNorm2d

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000, regime_momentum=0.9, regime_lr=1e-2, regime_dampening=0):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            batch_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            batch_norm(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            batch_norm(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            batch_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            batch_norm(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': regime_lr,
             'weight_decay': 5e-4, 'momentum': regime_momentum, 'dampening': regime_dampening},
            {'epoch': 10, 'lr': regime_lr / 2},
            {'epoch': 15, 'lr': regime_lr / 10, 'weight_decay': 0},
            {'epoch': 20, 'lr': regime_lr / 20},
            {'epoch': 25, 'lr': regime_lr / 100}
        ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.data_regime = [{
            'transform': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        }]
        self.data_eval_regime = [{
            'transform': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])
        }]

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(**config):
    global batch_norm
    num_classes = getattr(config, 'num_classes', 1000)
    dataset = config.pop('dataset', 'imagenet')
    if config.pop('gbn', False):
        from models.modules.gbn import GhostBatchNorm
        batch_norm = GhostBatchNorm
    return AlexNetOWT_BN(num_classes, **config)
