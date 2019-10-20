import torch
from torch.nn import Module, BatchNorm2d


class GhostBatchNorm(Module):

    def __init__(self, num_features, chunk_size=128, momentum=0.1):
        print('Using Ghost Batch Norm of size {}'.format(chunk_size))
        super(GhostBatchNorm, self).__init__()
        self.bn = BatchNorm2d(num_features, momentum=momentum)
        self.num_features = num_features
        self.chunk_size = chunk_size
        self.weight = self.bn.weight
        self.bias = self.bn.bias
        self.running_mean = self.bn.running_mean
        self.running_var = self.bn.running_var

    def forward(self, input):
        input_bn = list()
        input_chunks = torch.split(input, self.chunk_size)
        for x in input_chunks:
            input_bn.append(self.bn(x))
        x = torch.cat(input_bn)
        return x
