from functools import reduce

import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(32 * 6 * 6, 10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def flatten(self, input):
        size = input.size()[1:]
        flattened_size = reduce(lambda x, y: x * y, size)
        flattened_input = input.view(-1, flattened_size)
        return flattened_input
