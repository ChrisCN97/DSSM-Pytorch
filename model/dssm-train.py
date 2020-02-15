"""
Input:
    size: (batchsize, trigram_dimension) e.g.(1024, 30k)
    Query sample: num: 1
    Doc Positive sample: num: 1
    Doc Negative sample: num: 4
Output:
    -log(P(Doc Positive sample | Query))
"""

from data-prepro

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(30000, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 128)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        return x