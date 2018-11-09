# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        h = F.relu(F.dropout(self.fc1(x), p=0.5))
        h = F.relu(F.dropout(self.fc2(h), p=0.5))
        h = F.relu(F.dropout(self.fc3(h), p=0.5))
        return h
