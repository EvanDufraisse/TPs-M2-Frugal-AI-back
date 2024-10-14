# -*- coding: utf-8 -*-
""" FourLayerMLP class

@Author: Evan Dufraisse
@Date: Sun Oct 13 2024
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2024 CEA - LASTI
"""

import torch
import torch.nn as nn

class FourLayerMLP(nn.Module):
    def __init__(self):
        super(FourLayerMLP, self).__init__()
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x