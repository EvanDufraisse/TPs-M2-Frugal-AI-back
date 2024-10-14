# -*- coding: utf-8 -*-
""" Easily get the dataloaders from the splits variable

@Author: Evan Dufraisse
@Date: Sun Oct 13 2024
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2024 CEA - LASTI
"""
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

def build_dataloader(features: np.array, labels:List[int], batch_size:int=32, shuffle=False)-> DataLoader:
    """
    Get the dataloader from the splits variable.

    Args:
    splits (dict): Dictionary containing the splits.
    split_name (str): Name of the split to get the dataloader from.

    Returns:
    DataLoader: DataLoader of the split.
    """
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)