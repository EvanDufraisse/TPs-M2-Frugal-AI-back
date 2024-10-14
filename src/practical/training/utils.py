# -*- coding: utf-8 -*-
""" Utils for training and evaluation loops.

@Author: Evan Dufraisse
@Date: Sun Oct 13 2024
@Contact: evan[dot]dufraisse[at]cea[dot]fr
@License: Copyright (c) 2024 CEA - LASTI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def train_one_batch(model, criterion, optimizer, inputs, targets):
    """
    Train one batch of data.
    Args:
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        inputs (torch.Tensor): The input data.
        targets (torch.Tensor): The target data.
    Returns:
        float: The loss value of the batch.
    """

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_on_dataloader(model, loss_criterion, data_loader):
    """
    Evaluate the performance of a model on a given data loader.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loss_criterion (torch.nn.Module): The loss function to calculate the loss.
        data_loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
    Returns:
        float: The average loss over the evaluation data.
        float: The accuracy of the model on the evaluation data.
        list: The predicted labels for the evaluation data.
        list: The true labels for the evaluation data.
    
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_criterion(outputs, targets)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            y_pred.extend(predicted.tolist())
            y_true.extend(targets.tolist())
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / len(data_loader), total_correct / total_samples, y_pred, y_true




class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else: 
            return loss.sum()