#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn


class AverageKeeper(object):
    """
    Helper class to keep track of averages
    """
    def __init__(self):
        self.sum = 0
        self.n = 0
    def add(self, x):
        self.sum += x
        self.n += 1
    def calculate(self):
        return self.sum / self.n if self.n != 0 else 0
    def reset(self):
        self.sum = 0
        self.n = 0

# TODO move this to model.py
def softmax_pred(linear_out):
    """Apply softmax and get the predictions

    Parameters
    ----------
    linear_out: torch.Tensor
        The tensor output of the pytorch nn model. Assumes 2D, stacked vectors

    Returns
    -------
    A numpy array of the argmax for each vector
    """
    softmax_out = nn.Softmax(dim=1)(linear_out)
    pred = torch.argmax(softmax_out, dim=1)
    pred = pred.squeeze()
    return pred.numpy()