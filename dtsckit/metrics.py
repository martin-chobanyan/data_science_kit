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


#####################################
######PYTORCH HELPER FUNCTIONS#######
#####################################


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


def binary_breakdown(x):
    """Number of positive and negative examples in a binary tensor
    
    Parameters
    ----------
    x: torch.tensor
        A tensor consisting of only ones and zeros (or True/False)
        
    Returns
    -------
    A tuple(int, int) for the number of postive and negative examples, respectively
    """
    n_elements = np.prod(x.size()).item()
    n_pos = x.sum().item()
    n_neg = n_elements - n_pos
    return n_pos, n_neg


def tensor_cnf_mtrx(gt, pred):
    """Tensor confusion matrix
    
    Create a confusion matrix from a tensor of binary variables (0s and 1s).
    The matrix will be ordered as such
    
    ----------------------------------------
    |  True Positive   |   False Negative  |
    ----------------------------------------
    |  False Positive  |   True Negative   |
    ----------------------------------------
    
    Parameters
    ----------
    gt: torch.tensor
        The ground truth tensor
    pred: torch.tensor
        The predicted tensor
        
    Returns   
    -------
    The confusion matrix as a 2x2 numpy array
    """
    
    # get the confusion results
    gt, pred = gt.float(), pred.float()
    confusion_results = pred / gt
    
    tp = torch.sum(confusion_results == 1).item()
    fn = torch.sum(confusion_results == 0).item()
    fp = torch.sum(confusion_results == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_results)).item()
    
    return np.array([[tp, fn], [fp, tn]])
