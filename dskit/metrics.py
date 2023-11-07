#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class AverageKeeper(object):
    """
    Helper class to keep track of averages
    """
    def __init__(self):
        self.sum = 0
        self.n = 0
        self.running_avg = []

    def add(self, x):
        """Update the current running sum"""
        self.sum += x
        self.n += 1

    def calculate(self):
        """Calculate the current average and append to the running average"""
        avg = self.sum / self.n if self.n != 0 else 0
        self.running_avg.append(avg)
        return avg

    def reset(self, complete=False):
        """Reset the average counter

        Parameters
        ----------
        complete: bool
            If complete is True, then the running average will be reset as well
        """
        self.sum = 0
        self.n = 0
        if complete:
            self.running_avg = []

# ------------------------------------
#       PYTORCH HELPER FUNCTIONS
# ------------------------------------


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
