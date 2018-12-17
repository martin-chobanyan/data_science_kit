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
