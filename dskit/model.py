#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import random_split
from dskit.metrics import AverageKeeper


def train_epoch(epoch, model, dataloader, criterion, optimizer, device, print_rate=50):
    """Train the model for an epoch and return the average training loss

    Parameters
    ----------
    epoch: int
        The number id of this epoch
    model: torch.nn.Module
        The pytorch neural network model
    dataloader: torch.utils.data.DataLoader
        The dataloader that will shuffle and batch the dataset
    criterion: nn.Module/callable
        The loss criterion for the model
    optimizer: pytorch Optimizer
        The optimizer for this model
    device: torch.device
        The device for where the model will be trained
    print_rate: int
        The number of batches to print the status update. If -1 nothing will be printed (default=50)
    """
    print_stats = (print_rate != -1)
    if print_stats:
        print('----------------------')
        print(f'Training epoch {epoch}')
        print('----------------------')
        print('Batch\tAverage Loss')
    loss_avg = AverageKeeper()
    model = model.train()
    for i, batch in enumerate(dataloader):
        batch_x = batch[0].to(device)
        batch_y = batch[1].to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        loss_avg.add(loss.detach().item())
        if print_stats and i % print_rate == 0:
            print(f'{i}\t{round(loss_avg.calculate(), 6)}')
    return loss_avg.calculate()


# TODO: fix 'images' and 'breeds'
def validate_epoch(epoch, model, dataloader, criterion, device, print_rate=50):
    """Validate the model for an epoch and return the average validation loss

    Parameters
    ----------
    epoch: int
        The number id of this epoch
    model: torch.nn.Module
        The pytorch neural network model
    dataloader: torch.utils.data.DataLoader
        The dataloader that will shuffle and batch the dataset
    criterion: nn.Module/callable
        The loss criterion for the model
    device: torch.device
        The device for where the model will be trained
    print_rate: int
        The number of batches to print the status update. If -1 nothing will be printed (default=50)
    """
    print_stats = (print_rate != -1)
    if print_stats:
        print('----------------------')
        print(f'Validation epoch {epoch}')
        print('----------------------')
        print('Batch\tAverage Loss')
    loss_avg = AverageKeeper()
    model = model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss_avg.add(loss.detach().item())
            if print_stats and i % print_rate == 0:
                print(f'{i}\t{round(loss_avg.calculate(), 6)}')
    return loss_avg.calculate()


def early_stop(train_loader, eval_loader, model, optimizer, criterion, device,
               check=1, patience=5, maxepochs=500, print_rate=50):
    """Train with early stopping

    An early stopping implementation using the 'train' and 'validate' functions.
    As the model trains, a validation epoch is run every 'check' number of epochs.
    If a new min validation loss is not achieved within 'patience' number of checks, then the model will
    stop training  and return the latest epoch along with the recorded training and validation losses.

    Parameters
    ----------
    train_loader: DataLoader
        The data loader for the training instances
    eval_loader: DataLoader
        The data loader for the validation instances
    model: nn.Module
        The neural network module
    optimizer: nn.optim
        The optimizer associated with the model
    criterion: nn.Module
        The loss criterion associated for the model
    device: torch.device
        The device where the model will train
    check: int
        The number of epochs that must pass to periodically check the validation loss (default=1)
    patience: int
        The number of times that the validation can avoid not reaching a new minimum (default=5)
    maxepochs: int
        The maximum number of epochs the model can train (default=args.maxepochs)
    print_rate: int
        The print rate for the training and validation losses. Set to -1 for no printing (default=50)

    Returns
    -------
    A tuple of the stopping epoch number, training losses (list(float)), and validation losses (list(float))
    """
    epoch = 0
    p = 0
    best_validation_loss = float('inf')
    stop_epoch = 0

    training_losses = []
    validation_losses = []

    # while the validation loss has not consistently increased
    while p < patience:

        # train the model for check steps
        for i in range(check):

            if epoch == maxepochs:
                return stop_epoch, training_losses, validation_losses


def num_params(model, trainable=False):
    """Get the number of parameters in the model

    Parameters
    ----------
    model: nn.Module
        The target pytorch model
    trainable: bool
        If True then ignore parameters that cannot train

    Returns
    -------
    The total int number of parameters in the neural network.
    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def checkpoint(model, filepath):
    """Save the state of the model

    To restore the model do the following:
    >> the_model = TheModelClass(*args, **kwargs)
    >> the_model.load_state_dict(torch.load(PATH))

    Parameters
    ----------
    model: nn.Module
        The pytorch model to be saved
    filepath: str
        The filepath of the pickle
    """
    torch.save(model.state_dict(), filepath)


def create_training_fold(dataset, k=5):
    """Split the dataset into training and validation

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        A pytorch Dataset representing the full training set
    k: int
        The number of folds in the k-fold cross validation (default=5)
    """
    n_validation = int((1/k) * len(dataset))
    return random_split(dataset, [len(dataset)-n_validation, n_validation])
