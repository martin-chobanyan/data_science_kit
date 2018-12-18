#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from dtsckit.metrics import AverageKeeper


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
    optimizer: torch.optim.Optimizer
        The optimizer for this model
    device: torch.device
        The device for where the model will be trained
    print_rate: int
        The number of batches to print the status update (default=50)
    """
    print('----------------------')
    print(f'Training epoch {epoch}')
    print('----------------------')
    print('Batch\tAverage Loss')
    loss_avg = AverageKeeper()
    model = model.train()
    for i, batch in enumerate(dataloader):
        images = batch[0].to(device)
        breeds = batch[1].to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, breeds)
        loss.backward()
        optimizer.step()

        loss_avg.add(loss.detach().item())
        if i % print_rate == 0:
            print(f'{i}\t{round(loss_avg.calculate(), 6)}')

    return loss_avg.calculate()


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
        The number of batches to print the status update (default=50)
    """
    print('----------------------')
    print(f'Validation epoch {epoch}')
    print('----------------------')
    print('Batch\tAverage Loss')
    loss_avg = AverageKeeper()
    model = model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch[0].to(device)
            breeds = batch[1].to(device)
            out = model(images)
            loss = criterion(out, breeds)
            loss_avg.add(loss.detach().item())
            if i % print_rate == 0:
                print(f'{i}\t{round(loss_avg.calculate(), 6)}')

    return loss_avg.calculate()


# TODO: patch this up so that it works...
def early_stop(train_loader, eval_loader, model, optimizer, criterion,
               maxepochs, device, n=1, patience=3):

    epoch_i = 0
    patience_i = 0
    best_validation_loss = float('inf')
    stop = 0

    # while the validation loss has not consistently increased
    while patience_i < patience:

        # train the model for n steps
        for i in range(n):

            if epoch_i == maxepochs:
                return stop

            train(epoch_i, train_loader, model, optimizer, criterion, device)
            epoch_i += 1

        # get the validation loss
        validation_loss = validate(epoch_i, eval_loader, model, criterion, device)

        if validation_loss < best_validation_loss:

            patience_i = 0
            best_validation_loss = validation_loss

            # save the model and update the stopping epoch
            checkpoint(args.outputfolder, epoch_i, model, best_validation_loss)
            stop = epoch_i

        else:
            patience_i += 1

    return stop
