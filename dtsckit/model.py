#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


