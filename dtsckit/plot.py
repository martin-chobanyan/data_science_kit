#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# -------------------------------------------
#                FUNCTIONS
# -------------------------------------------

def set_plot_size(*new_size):
    """Set the plot size to the given dimensions

    Parameters
    ----------
    new_size: the width, height dimensions of the plot
    """
    plt.rcParams['figure.figsize'] = new_size


def reset_plot_size():
    """Reset the plot size to the matplotlib standard"""
    set_plot_size(6.0, 4.0)


class CustomPlotSize(object):
    """
    This function defines a `with` statment that applies `set_plot_size` as the entrance
    and `reset_plot_size` as the exit
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __enter__(self):
        set_plot_size(self.width, self.height)

    def __exit__(self, exc_type, exc_val, exc_tb):
        reset_plot_size()


def scatter_categorical(data, labels, color_map, dim=2, plot_size=(12, 8), alpha=0.6):
    """Scatter plot categorical data in two or three dimensions

    Parameters
    ----------
    data: np.array
    labels: iterable
        Labels of each instance
    color_map: iterable or dict
        mapping of category label to color
    dim: int
        plot dimension, two or three (default=2)
    plot_size: tuple(int, int)
        The dimensions of the plot figure in inches according to matplotlib (default=(12, 8))
    alpha: float
        The alpha gradient of the plot(default=0.6)
    """

    # set up the plot
    set_plot_size(*plot_size)
    if dim == 2:
        fig, ax = plt.subplots()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError(f'dim must be 2 or 3, not {dim}')

    # plot the points for each category
    labels_unique = list(set(labels))
    for label in labels_unique:
        label_idx = np.where(labels == label)[0]
        if dim == 2:
            ax.scatter(data[label_idx, 0],
                       data[label_idx, 1],
                       c=color_map[label],
                       label=label,
                       alpha=alpha)
        elif dim == 3:
            ax.scatter(data[label_idx, 0],
                       data[label_idx, 1],
                       data[label_idx, 2],
                       c=color_map[label],
                       label=label,
                       alpha=alpha)
    ax.legend(loc='best')
    plt.show()
    reset_plot_size()


# this should maybe be changed to be more general or wrapped by another function
# TODO: maybe use torchvision.utils.make_grid for the pytorch case and wrap it with a more generic caller
def display_images(images, grid_shape=(1, 2), plot_size=(6, 4)):
    """Plot images of pytorch tensors

    Parameters
    ----------
    images: Tensor, list(Tensor)
        An iterable of pytorch Tensor objects representing images
    grid_shape: tuple(int, int)
        Specifies the number of rows and columns in the grid, respectively
    plot_size: tuple(int, int)
        How big each grid image should be
    """
    set_plot_size(*plot_size)
    fig, axes = plt.subplots(*grid_shape)
    axes = axes.reshape(-1)
    n_leftover_axes = len(axes) - len(images)

    # if there is an extra dimension then remove it
    if len(images[0].size()) == 4:
        images = [img.squeeze() for img in images]

    images = chain(images, [None] * n_leftover_axes)
    for ax, img in zip(axes, images):
        if img is not None:
            ax.imshow(ToPILImage()(img))
        ax.axis('off')
    plt.show()
    reset_plot_size()


# Print the feature maps produced by the subset of a CNN model
# TODO: make this use np.ndenumerate instead of reshape
def compare_feature_maps(image, model, device, grid_shape,
                         plot_size=(12, 8), title='', vmin=-100, vmax=100):
    set_plot_size(*plot_size)
    with torch.no_grad():
        print(title)
        image = image.to(device)
        output = model(image.unsqueeze(0))

        # set up the plot figure
        fig, axes = plt.subplots(*grid_shape)
        axes = axes.reshape(-1)

        # chain the image rasters and fill the remaining spots with Nones
        n_leftover_axes = len(axes) - len(output) - 1
        image_rasters = chain(output.cpu().squeeze(), [None for _ in range(n_leftover_axes)])
        for i, (axis, img) in enumerate(zip(axes, image_rasters)):
            if img is not None:
                axis.imshow(ToPILImage()(img.unsqueeze(0)), vmin=vmin, vmax=vmax, cmap='gray')
            axis.axis('off')
    plt.show()
    print('------------------------------------------------------------------------------------------')
    reset_plot_size()


def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues, figsize=(6, 4)):
    """
    This function is based off of the following matplotlib tutorial:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
