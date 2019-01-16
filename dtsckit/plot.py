#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain, product
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchvision.transforms import ToPILImage


def set_plot_size(*new_size):
    """Set the plot size to the given dimensions

    Parameters
    ----------
    new_size: the width, height dimensions of the plot
    """
    plt.rcParams['figure.figsize'] = new_size


def reset_plot_size():
    """Reset the plot size to the matplotlib standard """
    set_plot_size(6.0, 4.0)


def scatter_categorical(data, labels, color_map, dim=2, plot_size=(12,8), alpha=0.6):
    """Scatter plot categorical data in two or three dimensions

	Parameters
	----------
	data: np.array
	labels: iterable
		labels of each instance
	color_map: iterable or dict
		mapping of category label to color
	dim: int
		plot dimension, two or three (default=2)
	plot_size: tuple(int, int)
		The dimensions of the plot figure (default=(12,8))
	alpha: float
		The alpha gradient of the plot (default=0.6)
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
    unique_labels = list(set(labels))
    for label in unique_labels:
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
def display_images(images, grid_shape, plot_size=(6, 4)):
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
        images = [images.squeeze() for img in images]

    images = chain(images, [None]*n_leftover_axes)
    for ax, img in zip(axes, images):
        if img is not None:
            ax.imshow(ToPILImage()(img))
        ax.axis('off')
    plt.show()
    reset_plot_size()


# Print the feature maps produced by the subset of a CNN model
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
        n_leftover_axes = len(axes)-len(output)-1
        image_rasters = chain(output.cpu().squeeze(), [None for _ in range(n_leftover_axes)])
        for i, (axis, img) in enumerate(zip(axes, image_rasters)):
            if img is not None:
                axis.imshow(ToPILImage()(img.unsqueeze(0)), vmin=vmin, vmax=vmax, cmap='gray')
            axis.axis('off')
    plt.show()
    print('------------------------------------------------------------------------------------------')
    reset_plot_size()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
