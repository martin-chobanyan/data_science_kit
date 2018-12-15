#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

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
