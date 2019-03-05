#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle


def read_pickle(path):
    """Read an existing pickle object to memory

    Parameters
    ----------
    path: str
        The file path to the pickle
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def write_pickle(obj, path):
    """Dump any object as a pickle

    Parameters
    ----------
    obj: object
        The target object to be pickled
    path: str
        The file path for the new pickle file
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
