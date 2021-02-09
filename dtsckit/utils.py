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
        
           
def get_imagenet_label_index(filepath):
    label_names = []
    with open(filepath, 'r') as file:
        for line in file:
            # clean up the text first
            line = line.replace('{', '')
            line = line.replace('}', '')
            line = line.replace('\n', '')
            line = line.replace('\s', '')
            line = line.replace('\'', '')
            line = line.replace('\"', '')

            # isolate the labels
            labels = line.split(':')[1:]
            labels = [label.strip() for label in labels]
            label_names.append(labels)
    return label_names
