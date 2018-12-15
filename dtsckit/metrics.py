#!/usr/bin/env python
# -*- coding: utf-8 -*-

class AverageKeeper(object):
    """
    Helper class to keep track of averages
    """
    def __init__(self):
        self.sum = 0
        self.n = 0
    def add(self, x):
        self.sum += x
        self.n += 1
    def calculate(self):
        return self.sum / self.n if self.n != 0 else 0
    def reset(self):
        self.sum = 0
        self.n = 0


