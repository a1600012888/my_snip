#!/usr/bin/env mdl
import torch
import numpy as np
import os
import math
class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1

        self.minibatch = 0


class AvgMeter(object):
    name = 'No name'
    def __init__(self, name = 'No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0
    def update(self, mean_var, count = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


# vim: ts=4 sw=4 sts=4 expandtab
