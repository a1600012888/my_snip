#!/usr/bin/env mdl
import torch
import numpy as np
import os
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
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num

class TorchCheckpoint(object):

    def __init__(self, base_path = './data/', high = True):
        assert os.path.isdir(base_path)

        self.high = high
        self.best_epoch = 0
        self.best = 0
        self.base_path = base_path
        self.best_path = os.path.join(base_path, 'best.pt')
        self.metrics = []

        print('checkpoint initialized!!')

    def __call__(self, state_dict, metric, epoch):

        self.metrics.append(metric)

        now_path = os.path.join(self.base_path, 'exp-{}.pt'.format(epoch))

        torch.save(state_dict, now_path)
        print('Model with metric as {} saved in {}!'.format(metric, now_path))
        is_best = False
        if self.high:
            best = max(self.metrics)

            if metric >= best:
                is_best = True
                self.best_epoch = epoch

        else:
            best = min(self.metrics)

            if metric <= best:
                is_best = True
                self.best_epoch = epoch
        self.best = best

        if is_best:
            torch.save(state_dict, self.best_path)

        print('Best model after epoch{}, metric as: {} saved in {}. '.format(self.best_epoch, best, self.best_path))

        return is_best
# vim: ts=4 sw=4 sts=4 expandtab
