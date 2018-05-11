#!/usr/bin/env mdl
import numpy as np
from typing import Tuple, List, Dict
from abc import abstractmethod, abstractproperty, ABCMeta

class MultiStageLearningRatePolicy(object):
    '''

    '''

    _stages = None
    def __init__(self, stages:List[Tuple[int, float]]):

        assert(len(stages) >= 1)
        self._stages = stages


    def __call__(self, cur_ep:int) -> float:
        e = 0
        for pair in self._stages:
            e += pair[0]
            if cur_ep < e:
                return pair[1]
      #  return pair[-1][1]
        return pair[-1]




class Config(object):

    lr = [[10, 0.1],
               [10, 0.01],
               [10, 0.001],
               [10, 0.0001]]

    weight_decacy = {'fc': 1e-4,
                     'conv': 1e-4,
                     'bn': 1e-4}

    minibatch_size = 32

    epochs = 45


    def __init__(self):
        pass

    @abstractmethod
    def get_learning_rate(self, cur_ep:int) -> float:

        pass

    @abstractmethod
    def create_oprimizer(self):
        pass


# vim: ts=4 sw=4 sts=4 expandtab
