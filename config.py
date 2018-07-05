#!/usr/bin/env mdl
import numpy as np
from typing import Tuple, List, Dict
from abc import abstractmethod, abstractproperty, ABCMeta
import os
import json
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

def save_args(args, save_dir = None):
    if save_dir == None:
        param_path = os.path.join(args.resume, "params.json")
    else:
        param_path = os.path.join(save_dir, 'params.json')

    #logger.info("[*] MODEL dir: %s" % args.resume)
    #logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)



def dump_dict(json_file_path, dic):
    with open(json_file_path, 'a') as f:
        f.write('\n')
        json.dump(dic, f)

# vim: ts=4 sw=4 sts=4 expandtab
