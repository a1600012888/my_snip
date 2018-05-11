#!/usr/bin/env mdl
import numpy as np
import torch

'''
ensure_class_number, get_accuracy_count, get_accuracy_top_n_count
takes np array as input
'''

def torch_accuracy(output, target, topk = (1, )):
    '''
    param output, target: should be torch Variable

    '''
    #assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    #assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    #print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim = True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def ensure_class_number(label):
    '''
    label should be an numpy array
    '''
    if label.ndim == 1:
        return label
    return label.argmax(axis=1)

def get_accuracy_count(pred, label):
    '''
    :param pred: a 1D or 2D tensor representing classification results
    :param label: a 1D or 2D tensor representing classification groundtruth
            labels
    :return: (nr_correct, total)
    '''
    pred, label = map(ensure_class_number, (pred, label))
    nr_correct = np.sum(pred == label)
    return nr_correct, label.shape[0]


def get_accuracy_top_n_count(pred, label, top_n = 3):
    label = ensure_class_number(label)
    assert (pred.shape[0] == label.shape[0] and label.ndim == 1
            and pred.ndim == 2), (pred.shape, label.shape)
    assert top_n >= 1, top_n
    top_n_label = np.argpartition(-pred, top_n - 1, axis=1)[:, :top_n]
    nr_correct = np.sum(top_n_label == label[:, np.newaxis])
    return nr_correct, label.shape[0]

