#!/usr/bin/env mdl
import functools
import collections
from typing import Dict, Any, Tuple, Union
from enum import Enum
import numpy as np



class BaseDataset:
    """
    A BaseDataset is to be inherited by a user defined Dataset instance in `dataset.py`.

    Some important attributes:

    - ``minibatch_size``: number of instances per minibatch
    - ``instance_per_epoch``: if a dataset is consumed by epochs, ``instance_per_epoch``
       is the number of instances every epoch

    """

    _attrs = ['minibatch_size', 'instance_per_epoch']

    def __init__(self, dataset_name):
        """
        :param dataset_name: specify the name of the dataset, usually one of
            'train', 'validation' and 'test'.

        Note that only metainfo of this dataset should be initialized in :func:`__init__`,
        costly initialization should be implemented via :func:`load`.
        """
        self.dataset_name = dataset_name

    def load(self):
        """
        This is designed to do memory consumptive or other costly
        initialization works. It should be called before ``instance_generator``.

        :return: The dataset itself.
        :rtype: BaseDataset
        """
        return self



    def instance_generator(self, encoded=False):
        raise NotImplementedError()


    def batch_generator(self, batch_size=None, encoded=False):
        """\
        Stack several instances to a batch, then yield.

        :param batch_size: number of instances in the batch
        :param encoded: whether the generated data are eoncoded fro remote consumption, if
            it is true, the type of data batch will be ``tuple`` instead of ``np.array``.
        """

        count = 0
        buffer = collections.defaultdict(list)

        for instance in self.instance_generator(encoded=encoded):
            for k, v in instance.items():
                buffer[k].append(v)
            count += 1

            if count == batch_size:
                if encoded:
                    # when data is encoded, it may have variable length
                    batch = {
                        k: tuple(v) for k, v in buffer.items()
                    }
                else:
                    batch = {
                        k: list2nparray(v) for k, v in buffer.items()
                    }
                yield batch

                buffer = collections.defaultdict(list)
                count = 0

    def minibatch_generator(self):
        """\
        Generate data at minibatch.
        """
        yield from self.batch_generator(batch_size=self.minibatch_size, encoded=False)


def EpochDataset(dataset: BaseDataset, minibatch_size=None):
    """\
    Make dataset consumeble through epoch-minibatch pattern by adding an
    :func:`epoch_generator` method to dataset.

    :param dataset: a dataset that provides minibatch_generator
    :param minibatch_size: override the minibatch_size set in dataset.
    """

    import types

    def epoch_generator(self):
        feed = iter(self.minibatch_generator())

        def epoch(feed):
            for i in range(self.minibatch_per_epoch):
                yield next(feed)

        while True:
            yield epoch(feed)

    dataset.minibatch_size = minibatch_size or dataset.minibatch_size
    dataset.minibatch_per_epoch = dataset.instance_per_epoch // dataset.minibatch_size

    dataset.epoch_generator = types.MethodType(epoch_generator, dataset)
    return dataset

def list2nparray(lst, dtype=None):
    """fast conversion from nested list to ndarray by pre-allocating space"""
    if isinstance(lst, np.ndarray):
        return lst
    assert isinstance(lst, (list, tuple)), 'bad type: {}'.format(type(lst))
    assert lst, 'attempt to convert empty list to np array'
    if isinstance(lst[0], np.ndarray):
        dim1 = lst[0].shape
        assert all(i.shape == dim1 for i in lst)
        if dtype is None:
            dtype = lst[0].dtype
            assert all(i.dtype == dtype for i in lst), \
                'bad dtype: {} {}'.format(dtype, set(i.dtype for i in lst))
    elif isinstance(lst[0], (int, float, complex, np.number)):
        return np.array(lst, dtype=dtype)
    else:
        dim1 = list2nparray(lst[0])
        if dtype is None:
            dtype = dim1.dtype
        dim1 = dim1.shape
    shape = [len(lst)] + list(dim1)
    rst = np.empty(shape, dtype=dtype)
    for idx, i in enumerate(lst):
        rst[idx] = i

    return rst

# vim: ts=4 sw=4 sts=4 expandtab
