#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- PyTorch     1.4.0
- arrow       0.13.1
"""

import torch 
import arrow
import numpy as np
from torchvision import datasets, transforms

def dataloader4mnistNclasses(classes, batch_size, n_sample):
    """
    dataloader for mnist dataset with N selected classes

    Note: invalid batch (number of appeared classses in the batch is less than the number of 
    classes speficied in `classes`) will be discarded. 
    """
    # download mnist dataset to subfolder named by "data" in the current directory
    data = datasets.MNIST('data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    indices      = [ idx for idx in range(data.targets.shape[0]) if data.targets[idx] in classes ]
    data.targets = data.targets[indices]
    data.data    = data.data[indices]
    dataloader   = torch.utils.data.DataLoader(data, batch_size=n_sample, shuffle=True)
    # yield data samples with number of `n_sample` as a single batch.
    batch_X, batch_Y = [], []
    for data, target in dataloader:
        if torch.unique(target).shape[0] == len(classes):
            X = data   # [n_sample, n_channel, n_xpixel, n_ypixel]
            Y = target # [n_sample]
            batch_X.append(X) 
            batch_Y.append(Y)
        else: 
            print("[%s] invalid batch with number of classes %d < %d" % \
                (arrow.now(), torch.unique(target).shape[0], len(classes)))
        # only yield when each batch has `batch_size` set of data samples.
        if len(batch_X) >= batch_size:
            yield_X = torch.stack(batch_X, dim=0) # [batch_size, n_sample, n_channel, n_xpixel, n_ypixel]
            yield_Y = torch.stack(batch_Y, dim=0) # [batch_size, n_sample]
            # sort X, Y by class
            yield_X, yield_Y = sortbyclass(yield_X, yield_Y)
            # calculate empirical distribution Q
            yield_Q = sortedY2Q(yield_Y)          # [batch_size, n_class, n_sample]
            # clear batch
            batch_X, batch_Y = [], []
            yield yield_X, yield_Y, yield_Q

def sortbyclass(X, Y):
    """
    return the sorted data _X and label _Y by their classes (value of Y)
    
    input
    - X: [batch_size, n_sample, n_feature]
    - Y: [batch_size, n_sample]
    output
    - _X: [batch_size, n_sample, n_feature]
    - _Y: [batch_size, n_sample]
    """
    sorted_ids = torch.argsort(Y, dim=1)
    _Y = [ Y[batch_idx, sorted_id] for batch_idx, sorted_id in enumerate(sorted_ids) ]
    _X = [ X[batch_idx, sorted_id] for batch_idx, sorted_id in enumerate(sorted_ids) ]
    _Y = torch.stack(_Y, dim=0)
    _X = torch.stack(_X, dim=0)
    return _X, _Y

def sortedY2Q(_Y):
    """
    transform the sorted input label into the empirical distribution matrix Q, where
        Q_k^l = 1 / n_k, for n_{k-1} \le l \le n_{k+1}
              = 0, otherwise

    input
    - Y: [batch_size, n_sample]
    output
    - Q: [batch_size, n_class, n_sample]
    """
    batch_size, n_sample = _Y.shape
    # NOTE:
    # it is necessary to require that the number of data points of each class in a single batch 
    # is no less than 1 here.
    classes = torch.unique(_Y)
    n_class = classes.shape[0]
    # N records the number of data points of each class in each batch [batch_size, n_class]
    N = [ torch.unique(_y, return_counts=True)[1] for _y in _Y.split(split_size=1) ]
    N = torch.stack(N, dim=0)
    # construct an empty Q matrix with zero entries
    Q = torch.zeros(batch_size, n_class, n_sample)
    for batch_idx in range(batch_size):
        for class_idx in range(n_class):
            _from = N[batch_idx, :class_idx].sum()
            _to   = N[batch_idx, :class_idx+1].sum()
            n_k   = N[batch_idx, class_idx].float()
            Q[batch_idx, class_idx, _from:_to] = 1 / n_k
    return Q
    