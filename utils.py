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

class Dataloader4MNIST(torch.utils.data.Dataset):
    """Dataloader for MNIST"""

    def __init__(self, classes, batch_size, n_sample, shuffle=True):
        """
        Args:
        - classes:    selected classes in the dataset (from 0 to 9)
        - batch_size: number of sets of samples in one batch
        - n_sample:   number of samples in one set.
        """
        # MNIST dataset
        self.dataset    = datasets.MNIST("data", train=True, download=True)
        # configurations
        self.batch_size = batch_size
        self.n_sample   = n_sample
        self.n_channel  = 1
        self.n_pixel    = self.dataset.data.shape[1]
        # only keep classes specified in the argument `classes` from the dataset
        indices              = [ idx 
            for idx in range(self.dataset.targets.shape[0]) 
            if self.dataset.targets[idx] in classes ]
        # shuffle the dataset
        if shuffle:
            np.random.shuffle(indices)
        self.dataset.targets = self.dataset.targets[indices]
        self.dataset.data    = self.dataset.data[indices]
        # organize data as sets of samples
        n                    = int(len(indices) - len(indices) % self.n_sample)
        n_set                = int(n / self.n_sample)
        self.dataset.targets = self.dataset.targets[:n].view(n_set, self.n_sample)
        self.dataset.data    = self.dataset.data[:n].view(n_set, self.n_sample, 1, self.n_pixel, self.n_pixel)
        # remove sets whose class set are intact.
        valid_set_indices = []
        for set_idx in range(n_set):
            if torch.unique(self.dataset.targets[set_idx]).shape[0] == len(classes):
                valid_set_indices.append(set_idx)
        n_valid_set          = int(len(valid_set_indices) - len(valid_set_indices) % self.batch_size)
        self.n_batch         = int(n_valid_set / self.batch_size)
        self.dataset.targets = self.dataset.targets[valid_set_indices][:n_valid_set]
        self.dataset.data    = self.dataset.data[valid_set_indices][:n_valid_set]
        # normalization
        self.dataset.data    = (self.dataset.data.float() - torch.min(self.dataset.data).float()) /\
            (torch.max(self.dataset.data).float() - torch.min(self.dataset.data).float())
    
    def __len__(self):
        return self.n_batch

    def __getitem__(self, batch_idx):
        # batch indices for current batch
        batch_indices = np.arange(
            self.batch_size * batch_idx, 
            self.batch_size * (batch_idx + 1))
        # get data in current batch
        X = self.dataset.data[batch_indices]
        Y = self.dataset.targets[batch_indices]
        # sort X, Y by class
        _X, _Y = sortbyclass(X, Y)
        # calculate empirical distribution Q
        Q = sortedY2Q(_Y)                               # [batch_size, n_class, n_sample]
        return _X, _Y, Q



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
    