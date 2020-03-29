#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader
"""

import utils
import torch 
import arrow
import numpy as np
from torchvision import datasets, transforms

class MiniMnist(torch.utils.data.Dataset):
    """
    Dataloader for a mini MNIST

    This data loader only utilizes a very small portion of data in MNIST, which contains `N` images for each category (`2 * N` in total). Every iteration, the data loader would yield a batch of sample sets, where 
    each sample set contains `n_sample` samples and each class in this set at least has one sample. 

    You may be able to access:
    - self.data and self.targets for the whole dataset in MNIST
    - self.X and self.Y for the selected mini dataset

    or iterate the dataloader to access the mini dataset in an iterative fashion.
    """

    def __init__(self, classes, batch_size, n_sample, is_train=True, N=50):
        """
        Args:
        - classes:    selected classes in the dataset (from 0 to 9)
        - batch_size: number of sets of samples in one batch
        - n_sample:   number of samples in one set.
        """
        # MNIST dataset
        self.dataset    = datasets.MNIST("data", train=is_train, download=True)
        # configurations
        assert n_sample < N, "n_sample (%d) should be less than N (%d)." % (n_sample, N)
        self.classes    = classes
        self.batch_size = batch_size
        self.n_sample   = n_sample
        self.N          = N

        # data extraction and normalization
        self.data    = (self.dataset.data.float() - torch.min(self.dataset.data).float()) /\
            (torch.max(self.dataset.data).float() - torch.min(self.dataset.data).float())
        self.targets = self.dataset.targets

        # only keep classes specified in the argument `classes` from the dataset
        n        = self.targets.shape[0] # total number of samples
        self.ids = []                    # sets of indices, each set corresponds to a unique class
        for _class in self.classes:
            indices = [ idx for idx in range(n) if self.targets[idx] == _class ]
            assert len(indices) >= N, "data with class %d are less than N (%d)." % (_class, N)
            np.random.shuffle(indices)
            self.ids.append(indices[:N])
        # ids contains the indices of selected samples for each class
        self.ids = np.array(self.ids)    # [n_class, N]

        # for accessing selected data more easier (X, Y have been correctly ordered)
        indices = np.concatenate(self.ids, axis=0)
        self.X  = self.data[indices]
        self.Y  = self.targets[indices]

    def __len__(self):
        # calculate the number of all possible combinations
        return 2 * self.N

    def __str__(self):
        return "Mini MNIST dataset contains %d images with %d classes. For the training, each batch includes %d sets of images, where each set has %d images." 

    def __getitem__(self, _):
        X, Y = [], []
        for j in range(self.batch_size):
            # a single sample set
            x, y = [], []
            # fetch X and Y from each class iteratively
            n_remain_samples = self.n_sample
            _indices         = self.ids.copy()
            for i, _class in enumerate(self.classes):
                # determine numbers of samples selected from each class (>= 1)
                n_samples_class_i = np.random.randint(n_remain_samples - 1) + 1 \
                    if i != len(self.classes) - 1 else n_remain_samples
                n_remain_samples -= n_samples_class_i
                # randomly sample from remained indices in the list
                np.random.shuffle(_indices[i])
                indices_class_i   = _indices[i, :n_samples_class_i]
                # fetch X and Y of class i according to their indices
                x.append(self.data[indices_class_i])
                y.append(self.targets[indices_class_i])
            x = torch.cat(x, dim=0).unsqueeze(1)
            y = torch.cat(y, dim=0)
            X.append(x)
            Y.append(y)
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)
        return X, Y