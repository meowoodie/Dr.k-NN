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
        



# class Dataloader4mnist(torch.utils.data.Dataset):
#     """Dataloader for MNIST"""

#     def __init__(self, classes, batch_size, n_sample, shuffle=True, is_train=True, train_ratio=0.9):
#         """
#         Args:
#         - classes:    selected classes in the dataset (from 0 to 9)
#         - batch_size: number of sets of samples in one batch
#         - n_sample:   number of samples in one set.
#         """
#         # MNIST dataset
#         self.dataset      = datasets.MNIST("data", train=True, download=True)
#         # configurations
#         self.classes      = classes
#         self.batch_size   = batch_size
#         self.n_sample     = n_sample
#         self.n_channel    = 1
#         self.n_pixel      = self.dataset.data.shape[1]
#         self.shuffle      = shuffle
#         self.train_ratio  = train_ratio
#         self.is_train     = is_train
#         # normalization
#         self.dataset.data = (self.dataset.data.float() - torch.min(self.dataset.data).float()) /\
#             (torch.max(self.dataset.data).float() - torch.min(self.dataset.data).float())
#         # for train or for test?
#         if self.is_train:
#             self._prepare_trainset()
#         else:
#             self._prepare_testset()
        
#     def _prepare_trainset(self):
#         """
#         prepare trainset, which is organized by [n_batch, batch_size, n_sample, (data_dim)]
#         """
#         n                    = self.dataset.targets.shape[0]
#         data_range           = range(0, int(n * self.train_ratio))
#         # only keep classes specified in the argument `classes` from the dataset
#         indices              = [ idx for idx in data_range if self.dataset.targets[idx] in self.classes ]
#         # shuffle the dataset
#         if self.shuffle:
#             np.random.shuffle(indices)
#         self.dataset.targets = self.dataset.targets[indices]
#         self.dataset.data    = self.dataset.data[indices]
#         # organize data as sets of samples
#         n                    = int(len(indices) - len(indices) % self.n_sample)
#         n_set                = int(n / self.n_sample)
#         self.dataset.targets = self.dataset.targets[:n].view(n_set, self.n_sample)
#         self.dataset.data    = self.dataset.data[:n].view(n_set, self.n_sample, 1, self.n_pixel, self.n_pixel)
#         # remove sets whose class set are intact.
#         valid_set_indices = []
#         for set_idx in range(n_set):
#             if torch.unique(self.dataset.targets[set_idx]).shape[0] == len(self.classes):
#                 valid_set_indices.append(set_idx)
#         n_valid_set          = int(len(valid_set_indices) - len(valid_set_indices) % self.batch_size)
#         self.dataset.targets = self.dataset.targets[valid_set_indices][:n_valid_set]
#         self.dataset.data    = self.dataset.data[valid_set_indices][:n_valid_set]
#         # number of batches
#         self.n               = int(n_valid_set / self.batch_size)
    
#     def _prepare_testset(self):
#         """
#         prepare testset, which is organized by [n_batch, batch_size, (data_dim)]
#         """
#         n                    = self.dataset.targets.shape[0]
#         data_range           = range(int(n * self.train_ratio), n)
#         # only keep classes specified in the argument `classes` from the dataset
#         indices              = [ idx for idx in data_range if self.dataset.targets[idx] in self.classes ]
#         # shuffle the dataset
#         if self.shuffle:
#             np.random.shuffle(indices)
#         self.dataset.targets = self.dataset.targets[indices]
#         self.dataset.data    = self.dataset.data[indices]
#         self.dataset.data    = self.dataset.data.unsqueeze(1)
#         # number of batches
#         self.n               = len(indices)
    
#     def __len__(self):
#         return self.n

#     def __getitem__(self, idx):
#         if self.is_train:
#             # batch indices for current batch
#             batch_indices = np.arange(
#                 self.batch_size * idx, 
#                 self.batch_size * (idx + 1))
#             # get data in current batch
#             X = self.dataset.data[batch_indices]
#             Y = self.dataset.targets[batch_indices]
#             # sort X, Y by class
#             _X, _Y = utils.sortbyclass(X, Y)
#             # calculate empirical distribution Q
#             Q = utils.sortedY2Q(_Y) # [batch_size, n_class, n_sample]
#             return _X, _Y, Q
#         else:
#             # get data in current iteration
#             x = self.dataset.data[idx]    # [1, n_pixel, n_pixel]
#             y = self.dataset.targets[idx] # []
#             return x, y