#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loader
"""

import utils
import torch 
import arrow
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import combinations 
from sklearn.datasets import make_swiss_roll

np.random.seed(15)

class MiniSetLoader(torch.utils.data.Dataset):
    """
    Dataloader in a ``mini-set'' fashion.

    This data loader only utilizes a very small portion of data in dataset, which contains `N` points 
    for each category (`2 * N` in total). Every iteration, the data loader would yield a batch of 
    sample sets, where each sample set contains `n_sample` samples and each class in this set at least 
    has one sample. 

    You may be able to access:
    - self.data and self.targets for the whole dataset
    - self.X and self.Y for the selected mini dataset

    or iterate the dataloader to access the mini dataset in an iterative fashion.
    """

    def __init__(self, dataset, classes, batch_size, n_sample, is_normalized=True, N=50):
        """
        Args:
        - dataset:    specified dataset, such as `datasets.MNIST("data", train=is_train, download=True)'
        - classes:    selected classes in the dataset (from 0 to 9)
        - batch_size: number of sets of samples in one batch
        - n_sample:   number of samples in one set.
        - N:          total number of samples for each class
        """
        # MNIST dataset
        self.dataset    = dataset # datasets.MNIST("data", train=is_train, download=True)
        # configurations
        assert n_sample < N, "n_sample (%d) should be less than N (%d)." % (n_sample, N)
        self.classes    = classes
        self.batch_size = batch_size
        self.n_sample   = n_sample
        self.N          = N
        self.n_sampless = self._random_split_n_classes() # all possible combinations of  

        # data extraction and normalization
        self.data    = self.dataset.data \
            if is_normalized \
            else (self.dataset.data.float() - torch.min(self.dataset.data).float()) /\
                (torch.max(self.dataset.data).float() - torch.min(self.dataset.data).float())
        self.targets = self.dataset.targets


        # only keep classes specified in the argument `classes` from the dataset
        n        = self.targets.shape[0] # total number of samples
        self.ids = []                    # sets of indices, each set corresponds to a unique class
        for i, _class in enumerate(self.classes):
            indices = [ idx for idx in range(n) if self.targets[idx] == _class ]
            assert len(indices) >= N, "data with class %d are less than N (%d)." % (_class, N)
            # relabel data
            self.targets[indices] = i
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
        return len(self.classes) * self.N

    def __str__(self):
        return "Mini MNIST dataset contains %d images with %d classes. For the training, each batch includes %d sets of images, where each set has %d images." % \
            (len(self.classes) * self.N, len(self.classes), self.batch_size, self.n_sample)

    def __getitem__(self, _):
        X, Y = [], []
        for j in range(self.batch_size):
            # a single sample set
            x, y = [], []
            # fetch X and Y from each class iteratively
            n_remain_samples = self.n_sample
            _indices         = self.ids.copy()
            n_samples_class  = self.n_sampless[np.random.randint(len(self.n_sampless))]
            for i, _class in enumerate(self.classes):
                # n_samples_class[i] is the number of samples selected from class i (>= 1)
                np.random.shuffle(_indices[i])
                indices_class_i   = _indices[i, :n_samples_class[i]]
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
    
    def _random_split_n_classes(self):
        # list all possible combinations
        combs = combinations(
            range(1, self.n_sample - 1), # from n_sample - 1 possibilities 
            len(self.classes) - 1)       # select n_class - 1 positions
        # convert combinations to numbers of samples we need to split
        n_sampless = []
        for pos in combs:
            pos = [0] + list(pos) + [self.n_sample]
            n_samples = [ pos[i+1] - pos[i] for i in range(len(self.classes)) ]
            n_sampless.append(n_samples)
        return np.array(n_sampless)
    
    def save_figures(self):
        cmap = cm.get_cmap('Greys')
        for i in np.array(self.ids).flatten():
            fig, ax = plt.subplots(1, 1)
            implot  = ax.imshow(self.data[i], vmin=self.X.min(), vmax=self.X.max(), cmap=cmap)
            plt.axis('off')
            plt.savefig(
                "dataimgs/img_num%d_id%d.pdf" % (self.classes[self.targets[i]], i), 
                bbox_inches='tight')
            plt.clf()

# SYNTHETIC DATASETS

class SyntheticGaussianDataset(object):
    """
    Generate synthetic data from multiple Gaussian distributions.
    """

    def __init__(self, n_class, means, covs, N):
        assert n_class == len(means) and n_class == len(covs), \
            "n_class (%d) should be consistent with the number of sets of means and variance (%d, %d)." % \
            (n_class, len(means), len(covs))
        
        self.data    = []
        self.targets = []
        for y, (mean, cov) in enumerate(zip(means, covs)):
            X = np.random.multivariate_normal(mean, cov, N)
            Y = y * np.ones(N)
            self.data.append(X)
            self.targets.append(Y)
        self.data    = torch.Tensor(np.concatenate(self.data))
        self.targets = torch.Tensor(np.concatenate(self.targets))
    
class SyntheticSwissrollDataset(object):
    """
    Generate synthetic data from multiple swiss rolls.
    """

    def __init__(self, N):
        # class 1
        X1, _ = make_swiss_roll(n_samples=N, noise=0.6, random_state=None)
        X1    = X1[:, [0, 2]]
        Y1    = 0 * np.ones(N)
        # class 2
        X2, _ = make_swiss_roll(n_samples=N, noise=0.6, random_state=None)
        dX2   = X2[:, [0, 2]] - np.array([0., 0.])
        X2    = np.array([0., 0.]) - dX2
        Y2    = 1 * np.ones(N)
        # merge
        self.data    = torch.Tensor(np.concatenate([X1, X2]))
        self.targets = torch.Tensor(np.concatenate([Y1, Y2]))
