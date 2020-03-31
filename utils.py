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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(1)

def visualize_embedding(H, p_hat, perplexity=20):
    """
    Visualize data embedding on a 2D space using TSNE. 
    
    input
    - H:     [n_sample, n_feature]
    - p_hat: [n_class, n_sample]
    """
    # configuration
    n_class  = p_hat.shape[0] 
    n        = H.shape[0]
    H        = H.numpy()
    p_hat    = p_hat.numpy()
    # fit TSNE
    tsne     = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
    E2D      = tsne.fit_transform(H)

    # plot 
    fig, axs = plt.subplots(1, n_class)
    # ax 
    # ax1      = axs[0]
    # ax2      = axs[1]
    # plot embedding colored by their labels
    # cm1 = plt.cm.get_cmap('Reds')
    # cm2 = plt.cm.get_cmap('Blues')
    cms = [ plt.cm.get_cmap(c) for c in ['Reds', 'Blues', 'Greens'] ]

    for i in range(n_class):
        axs[i].scatter(E2D[:, 0], E2D[:, 1], c=p_hat[i, :], vmin=p_hat[i, :].min(), vmax=p_hat[i, :].max(), cmap=cms[i])
    plt.savefig("results/scatter_%s.pdf" % arrow.now())

    # # plot 
    # fig, axs = plt.subplots(1, 2)
    # cm       = plt.cm.get_cmap('RdYlBu')
    # ax1      = axs[0]
    # ax2      = axs[1]
    # # plot embedding colored by their labels
    # ax1.scatter(E2D[:int(n/2), 0], E2D[:int(n/2), 1], c="b")
    # ax1.scatter(E2D[int(n/2):, 0], E2D[int(n/2):, 1], c="r")
    # # plot embedding colored by p_hat
    # p_hat = p_hat[0] / (p_hat[0] + p_hat[1])
    # ax2.scatter(E2D[:, 0], E2D[:, 1], c=p_hat, vmin=0, vmax=1, cmap=cm)
    # plt.savefig("results/scatter_%s.pdf" % arrow.now())  

def sortedY2Q(Y):
    """
    transform the sorted input label into the empirical distribution matrix Q, where
        Q_k^l = 1 / n_k, for n_{k-1} \le l \le n_{k+1}
              = 0, otherwise

    input
    - Y: [batch_size, n_sample]
    output
    - Q: [batch_size, n_class, n_sample]
    """
    batch_size, n_sample = Y.shape
    # NOTE:
    # it is necessary to require that the number of data points of each class in a single batch 
    # is no less than 1 here.
    classes = torch.unique(Y)
    n_class = classes.shape[0]
    # N records the number of data points of each class in each batch [batch_size, n_class]
    N = [ torch.unique(y, return_counts=True)[1] for y in Y.split(split_size=1) ]
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

# def sortbyclass(X, Y):
#     """
#     return the sorted data _X and label _Y by their classes (value of Y)
    
#     input
#     - X: [batch_size, n_sample, n_feature]
#     - Y: [batch_size, n_sample]
#     output
#     - _X: [batch_size, n_sample, n_feature]
#     - _Y: [batch_size, n_sample]
#     """
#     sorted_ids = torch.argsort(Y, dim=1)
#     _Y = [ Y[batch_idx, sorted_id] for batch_idx, sorted_id in enumerate(sorted_ids) ]
#     _X = [ X[batch_idx, sorted_id] for batch_idx, sorted_id in enumerate(sorted_ids) ]
#     _Y = torch.stack(_Y, dim=0)
#     _X = torch.stack(_X, dim=0)
#     return _X, _Y