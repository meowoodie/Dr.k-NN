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

def tvloss(p_hat):
    """TV loss"""
    # p_max, _ = torch.max(p_hat, dim=1) # [batch_size, n_sample]
    # return p_max.sum(dim=1).mean()     # scalar
    p_min, _ = torch.min(p_hat, dim=1) # [batch_size, n_sample]
    return p_min.sum(dim=1).mean()     # scalar

def celoss(p_hat):
    """cross entropy loss"""
    crossentropy = - p_hat * torch.log(p_hat) # [batch_size, n_sample]
    return crossentropy.sum(dim=1).mean()     # scalar

def pairwise_dist(X, Y):
    """
    calculate pairwise l2 distance between X and Y
    """
    X_norm = (X**2).sum(dim=1).view(-1, 1)            # [n_xsample, 1]
    Y_t    = torch.transpose(Y, 0, 1)                 # [n_feature, n_ysample]
    Y_norm = (Y**2).sum(dim=1).view(1, -1)            # [1, n_ysample]
    dist   = X_norm + Y_norm - 2.0 * torch.mm(X, Y_t) # [n_xsample, n_ysample]
    return dist 

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

# def plot_acc_over_k():
#     with open("resultsacc/knn.txt", "r") as f:
#         data = [ [ float(d) for d in line.strip("\n").split(",") ] for line in f.readlines() ]

#     t = np.arange(len(data[0]))

#     # the steps and position
#     X = np.array(data).mean(0)

#     # the 1 sigma upper and lower analytic population bounds
#     lower_bound = np.array(data).min(0)
#     upper_bound = np.array(data).max(0)

#     fig, ax = plt.subplots(1)
#     ax.plot(t, X, lw=2, label='mean accuracy', color='blue')
#     # ax.plot(t, mu*t, lw=1, label='population mean', color='black', ls='--')
#     ax.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5,
#                     label='3 sigma range')
#     ax.legend(loc='upper left')

#     # here we use the where argument to only fill the region where the
#     # walker is above the population 1 sigma boundary
#     # ax.fill_between(t, upper_bound, X, where=X > upper_bound, facecolor='blue',
#     #                 alpha=0.5)
#     ax.set_xlabel('k')
#     ax.set_ylabel('accuracy')
#     ax.grid()
#     plt.show()