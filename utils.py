#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- PyTorch     1.4.0
"""

import torch 
import numpy as np

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
    

    
if __name__ == "__main__":
    batch_size, n_sample, n_feature = 2, 10, 5

    X = torch.randn(batch_size, n_sample, n_feature, requires_grad=True)
    Y = torch.randn(batch_size, n_sample, requires_grad=True)

    res = sortbyclass(X, Y)