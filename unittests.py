#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unittests for key components

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- PyTorch     1.4.0
"""

import utils
import torch 
import numpy as np
import robustclassifier

def unittest_3():
    """
    UNITTEST 3
    - func: robustclassifier.RobustClassifierLayer
    """
    batch_size, n_sample, n_feature, n_class = 5, 10, 7, 2
    X_tch     = torch.randn(batch_size, n_sample, n_feature, requires_grad=True)
    Q_tch     = torch.randn(batch_size, n_class, n_sample, requires_grad=True)
    theta_tch = torch.randn(batch_size, n_class, requires_grad=True)

    model = robustclassifier.RobustClassifierLayer(n_class, n_sample, n_feature)
    p_hat = model(X_tch, Q_tch, theta_tch)

    print(p_hat)
    print(model.parameters())

def unittest_2():
    """
    UNITTEST 2
    - func: utils.dataloader4mnistNclasses
    """
    classes    = [1, 2]
    batch_size = 20 
    n_sample   = 50
    dataloader = utils.Dataloader4MNIST(classes, batch_size, n_sample)
    print(len(dataloader))
    for batch_idx, (X, Y, Q) in enumerate(dataloader):
        print(batch_idx)
        break

def unittest_1():
    """
    UNITTEST 1 
    - func: utils.sortbyclass
    - func: utils.sortedY2Q

    Expected output
    tensor([
        [[0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],

        [[0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500, 0.2500]],

        [[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333]],

        [[0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000]]])
    """
    batch_size, n_sample, n_feature = 4, 7, 5
    X = torch.randn(batch_size, n_sample, n_feature, requires_grad=True)
    Y = torch.tensor([
        [2,1,1,2,2,2,2], # 1: 2, 2: 5
        [1,2,1,2,1,2,2], # 1: 3, 2: 4
        [1,1,2,1,2,1,2], # 1: 4, 2: 3
        [1,2,2,2,2,1,2]  # 1: 2, 2: 5
    ])
    _X, _Y = utils.sortbyclass(X, Y)
    Q      = utils.sortedY2Q(_Y)
    print(Q)

if __name__ == "__main__":
    # unittest_1()
    unittest_2()
    # unittest_3()