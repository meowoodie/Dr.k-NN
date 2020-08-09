#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script defines multiple neural networks for the embedding purpose.
References:
- https://github.com/pytorch/examples/blob/master/dcgan/main.py#L157
- 
"""

import arrow
import utils
import torch 
import cvxpy as cp
import numpy as np
import torch.nn.functional as F

class SimpleImage2Vec(torch.nn.Module):
    """
    Convert a simple image into a feature vector using CNNs
    """

    def __init__(self, n_feature, 
        in_channel=1, out_channel=7, n_pixel=28, kernel_size=3, stride=1, keepprob=0.2):
        """
        Args:
        - n_feature:   size of the output feature (output of CNN)
        - n_pixel:     number of pixels along one axis for an image (n_pixel * n_pixel) 
                       28 in default (mnist)
        - in_channel:  number of in channels for an image
                       1 in default (mnist)
        - out_channel: number of out channels from CNN
                       5 in default
        - kernel_size: size of the kernel in CNN
                       3 in default
        - stride:      the stride for the cross-correlation, a single number or a tuple.
                       1 in default
        - keepprob:    keep probability for dropout layer
                       0.2 in default
        """
        super(SimpleImage2Vec, self).__init__()
        # CNN layer 1
        self.conv1     = torch.nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=kernel_size, 
            stride=stride, padding=0)
        # dropout layer 1
        self.dropout1  = torch.nn.Dropout2d(keepprob)
        # fully connected layer 1
        n_feature_1    = int((n_pixel - kernel_size + stride) / stride)
        n_feature_2    = int(n_feature_1 / 2)
        self.fc1       = torch.nn.Linear(out_channel * (n_feature_2**2), n_feature)
    
    def forward(self, X):
        """
        customized forward function.
        input
        - X:     [batch_size, in_channel, n_pixel, n_pixel]
        output
        - Z:     [batch_size, n_feature]
        """
        # CNN layer
        X = self.conv1(X)                       # [batch_size, out_channel, n_feature_1, n_feature_1] 
        X = F.relu(X)
                                                # NOTE: n_feature_2 = n_feature_1 / kernel_size=2
        X = F.max_pool2d(X, 2)                  # [batch_size, out_channel, n_feature_2, n_feature_1] 
        X = self.dropout1(X)
        X = torch.flatten(X, 1)                 # [batch_size, out_channel*n_feature_2*n_feature_2] 

        # fully-connected layer
        X = self.fc1(X)
        Z = F.relu(X)                           # [batch_size, n_feature]
        return Z