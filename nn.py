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
        n_pixel=28, in_channel=1, out_channel=7, kernel_size=3, stride=1, keepprob=0.2):
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



class Image2Vec(torch.nn.Module):
    """
    Convert an image into a feature vector using CNNs
    """
    def __init__(self, in_channel, nz):
        """
        Args:
        - nz:         size of the discriminative feature
        - in_channel: number of in channels for an image
        """
        super(Image2Vec, self).__init__()
        self.conv = torch.nn.Sequential(
            # input is (in_channel) x 64 x 64 / 28 x 28
            torch.nn.Conv2d(in_channel, nz, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (nz) x 32 x 32 / 14 x 14
            torch.nn.Conv2d(nz, nz * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nz * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (nz*2) x 16 x 16 / 7 x 7
            torch.nn.Conv2d(nz * 2, nz * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nz * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (nz*4) x 8 x 8 / 3 x 3
            torch.nn.Conv2d(nz * 4, nz * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(nz * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # # state size. (nz*8) x 4 x 4 / 1 x 1
        )
        self.fc = torch.nn.Linear((nz*8) * 1 * 1, nz)

    def forward(self, X):
        """
        customized forward function.

        input
        - Z:     [batch_size, nz]
        output
        - X:     [batch_size, out_channel, 64, 64]
        """
        X_tilde = self.conv(X)
        X_tilde = torch.flatten(X_tilde, 1)
        Z       = self.fc(X_tilde)
        Z       = F.relu(Z)
        return Z


    
class Generator(torch.nn.Module):
    """
    Generator

    Restore an image from a feature vector using transposed convolution
    """

    def __init__(self, nz, ngz, out_channel):
        """
        Args:
        - nz:          size of the input feature (input of transposed CNN)
        - ngz:         size of the generative feature
        - out_channel: number of out channels from transposed CNN
        """
        super(Generator, self).__init__()
        self.trans_conv = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(nz, ngz * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngz * 8),
            torch.nn.ReLU(True),
            # state size. (ngz*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngz * 8, ngz * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngz * 4),
            torch.nn.ReLU(True),
            # state size. (ngz*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngz * 4, ngz * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngz * 2),
            torch.nn.ReLU(True),
            # state size. (ngz) x 32 x 32
            torch.nn.ConvTranspose2d(ngz * 2, out_channel, 15, 1, 1, bias=False),
            torch.nn.Tanh()
            # state size. (out_channel) x 64 x 64
        )
        # # state size. (ngz*2) x 16 x 16
        # torch.nn.ConvTranspose2d(ngz * 2, ngz, 4, 2, 1, bias=False),
        # torch.nn.BatchNorm2d(ngz),
        # torch.nn.ReLU(True),
        # # state size. (ngz) x 32 x 32
        # torch.nn.ConvTranspose2d(ngz, out_channel, 4, 2, 1, bias=False),
        # torch.nn.Tanh()
        # # state size. (out_channel) x 64 x 64

    def forward(self, Z):
        """
        customized forward function.

        input
        - Z:     [batch_size, n_feature]
        output
        - X:     [batch_size, out_channel, 64, 64]
        """
        X = self.trans_conv(Z)
        return X