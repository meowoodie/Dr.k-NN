#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Classifier

Dependencies:
- Python  3.7.6
- NumPy   1.18.1
- cvxpy   1.1.0a3
- PyTorch 1.4.0
"""

import cvxpy as cp
import torch 
from cvxpylayers.torch import CvxpyLayer

class RobustClassifier(torch.nn.Module):
    """
    """

    def __init__(self, n_class, n_sample, n_feature):
        """
        Args:
        - n_class:  number of classes
        - n_sample: total number of samples in a single batch (including all classes)
        """
        super(RobustClassifier, self).__init__()
        self.cvxpylayer = self._cvxpylayer(n_class, n_sample)

    def forward(self, x):
        """customized forward function"""
        gamma_hat = self.cvxpylayer(theta_tch, Q_tch, C_tch) # (n_class [batch_size, n_sample, n_sample])
        gamma_hat = torch.stack(gamma_hat, axis=1)           # [batch_size, n_class, n_sample, n_sample]

    def _cvxpylayer(self, n_class, n_sample):
        """
        """
        # TODO: 
        # cvxpy currently doesn't support N-dim variables, see discussion and solution below:
        # * how to build N-dim variables?
        #   https://github.com/cvxgrp/cvxpy/issues/198
        # * how to stack variables?
        #   https://stackoverflow.com/questions/45212926/how-to-stack-variables-together-in-cvxpy 
        
        # Variables   
        # - gamma_k: the joint probability distribution on Omega^2 with marginal distribution Q_k and p_k
        gamma = [ cp.Variable((n_sample, n_sample)) for k in range(n_class) ]
        # - p_k: the marginal distribution of class k [n_class, n_sample]
        p     = [ cp.sum(gamma[k], axis=0) for k in range(n_class) ] 
        p     = cp.vstack(p) 

        # Parameters (indirectly from input data)
        # - theta: the threshold of wasserstein distance for each class
        theta = cp.Parameter(n_class)
        # - Q: the empirical distribution of class k obtained from the input label
        Q     = cp.Parameter((n_class, n_sample))
        # - C: the pairwise distance between data points
        C     = cp.Parameter((n_sample, n_sample))

        # Constraints
        cons = []
        for k in range(n_class):
            cons += [cp.sum(cp.multiply(gamma[k], C)) <= theta[k]]
            for l in range(n_sample):
                cons += [cp.sum(gamma[k], axis=1)[l] == Q[k, l]]

        # Problem setup
        obj   = cp.Minimize(cp.sum(cp.max(p, axis=0)))
        prob  = cp.Problem(obj, cons)
        assert prob.is_dpp()

        # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
        # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
        # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
        return CvxpyLayer(prob, 
            parameters=[theta, Q, C], 
            variables=gamma)








