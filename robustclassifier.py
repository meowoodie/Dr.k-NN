#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Classifier

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- cvxpy       1.1.0a3
- PyTorch     1.4.0
- cvxpylayers 0.1.2
"""

import utils
import torch 
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

# class RobustImageClassifier(torch.nn.Module):
#     """
#     A Robust Image Classifier based on CNN and Robust Classifier Layer defined below
#     """

#     def __init__(self, n_class, n_sample, n_feature):
#         """
#         """
#         super(RobustImageClassifier, self).__init__()

class RobustClassifierLayer(torch.nn.Module):
    """
    A Robust Classifier Layer via CvxpyLayer
    """

    def __init__(self, n_class, n_sample, n_feature):
        """
        Args:
        - n_class:  number of classes
        - n_sample: total number of samples in a single batch (including all classes)
        """
        super(RobustClassifierLayer, self).__init__()
        self.n_class, self.n_sample, self.n_feature = n_class, n_sample, n_feature
        self.cvxpylayer = self._cvxpylayer(n_class, n_sample)

    def forward(self, X_tch, Q_tch, theta_tch):
        """
        customized forward function. 
        X_tch is a single batch of the input data and Q_tch is the empirical distribution obtained from the 
        labels of this batch.
        input:
        - X_tch: [batch_size, n_sample, n_feature]
        - Q_tch: [batch_size, n_class, n_sample]
        - theta_tch: [batch_size, n_class]
        """
        C_tch     = self._wasserstein_distance(X_tch)        # [batch_size, n_sample, n_sample]
        gamma_hat = self.cvxpylayer(theta_tch, Q_tch, C_tch) # (n_class [batch_size, n_sample, n_sample])
        gamma_hat = torch.stack(gamma_hat, dim=1)            # [batch_size, n_class, n_sample, n_sample]
        p_hat     = gamma_hat.sum(dim=2)                     # [batch_size, n_class, n_sample]
        return p_hat

    @staticmethod
    def _wasserstein_distance(X):
        """
        the wasserstein distance for the input data via calculating the pairwise norm of two aribtrary 
        data points in the single batch of the input data, denoted as C here. 
        see reference below for pairwise distance calculation in torch:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        
        input
        - X: [batch_size, n_sample, n_feature]
        output
        - C_tch: [batch_size, n_sample, n_sample]
        """
        C_tch = []
        for x in X.split(split_size=1):
            x      = torch.squeeze(x, dim=0)                  # [n_sample, n_feature]
            x_norm = (x**2).sum(dim=1).view(-1, 1)            # [n_sample, 1]
            y_t    = torch.transpose(x, 0, 1)                 # [n_feature, n_sample]
            y_norm = x_norm.view(1, -1)                       # [1, n_sample]
            dist   = x_norm + y_norm - 2.0 * torch.mm(x, y_t) # [n_sample, n_sample]
            # Ensure diagonal is zero if x=y
            dist   = dist - torch.diag(dist)                  # [n_sample, n_sample]
            dist   = torch.clamp(dist, min=0.0, max=np.inf)   # [n_sample, n_sample]
            C_tch.append(dist)                                
        C_tch = torch.stack(C_tch, dim=0)                     # [batch_size, n_sample, n_sample]
        return C_tch

    @staticmethod
    def _cvxpylayer(n_class, n_sample):
        """
        construct a cvxpylayer that solves a robust classification problem
        see reference below for the binary case: 
        http://papers.nips.cc/paper/8015-robust-hypothesis-testing-using-wasserstein-uncertainty-sets
        """
        # NOTE: 
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
        return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)



def train(model, dataloader, optimizer, epoch):
    """train function"""
    model.train()
    for X, _, Q in enumerate(dataloader):
        
        # optimizer.zero_grad()
        # output = model(data)
        # loss = F.nll_loss(output, label)
        # loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

if __name__ == "__main__":
    batch_size = 32 
    n_class    = 2 
    n_sample   = 10
    n_feature  = 5

    # X_tch     = torch.randn(batch_size, n_sample, n_feature, requires_grad=True)
    # Q_tch     = torch.randn(batch_size, n_class, n_sample, requires_grad=True)
    # theta_tch = torch.randn(batch_size, n_class, requires_grad=True)

    # model = RobustClassifierLayer(n_class, n_sample, n_feature)
    # p_hat = model(X_tch, Q_tch, theta_tch)

    # print(p_hat)
    # print(model.parameters())

    

    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(batch_idx)
    #     print(target)

    for d, l in data:
        print(d)
