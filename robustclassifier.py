#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Classifier

NOTE:
in my conda environment, cvxpy can only be installed via pip. 
if you need install cvxpy in conda, please ref to the reference below:
https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment

Dependencies:
- Python      3.7.6
- NumPy       1.18.1
- cvxpy       1.1.0a3
- PyTorch     1.4.0
- cvxpylayers 0.1.2
- arrow       0.13.1
"""

import nn
import arrow
import utils
import torch 
import cvxpy as cp
import numpy as np
# import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer

def tvloss(p_hat):
    """TV loss"""
    # p_max, _ = torch.max(p_hat, dim=1) # [batch_size, n_sample]
    # return p_max.sum(dim=1).mean()     # scalar
    p_min, _ = torch.min(p_hat, dim=1) # [batch_size, n_sample]
    return p_min.sum(dim=1).mean()     # scalar

def train(model, optimizer, trainloader, testloader=None, n_iter=100, log_interval=10):
    """training procedure for one epoch"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    for batch_idx, (X, Y) in enumerate(trainloader):
        model.train()
        Q = utils.sortedY2Q(Y)   # calculate empirical distribution based on labels
        optimizer.zero_grad()    # init optimizer (set gradient to be zero)
        p_hat = model(X, Q)      # inference
        loss  = tvloss(p_hat)    # calculate tv loss
        loss.backward()          # gradient descent
        optimizer.step()         # update optimizer
        if batch_idx % log_interval == 0:
            print("[%s] Train batch: %d\tLoss: %.3f" % (arrow.now(), batch_idx, loss.item()))
            # TODO: temporarily place test right here, will remove it in the end.
            if testloader is not None:
                test(model, trainloader, testloader, K=5)
        if batch_idx > n_iter:
            break
        

def test(model, trainloader, testloader, K):
    """
    calculate the pairwise distance between the data points from the test set and the train set, 
    respectively, and return the K-nearest neighbors from the train set for each data point in 
    the test set.

    input
    - model:       torch model
    - trainloader: X: (n_batch, [batch_size, n_sample, in_channel, n_pixel, n_pixel])
    - testloader:  x: (n_test_sample, [in_channel, n_pixel, n_pixel])
    output
    - k_neighbors: [n_test_sample, K]
    - H_train:     [n_train_sample, n_feature]
    """
    # calculate pairwise l2 distance between X and Y
    def pairwise_dist(X, Y):
        X_norm = (X**2).sum(dim=1).view(-1, 1)            # [n_xsample, 1]
        Y_t    = torch.transpose(Y, 0, 1)                 # [n_feature, n_ysample]
        Y_norm = (Y**2).sum(dim=1).view(1, -1)            # [1, n_ysample]
        dist   = X_norm + Y_norm - 2.0 * torch.mm(X, Y_t) # [n_xsample, n_ysample]
        return dist

    # given hidden embedding, evaluate corresponding p_hat using the output of the robust classifier layer
    def evaluate_p_hat(H, Q, theta):
        n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
        rbstclf = RobustClassifierLayer(n_class, n_sample, n_feature)
        return rbstclf(H, Q, theta).data

    # fetch data from trainset and testset
    X_train = trainloader.X.unsqueeze(1)                  # [n_train_sample, 1, n_pixel, n_pixel] 
    Y_train = trainloader.Y.unsqueeze(0)                  # [1, n_train_sample]
    X_test  = testloader.X.unsqueeze(1)                   # [n_test_sample, 1, n_pixel, n_pixel] 
    Y_test  = testloader.Y                                # [n_test_sample]

    # get H (embeddings) and p_hat for trainset and testset
    # and calculate p_hat
    model.eval()
    with torch.no_grad():
        Q       = utils.sortedY2Q(Y_train)                # [1, n_class, n_sample]
        H_train = model.img2vec(X_train)                  # [n_train_sample, n_feature]
        H_test  = model.img2vec(X_test)                   # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)           # [1, n_class]
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)    # [n_class, n_train_sample]
        
    # find the indices of k-nearest neighbor in trainset
    dist   = pairwise_dist(H_test, H_train)
    dist  *= -1
    _, knb = torch.topk(dist, K, dim=1)                   # [n_test_sample, K]

    # calculate the class marginal probability (p_hat) for each test sample
    p_hat_test = torch.stack(
        [ p_hat[:, neighbors].mean(dim=1) 
            for neighbors in knb ], dim=0).t()            # [n_class, n_test_sample]
    # calculate tv loss for test samples
    test_loss  = tvloss(p_hat_test.unsqueeze(0))
    # calculate accuracy
    test_pred  = p_hat_test.argmax(dim=0)
    n_correct  = test_pred.eq(Y_test).sum().item()
    accuracy   = n_correct / len(testloader)
        
    print("[%s] Test set: Average loss: %.3f, Accuracy: %.3f (%d samples)" % \
        (arrow.now(), test_loss, accuracy, len(testloader)))
    # return H_train, test_pred



class RobustImageClassifier(torch.nn.Module):
    """
    A Robust Image Classifier based on multiple CNNs and a Robust Classifier Layer defined below
    """

    def __init__(self, n_class, n_sample, n_feature, max_theta=0.1,
        n_pixel=28, in_channel=1, out_channel=7, 
        kernel_size=3, stride=1, keepprob=0.2):
        """
        Args:
        - n_class:     number of classes
        - n_sample:    number of sets of samples
        - n_feature:   size of the output feature (output of CNN)
        - max_theta:   threshold for theta_k (empirical distribution)
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
        super(RobustImageClassifier, self).__init__()
        # configurations
        self.n_class   = n_class
        self.max_theta = max_theta
        # Image to Vec layer
        self.img2vec   = nn.Image2Vec(n_feature, 
            n_pixel, in_channel, out_channel, kernel_size, stride, keepprob)
        # robust classifier layer
        # NOTE: if self.theta is a parameter, then it cannot be reassign with other values, 
        #       since it is one of the attributes defined in the model.
        self.theta     = torch.nn.Parameter(torch.ones(self.n_class).float() * self.max_theta)
        self.theta.requires_grad = True
        # self.theta     = torch.ones(self.n_class) * self.max_theta
        self.rbstclf   = RobustClassifierLayer(n_class, n_sample, n_feature)
    
    def forward(self, X, Q):
        """
        customized forward function.

        input
        - X:     [batch_size, n_sample, in_channel, n_pixel, n_pixel]
        - Q:     [batch_size, n_class, n_sample]
        output
        - p_hat: [batch_size, n_class, n_sample]
        - X:     [batch_size, n_sample, n_feature]
        """
        batch_size = X.shape[0]
        n_sample   = X.shape[1]

        # CNN layer
        # NOTE: merge the batch_size dimension and n_sample dimension
        X = X.view(batch_size*n_sample, 
            X.shape[2], X.shape[3], X.shape[4])       # [batch_size*n_sample, in_channel, n_pixel, n_pixel]
        X = self.img2vec(X)                           # [batch_size*n_sample, n_feature]
                                                      # NOTE: reshape back to batch_size and n_sample
        X = X.view(batch_size, n_sample, X.shape[-1]) # [batch_size, n_sample, n_feature]

        # robust classifier layer
        # theta = torch.ones(batch_size, self.n_class, requires_grad=True) * self.max_theta
        theta = self.theta.unsqueeze(0).repeat([batch_size, 1]) # [batch_size, n_class]
        p_hat = self.rbstclf(X, Q, theta)                       # [batch_size, n_class, n_sample]
        return p_hat



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
        X_tch is a single batch of the input data and Q_tch is the empirical distribution obtained from  
        the labels of this batch.

        input:
        - X_tch: [batch_size, n_sample, n_feature]
        - Q_tch: [batch_size, n_class, n_sample]
        - theta_tch: [batch_size, n_class]
        output:
        - p_hat: [batch_size, n_class, n_sample]
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
        cons = [ g >= 0. for g in gamma ]
        for k in range(n_class):
            cons += [cp.sum(cp.multiply(gamma[k], C)) <= theta[k]]
            for l in range(n_sample):
                cons += [cp.sum(gamma[k], axis=1)[l] == Q[k, l]]

        # Problem setup
        # obj   = cp.Minimize(cp.sum(cp.max(p, axis=0)))
        obj   = cp.Maximize(cp.sum(cp.min(p, axis=0)))
        prob  = cp.Problem(obj, cons)
        assert prob.is_dpp()

        # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
        # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
        # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
        return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)