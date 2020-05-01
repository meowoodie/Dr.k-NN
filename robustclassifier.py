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
import plots
import cvxpy as cp
import numpy as np
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer

# TEST METHODS

def knn_regressor(H_test, H_train, p_hat_train, K=5):
    """
    k-Nearest Neighbor Regressor

    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the k-Nearest Neighbor rule.

    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    # find the indices of k-nearest neighbor in trainset
    dist   = utils.pairwise_dist(H_test, H_train)
    dist  *= -1
    _, knb = torch.topk(dist, K, dim=1)        # [n_test_sample, K]
    # calculate the class marginal probability (p_hat) for each test sample
    p_hat_test = torch.stack(
        [ p_hat_train[:, neighbors].mean(dim=1) 
            for neighbors in knb ], dim=0).t() # [n_class, n_test_sample]
    return p_hat_test

def kernel_smoother(H_test, H_train, p_hat_train, h=1e-1):
    """
    kernel smoothing test

    Given the train embedding and its corresponding optimal marginal distribution for each class,
    the function produce the prediction of p_hat for testing dataset given the test embedding based
    on the kernel smoothing rule with the bandwidth h.

    input
    - H_test:      [n_test_sample, n_feature]
    - H_train:     [n_train_sample, n_feature]
    - p_hat_train: [n_class, n_train_sample]
    output
    - p_hat_test:  [n_class, n_test_sample]
    """
    n_test_sample, n_feature = H_test.shape[0], H_test.shape[1]
    n_class, n_train_sample  = p_hat_train.shape[0], p_hat_train.shape[1]
    # calculate the pairwise distance between training sample and testing sample
    dist = utils.pairwise_dist(H_train, H_test)   # [n_train_sample, n_test_sample]
    # apply gaussian kernel
    G = 1 / ((np.sqrt(2*np.pi) * h) ** n_feature) * \
        torch.exp(- dist ** 2 / (2 * h ** 2))     # [n_train_sample, n_test_sample]
    G = G.unsqueeze(0).repeat([n_class, 1, 1])    # [n_class, n_train_sample, n_test_sample]
    p_hat_ext  = p_hat_train.unsqueeze(2).\
        repeat([1, 1, n_test_sample])             # [n_class, n_train_sample, n_test_sample]
    p_hat_test = (G * p_hat_ext).mean(dim=1)      # [n_class, n_test_sample]
    return p_hat_test

# GENERAL TRAIN PROCEDURE

def train(model, trainloader, testloader=None, n_iter=100, log_interval=10, lr=1e-2):
    """training procedure for one epoch"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for batch_idx, (X, Y) in enumerate(trainloader):
        model.train()
        Q = utils.sortedY2Q(Y)      # calculate empirical distribution based on labels
        optimizer.zero_grad()       # init optimizer (set gradient to be zero)
        p_hat = model(X, Q)         # inference
        loss  = utils.tvloss(p_hat) # calculate total variation loss
        # loss  = utils.celoss(p_hat) # calculate cross entropy loss
        loss.backward()             # gradient descent
        optimizer.step()            # update optimizer
        if batch_idx % log_interval == 0:
            print("[%s] Train batch: %d\tLoss: %.3f" % (arrow.now(), batch_idx, loss.item()))
            # TODO: temporarily place test right here, will remove it in the end.
            if testloader is not None:
                test(model, trainloader, testloader, K=5, h=1e-1)
        if batch_idx > n_iter:
            break
        
# GENERAL TEST PROCEDURE

def test(model, trainloader, testloader, K=5, h=1e-1):
    """testing procedure"""

    # given hidden embedding, evaluate corresponding p_hat 
    # using the output of the robust classifier layer
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
    model.eval()
    with torch.no_grad():
        Q       = utils.sortedY2Q(Y_train)                # [1, n_class, n_sample]
        H_train = model.data2vec(X_train)                 # [n_train_sample, n_feature]
        H_test  = model.data2vec(X_test)                  # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)           # [1, n_class]
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)    # [n_class, n_train_sample]
    # perform test
    p_hat_knn    = knn_regressor(H_test, H_train, p_hat, K)
    p_hat_kernel = kernel_smoother(H_test, H_train, p_hat, h)   
    # calculate accuracy
    knn_pred         = p_hat_knn.argmax(dim=0)
    knn_n_correct    = knn_pred.eq(Y_test).sum().item()
    knn_accuracy     = knn_n_correct / len(testloader)
    kernel_pred      = p_hat_kernel.argmax(dim=0)
    kernel_n_correct = kernel_pred.eq(Y_test).sum().item()
    kernel_accuracy  = kernel_n_correct / len(testloader)
    print("[%s] Test set: kNN accuracy: %.3f, kernel smoothing accuracy: %.3f (%d samples)" % (arrow.now(), knn_accuracy, kernel_accuracy, len(testloader)))
    return knn_accuracy, kernel_accuracy

# IMAGE CLASSIFIER

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
        self.n_feature = n_feature
        # Image to Vec layer
        self.data2vec  = nn.SimpleImage2Vec(n_feature, 
            in_channel, out_channel, n_pixel, kernel_size, stride, keepprob)
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
        Z = self.data2vec(X)                          # [batch_size*n_sample, n_feature]
                                                      # NOTE: reshape back to batch_size and n_sample
        Z = Z.view(batch_size, n_sample, Z.shape[-1]) # [batch_size, n_sample, n_feature]

        # robust classifier layer
        # theta = torch.ones(batch_size, self.n_class, requires_grad=True) * self.max_theta
        theta = self.theta.unsqueeze(0).repeat([batch_size, 1]) # [batch_size, n_class]
        p_hat = self.rbstclf(Z, Q, theta)                       # [batch_size, n_class, n_sample]
        return p_hat

# GENERAL CLASSIFIER

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
        # tv loss
        obj   = cp.Maximize(cp.sum(cp.min(p, axis=0)))
        # cross entropy loss
        # obj   = cp.Minimize(cp.sum(- cp.sum(p * cp.log(p), axis=0)))
        prob  = cp.Problem(obj, cons)
        assert prob.is_dpp()

        # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
        # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
        # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
        return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)

# OTHERS

def search_through(model, trainloader, testloader, n_grid=50, K=8, h=1e-1):
    """
    search through the embedding space, return the corresponding p_hat of a set of uniformly 
    sampled points in the space.

    NOTE: now it only supports 2D embedding space
    """
    # given hidden embedding, evaluate corresponding p_hat 
    # using the output of the robust classifier layer
    def evaluate_p_hat(H, Q, theta):
        n_class, n_sample, n_feature = theta.shape[1], H.shape[1], H.shape[2]
        rbstclf = RobustClassifierLayer(n_class, n_sample, n_feature)
        return rbstclf(H, Q, theta).data

    assert model.n_feature == 2
    # fetch data from trainset and testset
    X_train = trainloader.X.unsqueeze(1)                  # [n_train_sample, 1, n_pixel, n_pixel] 
    Y_train = trainloader.Y.unsqueeze(0)                  # [1, n_train_sample]
    X_test  = testloader.X.unsqueeze(1)                   # [n_test_sample, 1, n_pixel, n_pixel] 
    Y_test  = testloader.Y                                # [n_test_sample]
    # get H (embeddings) and p_hat for trainset and testset
    model.eval()
    with torch.no_grad():
        Q       = utils.sortedY2Q(Y_train)                # [1, n_class, n_sample]
        H_train = model.data2vec(X_train)                 # [n_train_sample, n_feature]
        H_test  = model.data2vec(X_test)                  # [n_test_sample, n_feature]
        theta   = model.theta.data.unsqueeze(0)           # [1, n_class]
        p_hat   = evaluate_p_hat(
            H_train.unsqueeze(0), Q, theta).squeeze(0)    # [n_class, n_train_sample]
    # uniformly sample points in the embedding space
    # - the limits of the embedding space
    min_H        = torch.cat((H_train, H_test), 0).min(dim=0)[0].numpy()
    max_H        = torch.cat((H_train, H_test), 0).max(dim=0)[0].numpy()
    min_H, max_H = min_H - (max_H - min_H) * .1, max_H + (max_H - min_H) * .1
    H_space      = [ np.linspace(min_h, max_h, n_grid + 1)[:-1] 
        for min_h, max_h in zip(min_H, max_H) ]           # (n_feature [n_grid])
    H            = [ [x, y] for x in H_space[0] for y in H_space[1] ]
    H            = torch.Tensor(H)                        # [n_grid * n_grid, n_feature]
    # perform test
    p_hat_knn    = knn_regressor(H, H_train, p_hat, K)    # [n_class, n_grid * n_grid]
    p_hat_kernel = kernel_smoother(H, H_train, p_hat, h)  # [n_class, n_grid * n_grid]

    plots.visualize_2Dspace_2class(
        n_grid, max_H, min_H, p_hat_knn,
        H_train, Y_train, H_test, Y_test, prefix="test")

    # # perform boundary knn test
    # _p_hat         = p_hat[0] / (p_hat[0] + p_hat[1])
    # print(_p_hat)
    # _indices       = ((_p_hat > 0.05) * (_p_hat < 0.95)).nonzero().flatten()
    # print(_indices)
    # bdry_p_hat     = p_hat[:, _indices]
    # bdry_H_train   = H_train[_indices]
    # bdry_Y_train   = Y_train[:, _indices]
    # print(bdry_p_hat.shape, bdry_H_train.shape)
    # bdry_p_hat_knn = knn_regressor(H, bdry_H_train, bdry_p_hat, 2)  # [n_class, n_grid * n_grid]

    # # plot the space and the training point
    # plots.visualize_2Dspace_LFD(
    #     n_grid, max_H, min_H, p_hat_kernel, 0,
    #     H_train, Y_train, H_test, Y_test, prefix="class0")
    # plots.visualize_2Dspace_LFD(
    #     n_grid, max_H, min_H, p_hat_kernel, 1,
    #     H_train, Y_train, H_test, Y_test, prefix="class1")
    # plots.visualize_2Dspace_LFD(
    #     n_grid, max_H, min_H, p_hat_kernel, 2,
    #     H_train, Y_train, H_test, Y_test, prefix="class2")
    # plots.visualize_2Dspace_Nclass(
    #     n_grid, max_H, min_H, p_hat_knn,
    #     H_train, Y_train, H_test, Y_test, prefix="knn")

    # plots.visualize_2Dspace_2class_boundary(
    #     n_grid, max_H, min_H, p_hat_knn,
    #     H_train, Y_train, H_test, Y_test, prefix="kernel_boundary")
    # plots.visualize_2Dspace_2class_boundary(
    #     n_grid, max_H, min_H, p_hat_kernel,
    #     H_train, Y_train, H_test, Y_test, prefix="knn_boundary")
    # plots.visualize_2Dspace_Nclass(
    #     n_grid, max_H, min_H, p_hat_knn,
    #     H_train, Y_train, H_test, Y_test, prefix="knn")
    # plots.visualize_2Dspace_Nclass(
    #     n_grid, max_H, min_H, bdry_p_hat_knn,
    #     bdry_H_train, bdry_Y_train, H_test, Y_test, prefix="knn_boundary_select")
