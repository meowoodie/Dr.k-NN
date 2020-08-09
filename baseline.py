#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import utils
import numpy as np
import dataloader
import robustclassifier as rc
from torchvision import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from sklearn.decomposition import TruncatedSVD, PCA
# from torchnca import NCA

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as colors


def test_phi_knn():
    torch.manual_seed(1234)

    # model configurations
    classes     = [0, 1]
    n_class     = len(classes)
    n_feature   = 2
    n_sample    = 5
    max_theta   = 1e-2
    batch_size  = 10

    # init model
    model       = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    trainloader = dataloader.MiniSetLoader(
        datasets.MNIST("data", train=True, download=True), 
        classes, batch_size, n_sample, N=10)
    testloader  = dataloader.MiniSetLoader(
        datasets.MNIST("data", train=False, download=True), 
        classes, batch_size, n_sample, N=200)
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # PCA transform to 2D space
    # pca = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
    # pca = PCA(n_components=2)
    # pca_X_test = pca.fit_transform(X_test)
    # pca_X_test = pca.transform(X_test)

    # training
    rc.train(model, trainloader, testloader=testloader, n_iter=10, log_interval=10, lr=1e-2)
    # rc.search_through(model, trainloader, testloader, K=5, h=8e-3)

    # testing
    X_test     = testloader.X.unsqueeze_(1).float()      # [ n_images, 1, 28, 28 ] 
    Y_test     = testloader.Y.float().numpy()            # [ n_images ] 
    phi_X_test = model.data2vec(X_test).detach().numpy() # [ n_images, n_feature ]
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    # knn.fit(phi_X_test, Y_test)
    predictions = knn.predict(phi_X_test)
    print("knn", accuracy_score(predictions, Y_test))

    

if __name__ == "__main__":
    test_phi_knn()