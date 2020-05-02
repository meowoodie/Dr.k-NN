#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import plots
import utils
import dataloader
import numpy as np
import robustclassifier as rc
from torchvision import datasets

def real_main():
    """main function for real data"""
    # model configurations
    classes     = [4, 6]
    n_class     = len(classes)
    n_feature   = 2
    n_sample    = 4 # 12
    max_theta   = 1e-2
    batch_size  = 10

    # init model
    model       = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    trainloader = dataloader.MiniSetLoader(
        datasets.MNIST("data", train=True, download=True), 
        classes, batch_size, n_sample, N=5)
    testloader  = dataloader.MiniSetLoader(
        datasets.MNIST("data", train=False, download=True), 
        classes, batch_size, n_sample, N=200)
    # trainloader.save_figures()
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # training
    rc.train(model, trainloader, testloader=testloader, n_iter=150, log_interval=5, lr=1e-2)
    rc.search_through(model, trainloader, testloader, K=5, h=8e-3)



def synthetic_main():
    """train function for synthetic data"""
    # model configurations
    classes     = [0, 1]
    n_class     = len(classes)
    n_feature   = 100
    n_sample    = 10 # 12
    max_theta   = 1e-2
    batch_size  = 10
    n_grid      = 100
    n_iter      = 20
    n_train     = 20

    means       = [[0, -2], [0, 2]]
    covs        = [
        [[4, 0.3], [0.3, 4]], 
        [[12, 1], [1, 1]]]

    # load synthetic dataset
    # dataset     = dataloader.SyntheticSwissrollDataset(N=500)
    dataset     = dataloader.SyntheticGaussianDataset(n_class, means, covs, N=1000)
    testloader  = dataloader.MiniSetLoader(dataset, classes, batch_size, n_sample, is_normalized=False, N=200)
    trainloader = dataloader.MiniSetLoader(dataset, classes, batch_size, n_sample, is_normalized=False, N=n_train)
    X_train, Y_train = trainloader.X, trainloader.Y
    X_test, Y_test   = testloader.X, testloader.Y
    # the observation space
    min_X, max_X, X = utils.evaluate_2Dspace(X_train, X_test, n_grid)

    # Train DR k-NN
    # init model
    model = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    rc.train(model, trainloader, testloader=testloader, n_iter=n_iter, log_interval=5, lr=1e-4)

    # DR k-NN results
    # - define robust classifier without neural networks
    model.eval()
    rclayer = rc.RobustClassifierLayer(n_class, 2 * n_train, n_feature)
    with torch.no_grad():
        Q       = utils.sortedY2Q(Y_train.unsqueeze(0))   # [1, n_class, n_sample]
        H_train = model.data2vec(X_train)                 # [n_train_sample, n_feature]
        H_test  = model.data2vec(X_test)                  # [n_test_sample, n_feature]
        H       = model.data2vec(X)                       # [n_grid * n_grid, n_feature]
        theta   = model.theta.data.unsqueeze(0)           # [1, n_class]
        p_hat   = rclayer(H_train.unsqueeze(0), Q, theta).data.squeeze(0) # [n_class, n_train_sample]
    # - perform classification for the space
    p_hat_knn = rc.knn_regressor(H, H_train, p_hat, K=5)      # [n_class, n_grid * n_grid]
    # p_hat_ks  = rc.kernel_smoother(H, H_train, p_hat, h=1e-2) # [n_class, n_grid * n_grid]
    # - visualization
    plots.visualize_2Dspace_2class(
        n_grid, max_X, min_X, p_hat_knn,
        X_train, Y_train, X_test, Y_test, prefix="drknn")
    # plots.visualize_2Dspace_2class(
    #     n_grid, max_X, min_X, p_hat_ks,
    #     X_train, Y_train, X_test, Y_test, prefix="kernel")

    # Naive k-NN results
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    # - define raw kNN model
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)
    print("knn", accuracy_score(predictions, Y_test))
    # - perform classification for the space
    pred       = knn.predict(X)
    space_pred = np.zeros((n_class, X.shape[0]))
    # - make pred as one-hot vector
    for i in range(X.shape[0]):
        space_pred[int(pred[i]), i] = 1
    space_pred = torch.Tensor(space_pred)
    # - visualization
    plots.visualize_2Dspace_2class(
        n_grid, max_X, min_X, space_pred, 
        X_train, Y_train, X_test, Y_test, prefix="naive-knn")




if __name__ == "__main__":
    synthetic_main()

    # test
    # Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # hs = [1e+5, 1e+4, 1e+3, 1e+2, 1e+1, 1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # knn_res    = []
    # kernel_res = []
    # for K, h in zip(Ks, hs):
    #     print("testing K=%d and h=%f" % (K, h))
    #     knn_acc, kernel_acc = rc.test(model, trainloader, testloader, K=K, h=h)
    #     knn_res.append(str(knn_acc))
    #     kernel_res.append(str(kernel_acc))
    # print("knn", ",".join(knn_res))
    # print("kernel", ",".join(kernel_res))



    # # save model
    # torch.save(model.state_dict(), "saved_model/mnist_cnn.pt")

    # # trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)