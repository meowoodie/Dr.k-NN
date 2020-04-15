#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import plots
import utils
import dataloader
import robustclassifier as rc

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
    p_hat_knn    = rc.knn_regressor(H, H_train, p_hat, K)    # [n_class, n_grid * n_grid]
    p_hat_kernel = rc.kernel_smoother(H, H_train, p_hat, h)  # [n_class, n_grid * n_grid]

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

def main():
    """train function"""
    # model configurations
    classes     = [0, 1]
    n_class     = len(classes)
    n_feature   = 2
    n_sample    = 6 # 12
    max_theta   = 1e-2
    batch_size  = 10
    # training parameters
    epochs      = 2
    
    gamma       = 0.7

    # init model
    model       = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    trainloader = dataloader.MiniMnist(classes, batch_size, n_sample, is_train=True, N=10)
    testloader  = dataloader.MiniMnist(classes, batch_size, n_sample, is_train=False, N=200)
    trainloader.save_figures()
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # training
    rc.train(model, trainloader, testloader=testloader, n_iter=150, log_interval=5, lr=1e-2)
    # search_through(model, trainloader, testloader, K=5, h=8e-3)
    
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

if __name__ == "__main__":
    main()