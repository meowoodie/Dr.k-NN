#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import plots
import utils
import dataloader
import robustclassifier as rc
from torchvision import datasets

def main():
    """train function"""
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