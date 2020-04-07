#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import dataloader
import robustclassifier as rc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def main():
    """train function"""
    # model configurations
    classes     = [0, 1]
    n_class     = len(classes)
    n_feature   = 2
    n_sample    = 12
    max_theta   = 1e-2
    batch_size  = 10
    # training parameters
    epochs      = 2
    lr          = 1e-2
    gamma       = 0.7

    # init model
    model       = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    trainloader = dataloader.MiniMnist(classes, batch_size, n_sample, is_train=True, N=15)
    testloader  = dataloader.MiniMnist(classes, batch_size, n_sample, is_train=False, N=100)
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # training
    optimizer   = optim.Adadelta(model.parameters(), lr=lr)
    scheduler   = StepLR(optimizer, step_size=1, gamma=gamma)
    rc.train(model, optimizer, trainloader, testloader, n_iter=100, log_interval=5)
    
    # save model
    torch.save(model.state_dict(), "saved_model/mnist_cnn.pt")

    # # trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

if __name__ == "__main__":
    main()