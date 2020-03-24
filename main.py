#!/usr/bin/env python
# -*- coding: utf-8 -*-

import utils
import robustclassifier as rc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def main():
    """train function"""
    # model configurations
    classes     = [0, 1]
    n_class     = 2
    n_sample    = 50
    n_feature   = 10
    max_theta   = 1e-2
    batch_size  = 10
    # training parameters
    epochs      = 10
    lr          = 1e-3
    gamma       = 0.7

    # init dataloader
    # dataloader = utils.dataloader4mnistNclasses(classes, batch_size, n_sample)
    dataloader = utils.Dataloader4MNIST(classes, batch_size, n_sample)
    # init model
    model      = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    optimizer  = optim.Adadelta(model.parameters(), lr=lr)
    scheduler  = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        rc.train(epoch, model, optimizer, dataloader, log_interval=5)
        scheduler.step()

if __name__ == "__main__":
    main()