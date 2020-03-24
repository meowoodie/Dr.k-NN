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
    n_sample    = 20
    n_feature   = 10
    max_theta   = 0.1
    batch_size  = 2
    # training parameters
    epochs      = 10
    lr          = 1e-3
    gamma       = 0.7

    # init dataloader
    dataloader = utils.dataloader4mnistNclasses(classes, batch_size, n_sample)
    # init model
    model     = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    print("yes")
    for epoch in range(1, epochs + 1):
        rc.train(epoch, model, optimizer, dataloader, log_interval=100)
        scheduler.step()

if __name__ == "__main__":
    main()