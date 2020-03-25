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
    n_sample    = 30
    n_feature   = 10
    max_theta   = 1e-2
    batch_size  = 10
    # training parameters
    epochs      = 5
    lr          = 1e-2
    gamma       = 0.7

    # init model
    model       = rc.RobustImageClassifier(n_class, n_sample, n_feature, max_theta)

    # # trainable parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # train and test
    trainloader = utils.Dataloader4mnist(classes, batch_size, n_sample)
    testloader  = utils.Dataloader4mnist(classes, batch_size, n_sample, is_train=False)
    print("[%s] train number of batches: %d, test number of batches: %d" % \
        (arrow.now(), len(trainloader), len(testloader)))
    optimizer   = optim.Adadelta(model.parameters(), lr=lr)
    scheduler   = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(epochs):
        rc.train(model, trainloader, optimizer, epoch, log_interval=5)
        rc.test(model, testloader)
        scheduler.step()
    
    torch.save(model.state_dict(), "saved_model/mnist_cnn.pt")

if __name__ == "__main__":
    main()