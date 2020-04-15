#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Generative Adversarial Network
"""

import nn
import utils
import torch
import arrow
import robustclassifier as rc
import torch.optim as optim
from torchvision import datasets, transforms


def train(model, trainloader, n_epoch=10, log_interval=10, lr=1e-2):
    """training procedure for one epoch"""
    optD = optim.Adam(model.netD.parameters(), lr=lr)
    optG = optim.Adam(model.netG.parameters(), lr=lr)

    for epoch in range(n_epoch):
        for batch_idx, (X, Y) in enumerate(trainloader, 0):
            # train discriminator
            model.netD.zero_grad()
            p_hat, _ = model(X)
            lossD    = utils.tvloss(p_hat)
            lossD.backward()
            optD.step()
            
            # train generator
            model.netG.zero_grad()
            p_hat, _ = model(X)
            lossG    = - 1 * utils.tvloss(p_hat)
            lossG.backward()
            optG.step()
            
            if batch_idx % log_interval == 0:
                print("[%s] Train batch: %d\tD Loss: %.3f,\tG Loss: %.3f" % \
                    (arrow.now(), batch_idx, lossD.item(), lossG.item()))



class RobustGAN(torch.nn.Module):
    """
    A Robust Generative Adversarial Network
    """

    def __init__(self, nz, ngz, nc, n_sample, max_theta=0.1):
        super(RobustGAN, self).__init__()
        self.nz        = nz
        self.nc        = nc
        self.n_sample  = n_sample
        # networks of discriminator and generator
        self.netG      = nn.Generator(nz, ngz, nc)
        self.netD      = nn.Image2Vec(nc, nz)
        # robust classifier layer
        # NOTE: if self.theta is a parameter, then it cannot be reassign with other values, 
        #       since it is one of the attributes defined in the model.
        self.theta     = torch.nn.Parameter(torch.ones(2).float() * max_theta)
        self.theta.requires_grad = True
        self.robustclf = rc.RobustClassifierLayer(2, n_sample, nz)

    def forward(self, X):
        """
        input:
        - X:     [batch_size, nc, n_pixel, n_pixel]
        - noise: [batch_size, nz, 1, 1]
        """
        # random noise and its fake generation
        noise  = torch.randn(X.shape[0], self.nz, 1, 1)
        fake_X = self.netG(noise)
        # discriminator loss
        p_hat  = self.D(X, fake_X)
        return p_hat, fake_X
 
    def D(self, true_X, fake_X):
        """
        input:
        - true_X, fake_X: [batch_size, nc, n_pixel, n_pixel]
        """
        batch_size, nc, n_pixel, n_pixel = true_X.shape
        n_halfsample = int(self.n_sample / 2)
        n_minibatch  = int(batch_size / n_halfsample)
        # prepare feature embeddings
        fake_X = fake_X.view(n_minibatch, n_halfsample, 1, n_pixel, n_pixel)
        true_X = true_X.view(n_minibatch, n_halfsample, 1, n_pixel, n_pixel)
        X      = torch.cat((fake_X, true_X), 1).view(n_minibatch * self.n_sample, 1, n_pixel, n_pixel)
        Z      = self.netD(X).view(n_minibatch, self.n_sample, -1)
        # prepare labels
        fake_Y = torch.zeros((n_minibatch, n_halfsample))
        true_Y = torch.ones((n_minibatch, n_halfsample))
        Y      = torch.cat((fake_Y, true_Y), 1)
        Q      = utils.sortedY2Q(Y)
        # discriminator output
        theta  = self.theta.unsqueeze(0).repeat([n_minibatch, 1]) # [batch_size, n_class]
        p_hat  = self.robustclf(Z, Q, theta)
        return p_hat

    
    
if __name__ == "__main__":
    # batch_size  = 16
    # in_channel  = 1
    # nz          = 10
    # ngz         = 5
    # out_channel = 1

    # img2vec = nn.Image2Vec(in_channel, nz)
    # vec2img = nn.Generator(nz, ngz, out_channel)

    # # x       = torch.randn(batch_size, in_channel, 28, 28)
    # # print(x.shape)
    # # z       = img2vec(x)
    # # print(z.shape)

    # z_hat   = torch.randn(batch_size, nz, 1, 1)
    # x_hat   = vec2img(z_hat)
    # print(x_hat.shape)

    nz         = 10
    nc         = 1
    ngz        = 5
    n_sample   = 10
    batch_size = 100

    def get_indices(dataset, class_name):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_name:
                indices.append(i)
        return indices

    dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    idx = get_indices(dataset, 1)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))

    # init model
    model = RobustGAN(nz, ngz, nc, n_sample)
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # training
    train(model, trainloader, n_epoch=10, log_interval=5, lr=1e-2)
