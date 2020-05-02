#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Generative Adversarial Network

Reference:
- DCGAN PyTorch: https://github.com/pytorch/examples/blob/master/dcgan/main.py
"""

import nn
import utils
import torch
import arrow
import robustclassifier as rc
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.cm as cm
import matplotlib.pyplot as plt



def train(model, trainloader, batch_size, K=5, n_epoch=10, log_interval=10, dlr=1e-2, glr=1e-2, num_img=3):
    """training procedure"""
    optD = optim.Adam(model.netD.parameters(), lr=dlr)
    optG = optim.Adam(model.netG.parameters(), lr=glr)

    for epoch in range(n_epoch):
        for batch_idx, (X, Y) in enumerate(trainloader, 0):
            if X.shape[0] != batch_size:
                print("[%s] data in current batch are insufficient." % arrow.now())
                continue
            model.train()
            # train discriminator
            model.netD.zero_grad()
            p_hat, _ = model(X)
            lossD    = utils.celoss(p_hat)
            lossD.backward()
            optD.step()
            
            # train generator
            for k in range(K):
                model.netG.zero_grad()
                p_hat, _ = model(X)
                lossG    = - 1 * utils.celoss(p_hat)
                lossG.backward()
                optG.step()
            
            if batch_idx % log_interval == 0:
                print("[%s] Epoch: %d\tTrain batch: %d\tD Loss: %.3f,\tG Loss: %.3f" % \
                    (arrow.now(), epoch, batch_idx, lossD.item(), lossG.item()))
                generate(model, num_img)

def generate(model, num_img):
    """generate procedure"""
    model.eval()
    noise     = torch.randn(num_img, model.nz)
    fake_imgs = model.netG(noise).detach().numpy()
    cmap = cm.get_cmap('Greys')
    for img in fake_imgs:
        fig, ax = plt.subplots(1, 1)
        img     = img[0]
        implot  = ax.imshow(img, vmin=img.min(), vmax=img.max(), cmap=cmap)
        plt.axis('off')
        plt.savefig("results/fake_img_%s.pdf" % arrow.now(), bbox_inches='tight')
        plt.clf()



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
        self.netG      = nn.SimpleGenerator(n_feature=nz, out_channel=nc)
        self.netD      = nn.SimpleImage2Vec(n_feature=nz, in_channel=nc)
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
        noise  = torch.randn(X.shape[0], self.nz)
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

    nz         = 10 # size of feature vector
    ngz        = 5  # size of intermediate (generator) feature vector
    nc         = 1  # number of channels of images
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

    idx = get_indices(dataset, 4)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))

    # init model
    model = RobustGAN(nz, ngz, nc, n_sample)
    print("[%s]\n%s" % (arrow.now(), trainloader))

    # training
    train(model, trainloader, batch_size, K=5, n_epoch=10, log_interval=20, dlr=1e-5, glr=1e-2)