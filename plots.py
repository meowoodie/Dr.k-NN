#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from numpy.ma import masked_array
from sklearn.manifold import TSNE

color_set = ["blue", "red", "green", "grey"]
cmap_set  = ["Blues", "Reds", "Greens", "Greys"]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """truncate colormap by proportion"""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def visualize_2Dspace_LFD(
    n_grid, max_H, min_H, p_hat_test, dim,
    H_train, Y_train, H_test, Y_test, prefix="test"):
    """
    visualize 2D embedding space and corresponding training data points.
    """
    n_class = p_hat_test.shape[0]
    assert n_grid * n_grid == p_hat_test.shape[1]
    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]
    # organize the p_hat as a matrix
    p_hat_test = p_hat_test.numpy()
    p_hat_show = p_hat_test[dim]
    p_hat_mat  = p_hat_show.reshape(n_grid, n_grid) # [n_grid, n_grid]
    # scale the training data to (0, n_grid)
    H_train = H_train.numpy()
    H_train = (H_train - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_train_sample, axis=0)
    H_train = np.nan_to_num(H_train) * n_grid
    # scale the testing data to (0, n_grid)
    H_test = H_test.numpy()
    H_test = (H_test - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_test_sample, axis=0)
    H_test = np.nan_to_num(H_test) * n_grid
    # prepare label set
    color_set = ["b", "r", "green"]
    Y_train   = Y_train.numpy()[0]
    Y_set     = list(set(Y_train))
    Y_set.sort()
    # plot the region
    fig, ax = plt.subplots(1, 1)
    if dim == 0:
        cmap = truncate_colormap(cm.get_cmap('Blues'), 0., 0.7) 
    elif dim == 1:
        cmap = truncate_colormap(cm.get_cmap('Reds'), 0., 0.7) 
    elif dim == 2:
        cmap = truncate_colormap(cm.get_cmap('Greens'), 0., 0.7) 
    implot  = ax.imshow(p_hat_mat, vmin=p_hat_mat.min(), vmax=p_hat_mat.max(), cmap=cmap)
    for c, y in zip(color_set, Y_set):
        Y_train_inds = np.where(Y_train == y)[0]
        Y_test_inds  = np.where(Y_test == y)[0]
        plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], s=40, c=c, linewidths="1", edgecolors="black")
        # plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], s=2, c=c, alpha=0.3)
    plt.axis('off')
    plt.savefig("results/%s_lfd_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()

def visualize_2Dspace_2class(
    n_grid, max_H, min_H, p_hat_test, 
    H_train, Y_train, H_test, Y_test, prefix="test"):
    """
    visualize 2D embedding space and corresponding training data points.
    """
    n_class = p_hat_test.shape[0]
    assert n_grid * n_grid == p_hat_test.shape[1]
    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]
    # organize the p_hat as a matrix
    p_hat_test = p_hat_test.numpy()
    p_hat_show = p_hat_test[0] / (p_hat_test[0] + p_hat_test[1])  # [n_grid * n_grid]
    p_hat_mat  = p_hat_show.reshape(n_grid, n_grid)               # [n_grid, n_grid]
    # scale the training data to (0, n_grid)
    H_train = H_train.numpy()
    H_train = (H_train - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_train_sample, axis=0)
    H_train = np.nan_to_num(H_train) * n_grid
    # scale the testing data to (0, n_grid)
    H_test = H_test.numpy()
    H_test = (H_test - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_test_sample, axis=0)
    H_test = np.nan_to_num(H_test) * n_grid
    # prepare label set
    color_set = ["b", "r"]
    Y_train   = Y_train.numpy().flatten()
    Y_set     = list(set(Y_train))
    Y_set.sort()
    # plot the region
    fig, ax = plt.subplots(1, 1)
    cmap    = truncate_colormap(cm.get_cmap('RdBu'), 0.3, 0.7)
    implot  = ax.imshow(p_hat_mat, vmin=p_hat_mat.min(), vmax=p_hat_mat.max(), cmap=cmap)
    for c, y in zip(color_set, Y_set):
        Y_train_inds = np.where(Y_train == y)[0]
        Y_test_inds  = np.where(Y_test == y)[0]
        plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], s=30, c=c, linewidths="1", edgecolors="black")
        plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], s=5, c=c, alpha=0.3)
    plt.axis('off')
    plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()

def visualize_2Dspace_2class_boundary(
    n_grid, max_H, min_H, p_hat_test,
    H_train, Y_train, H_test, Y_test, prefix="test"):
    """
    visualize 2D embedding space and corresponding training data points.
    """
    n_class = p_hat_test.shape[0]
    assert n_grid * n_grid == p_hat_test.shape[1]
    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]
    # organize the p_hat as a matrix
    p_hat_test = p_hat_test.numpy()
    p_hat_show = p_hat_test[0] / (p_hat_test[0] + p_hat_test[1])  # [n_grid * n_grid]
    p_hat_mat  = p_hat_show.reshape(n_grid, n_grid)               # [n_grid, n_grid]
    # select marginal points
    p_hat_mat  = (1 - abs( p_hat_mat - 0.5 )) # * (p_hat_mat > 0.1) * (p_hat_mat < 0.9)
    # scale the training data to (0, n_grid)
    H_train = H_train.numpy()
    H_train = (H_train - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_train_sample, axis=0)
    H_train = np.nan_to_num(H_train) * n_grid
    # scale the testing data to (0, n_grid)
    H_test = H_test.numpy()
    H_test = (H_test - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_test_sample, axis=0)
    H_test = np.nan_to_num(H_test) * n_grid
    # prepare label set
    color_set = ["b", "r"]
    Y_train   = Y_train.numpy()[0]
    Y_set     = list(set(Y_train))
    Y_set.sort()
    # plot the region
    fig, ax = plt.subplots(1, 1)
    cmap    = truncate_colormap(cm.get_cmap('Greys'), 0., 0.6)
    implot  = ax.imshow(p_hat_mat, vmin=p_hat_mat.min(), vmax=p_hat_mat.max(), cmap=cmap)
    # plot the points
    for c, y in zip(color_set, Y_set):
        Y_train_inds = np.where(Y_train == y)[0]
        Y_test_inds  = np.where(Y_test == y)[0]
        plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], s=30, c=c, linewidths="1", edgecolors="black")
        plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], s=5, c=c, alpha=0.3)
    # plot the contour
    cp = ax.contour(list(range(n_grid)), list(range(n_grid)), p_hat_mat, 
        levels=[0.6, 0.8, 0.9, 0.95])
    ax.clabel(cp, inline=True, fontsize=10)
    plt.axis('off')
    plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()

def visualize_2Dspace_Nclass(
    n_grid, max_H, min_H, p_hat_test, 
    H_train, Y_train, H_test, Y_test, prefix="test"):
    """
    visualize 2D embedding space and corresponding training data points.
    """
    n_class = p_hat_test.shape[0]
    assert n_grid * n_grid == p_hat_test.shape[1]
    n_train_sample = H_train.shape[0]
    n_test_sample  = H_test.shape[0]
    # organize the p_hat as multiple matrices for each class
    p_hat_test = p_hat_test.numpy().reshape(n_class, n_grid, n_grid)
    p_hat_max  = p_hat_test.argmax(0) # [n_grid, n_grid]
    p_hat_mats = []                   # (n_class [n_grid, n_grid]) 
    for i in range(n_class):
        p_hat_show = p_hat_test[i] / p_hat_test.sum(0)
        p_hat_mat  = masked_array(p_hat_show, p_hat_max != i)
        p_hat_mats.append(p_hat_mat)
    # scale the training data to (0, n_grid)
    H_train = H_train.numpy()
    H_train = (H_train - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_train_sample, axis=0)
    H_train = np.nan_to_num(H_train) * n_grid
    # scale the testing data to (0, n_grid)
    H_test = H_test.numpy()
    H_test = (H_test - min_H) /\
        np.repeat(np.expand_dims(max_H - min_H, 0), n_test_sample, axis=0)
    H_test = np.nan_to_num(H_test) * n_grid
    # prepare label set
    Y_train   = Y_train.numpy()[0]
    Y_set     = list(set(Y_train))
    Y_set.sort()
    # plot the region
    fig, ax = plt.subplots(1, 1)
    cmaps   = [ 
        truncate_colormap(cm.get_cmap(cmap), 0., 0.7) 
        for cmap in cmap_set[:n_class] ]
    implots = [ 
        ax.imshow(p_hat_mats[i], vmin=p_hat_mats[i].min(), vmax=p_hat_mats[i].max(), cmap=cmaps[i]) 
        for i in range(n_class) ]
    # plot the points
    for c, y in zip(color_set[:n_class], Y_set):
        Y_train_inds = np.where(Y_train == y)[0]
        Y_test_inds  = np.where(Y_test == y)[0]
        plt.scatter(H_train[Y_train_inds, 1], H_train[Y_train_inds, 0], 
            s=30, c=c, linewidths="1", edgecolors="black")
        plt.scatter(H_test[Y_test_inds, 1], H_test[Y_test_inds, 0], 
            s=5, c=c, alpha=0.3)
    plt.axis('off')
    plt.savefig("results/%s_map_%s.pdf" % (prefix, arrow.now()), bbox_inches='tight')
    plt.clf()

def visualize_embedding(H, p_hat, useTSNE=True, perplexity=20):
    """
    visualize data embedding on a 2D space using TSNE. 
    
    input
    - H:     [n_sample, n_feature]
    - p_hat: [n_class, n_sample]
    """
    # configuration
    n_class  = p_hat.shape[0] 
    n        = H.shape[0]
    H        = H.numpy()
    p_hat    = p_hat.numpy()
    # check data dimension
    assert useTSNE is True or H.shape[1] == 2
    # fit TSNE
    if useTSNE:
        tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity)
        E2D  = tsne.fit_transform(H)
    else:
        E2D  = H

    # plot 
    fig, axs = plt.subplots(1, n_class)
    # ax 
    # ax1      = axs[0]
    # ax2      = axs[1]
    # plot embedding colored by their labels
    # cm1 = plt.cm.get_cmap('Reds')
    # cm2 = plt.cm.get_cmap('Blues')
    cms = [ plt.cm.get_cmap(c) for c in ['Reds', 'Blues', 'Greens'] ]

    for i in range(n_class):
        axs[i].scatter(E2D[:, 0], E2D[:, 1], c=p_hat[i, :], vmin=p_hat[i, :].min(), vmax=p_hat[i, :].max(), cmap=cms[i])
    plt.savefig("results/scatter_%s.pdf" % arrow.now())
    plt.clf()

    # # plot 
    # fig, axs = plt.subplots(1, 2)
    # cm       = plt.cm.get_cmap('RdYlBu')
    # ax1      = axs[0]
    # ax2      = axs[1]
    # # plot embedding colored by their labels
    # ax1.scatter(E2D[:int(n/2), 0], E2D[:int(n/2), 1], c="b")
    # ax1.scatter(E2D[int(n/2):, 0], E2D[int(n/2):, 1], c="r")
    # # plot embedding colored by p_hat
    # p_hat = p_hat[0] / (p_hat[0] + p_hat[1])
    # ax2.scatter(E2D[:, 0], E2D[:, 1], c=p_hat, vmin=0, vmax=1, cmap=cm)
    # plt.savefig("results/scatter_%s.pdf" % arrow.now()) 