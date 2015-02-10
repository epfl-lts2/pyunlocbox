#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_img(img):
    plt.ion()
    fig = plt.figure()
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
