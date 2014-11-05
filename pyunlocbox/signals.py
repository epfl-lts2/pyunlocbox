#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class signals(object):
    r"""
    TODO
    Parameters
    ----------
    TODO

    Examples
    --------
    TODO
    """

    def __init__(self, name):
        self.name = name

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def lena(self):
        img = mpimg.imread('/home/nicolas/Documents/dev/lts2/pyunlocbox/pyunlocbox/signals/lena.png')
        gray_scale = self.rgb2gray(img)
        return gray_scale

# s = signals("signal_1")
# pic = s.lena()
# plt.imshow(pic, cmap=plt.get_cmap('gray'))
# plt.show()
