#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
This module implements signals objects which allow the user to load predefined
images.

* :class:`signals`: The base class of this module which contains a few methods
that loads predefined image (the :meth:`lena` method) and some tools methods
(the :meth:`rgb2gray` method)

"""

import os
import os.path
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class signals(object):
    r"""
    This class is used to load image through methods.

    Parameters
    ----------
    No parameters

    Examples
    --------
    Load the Lena picture and display it :

    >>> s = signals()
    >>> pic = s.lena()
    >>> plt.imshow(pic, cmap=plt.get_cmap('gray'))
    >>> plt.show()

    """

    def __init__(self):
        pass

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def lena(self):
        print os.path.dirname(os.path.realpath(__file__))
        img = mpimg.imread(os.path.dirname(os.path.realpath(__file__)) + '/signals/lena.png')
        gray_scale = self.rgb2gray(img)
        return gray_scale

# s = signals()
# pic = s.lena()
# plt.imshow(pic, cmap=plt.get_cmap('gray'))
# plt.show()
