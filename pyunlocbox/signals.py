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
import matplotlib.image as mpimg


class signals(object):
    r"""
    This class is used to load image through adequated methods.
    """

    def __init__(self):
        pass

    def rgb2gray(self, rgb):
        r"""
        Python version of the rgb2gray() matlab method. Convert RGB image or
        colormap to grayscale.

        Parameters
        ----------
        rgb : array_like
            Input image.

        Returns
        -------
        gray_scale : array_like
            The grayscale converted image.

        Examples
        --------
        >>> from pyunlocbox import signals
        >>> import matplotlib.pyplot as plt
        >>> s = signals.signals()
        >>> pic = s.lena()
        >>> plt.imshow(pic, cmap=plt.get_cmap('gray')) #doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at 0x...>
        >>> plt.show(block=False)

        """
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def lena(self):
        img = mpimg.imread(os.path.dirname(os.path.realpath(__file__)) +
                           '/signals/lena.png')
        gray_scale = self.rgb2gray(img)
        return gray_scale
