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
    This class is used to load signals through adequated methods.
    """

    def __init__(self):
        pass


class img(signals):
    r"""
    This class is used to load images through adequated methods.
    """

    def __init__(self, img=None):
        if img is not None:
            img = mpimg.imread(os.path.dirname(os.path.realpath(__file__)) +
                               '/signals/' + img)
            gray_scale = self.rgb2gray(img)
            self.gray_scale = gray_scale
        else:
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
        >>> img = signals.img()

        """

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def load_gray_img(self, pic):
        r"""
        Method to load an image as a signal and automatically convert it
        to gray scale

        Parameters
        --------
        pic : string
            path to the image to load

        Returns
        -------
        gray_scale : array_like
            The grayscale converted image.

        Examples
        --------
        >>> from pyunlocbox import signals
        >>> s = signals.img()

        """

        img = mpimg.imread(pic)
        gray_scale = self.rgb2gray(img)
        self.gray_scale = gray_scale


class lena(img):

    def __init__(self):
        super(lena, self).__init__('lena.png')


class whitecircle(img):

    def __init__(self):
        super(whitecircle, self).__init__('whitecircle.jpg')


class checkerboard(img):

    def __init__(self):
        super(checkerboard, self).__init__('checkerboard.png')
