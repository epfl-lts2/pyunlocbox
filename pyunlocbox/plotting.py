# -*- coding: utf-8 -*-

r"""
This module implements plotting functions :

* :meth:`plot_img` Plotting function to show images in grayscale

"""


def plot_img(img):
    r"""
    Just shows the image in argument

    Parameters
    ----------
    img : ndarray
        Image to show

    Examples
    --------
    >>> from pyunlocbox import signals, plotting
    >>> lena = signals.lena()
    >>> plotting.plot_img(lena.gray_scale)

    """

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.show()
    except:
        pass
