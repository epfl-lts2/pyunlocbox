# -*- coding: utf-8 -*-

r"""
The :mod:`pyunlocbox.operators` module implements the following operators:

.. autosummary::

    grad
    div

"""

import numpy as np


def grad(x, dim=2, **kwargs):
    r"""
    Returns the gradient of an array.

    Parameters
    ----------
    dim : int
        Dimension of the grad
    wx :  int
    wy :  int
    wz :  int
    wt :  int
        Weights to apply on each axis

    Returns
    -------
    dx, dy, dz, dt : ndarrays
        Gradients following each axes,
        only the necessary ones are returned

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import operators
    >>> x = np.arange(16).reshape(4, 4)
    >>> dx, dy = operators.grad(x)

    """

    WEIGHT_KEYS = ("wx", "wy", "wz", "wt")
    diffs = []
    for axis in range(dim):
        zeros_shape = list(x.shape)
        zeros_shape[axis] = 1

        diff = np.concatenate((np.diff(x, axis=axis),
                               np.zeros(zeros_shape)),
                              axis=axis)
        try:
            diff *= kwargs[WEIGHT_KEYS[axis]]
        except (KeyError, TypeError):
            pass
        diffs.append(diff)

    if dim == 1:
        return diffs[0]
    else:
        return diffs


def div(*args, **kwargs):
    r"""
    Returns the divergence of an array.

    Parameters
    ----------
    dx :  array_like
    dy :  array_like
    dz :  array_like
    dt :  array_like
        Arrays to operate on

    Returns
    -------
    x : array_like
        Divergence vector

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import operators
    >>> x = np.arange(16).reshape(4, 4)
    >>> dx, dy = operators.grad(x)
    >>> divx = operators.div(dx, dy)

    """

    if len(args) == 0:
        raise ValueError("Need to input at least one value")

    if len(args) >= 1:
        dx = args[0]
        try:
            dx *= np.conjugate(kwargs["wx"])
        except KeyError:
            pass

        x = np.concatenate((np.expand_dims(dx[0, ], axis=0),
                            dx[1:-1, ] - dx[:-2, ],
                            -np.expand_dims(dx[-2, ], axis=0)),
                           axis=0)

    if len(args) >= 2:
        dy = args[1]
        try:
            dy *= np.conjugate(kwargs["wy"])
        except KeyError:
            pass

        x += np.concatenate((np.expand_dims(dy[:, 0, ], axis=1),
                             dy[:, 1:-1, ] - dy[:, :-2, ],
                             -np.expand_dims(dy[:, -2, ], axis=1)),
                            axis=1)

    if len(args) >= 3:
        dz = args[2]
        try:
            dz *= np.conjugate(kwargs["wz"])
        except KeyError:
            pass

        x += np.concatenate((np.expand_dims(dz[:, :, 0, ], axis=2),
                             dz[:, :, 1:-1, ] - dz[:, :, :-2, ],
                             -np.expand_dims(dz[:, :, -2, ], axis=2)),
                            axis=2)

    if len(args) >= 4:
        dt = args[3]
        try:
            dt *= np.conjugate(kwargs["wt"])
        except KeyError:
            pass

        x += np.concatenate((np.expand_dims(dt[:, :, :, 0, ], axis=3),
                             dt[:, :, :, 1:-1, ] - dt[:, :, :, :-2, ],
                             -np.expand_dims(dt[:, :, :, -2, ], axis=3)),
                            axis=3)
    return x
