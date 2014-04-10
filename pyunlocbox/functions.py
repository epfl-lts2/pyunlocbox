#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements function objects which are then passed to solvers.
The *func* base class implements the interface whereas specialised classes
who inherit from it implement the methods. Theses classes include :

* :class:`norm_l2`: L2-norm which implements ``eval``, ``prox`` and ``grad``
"""

import numpy as np


class func:
    """
    This class defines a function object to be passed to solvers. It is
    intended to be a base class for standard functions which will implement
    the required methods. It can also be instantiated by user code and
    dynamically modified for rapid testing.

    Usage example :
    TODO (using doctest)
    """

    def eval(self, x):
        """
        Evaluation of the function at x.
        This method is used by all solvers.
        """
        raise NotImplementedError("Class user should define this method")

    def prox(self, x, T):
        """
        Proximal operator evaluated at x.
        This method is used by all proximal solvers.
        """
        raise NotImplementedError("Class user should define this method")

    def grad(self, x):
        """
        Function gradient evaluated at x.
        This method is only used by some solvers.
        """
        raise NotImplementedError("Class user should define this method")


class norm_l2(func):
    """
    This class defines the L2-norm function object to be passed to solvers.

    Parameters :

    * *lamb* : regularization parameter :math:`\lambda`
    * *weights* : weights for a weighted L2-norm (default 1)
    * *y* : measurements (default 0)
    * *A* : forward operator (default identity)
    * *At* : adjoint operator (default A)
    * *tight* : ``True`` if A is a tight frame,
      ``False`` otherwise (default True)
    * *nu* : bound on the norm of the operator A (default 1),
      i.e. :math:`||A x||^2 \leq \\nu ||x||^2`

    Usage example :
    TODO (using doctest)
    """

    def __init__(self, lamb, weights=1, y=0, A=None, At=None,
                 tight=True, nu=1):
        self.lamb = lamb
        self.weights = weights
        self.y = y
        if A:
            self.A = A
        else:
            self.A = lambda x: x
        if At:
            self.At = At
        else:
            At = A
        self.tight = tight
        self.nu = nu

    def eval(self, x):
        """
        Return :math:`\lambda ||A(x)-y||`
        """
        return self.lamb * np.linalg.norm(self.A(np.array(x)) - self.y)

    def prox(self, x, T):
        """
        L2-norm proximal operator. Return
        :math:`\min_{z} \\frac{1}{2} ||x - z||_2^2 + \gamma ||w(A z-y)||_2^2`
        where :math:`\gamma = \lambda \cdot T`
        """
        gamma = self.lamb * T
        if self.tight:
            sol = np.array(x) + gamma * 2 * self.At(self.y * self.weights**2)
            sol /= 1 + gamma * 2 * self.nu * self.weights**2
        else:
            raise NotImplementedError('Not implemented for non tight frame')
        return sol

    def grad(self, x):
        """
        Return :math:`2 \lambda A( A(x) - y )`
        """
        return 2 * self.lamb * self.A(self.A(np.array(x)) - self.y)
