#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements function objects which are then passed to solvers.
The :class:`func` base class implements the interface whereas specialised
classes who inherit from it implement the methods. These classes include :

* :class:`norm_l2`: L2-norm which implements :func:`eval`, :func:`prox`
  and :func:`grad`
* :class:`norm_l1`: L1-norm which implements :func:`eval` and :func:`prox`
"""

import numpy as np


def _soft_threshold(z, T):
    """
    Return the soft thresholded signal.

    Parameters :
    * *z* : input signal (real or complex)
    * *T* : threshold on the absolute value of ``z``
    """
    z = np.array(z)
    sol = np.maximum(abs(z)-T*abs(z), 0) * z
    sol /= np.maximum(abs(z)-T*abs(z), 0) + T*abs(z) + (abs(z) == 0)
    return sol


class func:
    """
    This class defines a function object to be passed to :func:`solve`. It is
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
        raise NotImplementedError("Class user should define this method.")

    def prox(self, x, T):
        """
        Proximal operator evaluated at x :
        :math:`prox_{f,\gamma}(x)=\min_z \\frac{1}{2} ||x-z||_2^2 +\gamma f(z)`
        This method is used by all proximal solvers.
        """
        raise NotImplementedError("Class user should define this method.")

    def grad(self, x):
        """
        Function gradient evaluated at x.
        This method is only used by some solvers.
        """
        raise NotImplementedError("Class user should define this method.")


class norm(func):
    """
    Base class which defines the attributes of the norm objects.

    Attributes :

    * *lamb*  : regularization parameter :math:`\lambda`
    * *w*     : weights for a weighted L2-norm (default :math:`w=1`)
    * *y*     : measurements (default :math:`y=0`)
    * *A*     : forward operator (default identity :math:`A(x)=x`)
    * *At*    : adjoint operator (default :math:`At(x)=A(x)`)
    * *tight* : ``True`` if :math:`A` is a tight frame,
      ``False`` otherwise (default ``True``)
    * *nu*    : bound on the norm of the operator :math:`A`,
      i.e. :math:`||A(x)||^2 \leq \\nu ||x||^2` (default :math:`\\nu=1`)
    """

    def __init__(self, lamb, w=1, y=0, A=None, At=None,
                 tight=True, nu=1):
        self.lamb = lamb
        self.w = np.array(w)
        self.y = np.array(y)
        if A:
            self.A = A
        else:
            self.A = lambda x: x
        if At:
            self.At = At
        else:
            self.At = self.A
        self.tight = tight
        self.nu = nu


class norm_l1(norm):
    """
    L1-norm function, object to be passed to solvers.
    """

    def eval(self, x):
        """
        Return :math:`\lambda ||w\cdot(A(x)-y)||_1`
        """
        sol = self.A(np.array(x)) - self.y
        sol = sum(abs(self.w * sol))
        return self.lamb * sol

    def prox(self, x, T):
        """
        Return
        :math:`\min_z \\frac{1}{2} ||x-z||_2^2 + \gamma ||w\cdot(A(z)-y)||_1`
        where :math:`\gamma = \lambda \cdot T`
        """
        gamma = self.lamb * T
        if self.tight:
            sol = self.A(x)
            sol = self.At(_solf_threshold(sol, gamma*self.nu*self.w) - sol)
            sol = x + sol / self.nu
        else:
            raise NotImplementedError('Not implemented for non tight frame.')
        return sol


class norm_l2(norm):
    """
    L2-norm function, object to be passed to solvers.
    """

    def eval(self, x):
        """
        Return :math:`\lambda ||w\cdot(A(x)-y)||_2^2`
        """
        sol = self.A(np.array(x)) - self.y
        sol = np.sum((self.w * sol)**2)
        return self.lamb * sol

    def prox(self, x, T):
        """
        Return
        :math:`\min_z \\frac{1}{2} ||x-z||_2^2 + \gamma ||w\cdot(A(z)-y)||_2^2`
        where :math:`\gamma = \lambda \cdot T`
        """
        gamma = self.lamb * T
        if self.tight:
            sol = np.array(x) + 2. * gamma * self.At(self.y * self.w**2)
            sol /= 1. + 2. * gamma * self.nu * self.w**2
        else:
            raise NotImplementedError('Not implemented for non tight frame.')
        return sol

    def grad(self, x):
        """
        Return :math:`2 \lambda \cdot At( w\cdot(A(x)-y) )`
        """
        sol = self.A(np.array(x)) - self.y
        return 2 * self.lamb * self.w * self.At(sol)
