# -*- coding: utf-8 -*-

r"""
This module implements function objects which are then passed to solvers.  The
:class:`func` base class defines the interface whereas specialised classes who
inherit from it implement the methods. These classes include :

* :class:`norm_l1`: L1-norm which implements the :meth:`eval` and :meth:`prox`
  methods
* :class:`norm_l2`: L2-norm which implements the :meth:`eval`, :meth:`prox`
  and :meth:`grad` methods
"""

import numpy as np


def _soft_threshold(z, T):
    r"""
    Return the soft thresholded signal.

    Parameters
    ----------
    z : array_like
        input signal (real or complex)
    T : float
        threshold on the absolute value of `z`
    """
    z = np.array(z)
    sol = np.maximum(abs(z)-T*abs(z), 0) * z
    sol /= np.maximum(abs(z)-T*abs(z), 0) + T*abs(z) + (abs(z) == 0)
    return sol


class func(object):
    r"""
    This class defines the function object interface.

    It is intended to be a base class for standard functions which will
    implement the required methods. It can also be instantiated by user code
    and dynamically modified for rapid testing.  The instanced objects are
    meant to be passed to the :func:`pyunlocbox.solvers.solve` solving
    function.

    Examples
    --------
    >>> import numpy as np
    >>> import pyunlocbox
    >>> f1 = pyunlocbox.functions.func()
    >>> f1.eval = lambda x : x**2
    >>> f1.grad = lambda x : 2*x
    >>> x = np.array([1, 2, 3, 4])
    >>> f1.eval(x)
    array([ 1,  4,  9, 16])
    >>> f1.grad(x)
    array([2, 4, 6, 8])
    """

    def eval(self, x):
        r"""
        Function evaluation.

        Parameters
        ----------
        x : array_like
            The evaluation point.

        Returns
        -------
        float
            The objective function evaluated at `x`.

        Notes
        -----
        This method is required to compute the objective function.
        """
        raise NotImplementedError("Class user should define this method.")

    def prox(self, x, T):
        r"""
        Function proximal operator.

        Parameters
        ----------
        x : array_like
            The evaluation point.
        T : float
            The regularization parameter.

        Returns
        -------
        ndarray
            The proximal operator evaluated at `x`.

        Notes
        -----
        This method is required by some solvers.

        The proximal operator is defined by :math:`prox_{f,\gamma}(x) = \min_z
        \frac{1}{2} ||x-z||_2^2 + \gamma f(z)`
        """
        raise NotImplementedError("Class user should define this method.")

    def grad(self, x):
        r"""
        Function gradient.

        Parameters
        ----------
        x : array_like
            The evaluation point.

        Returns
        -------
        ndarray
            The function gradient evaluated at `x`.

        Notes
        -----
        This method is required by some solvers.
        """
        raise NotImplementedError("Class user should define this method.")


class norm(func):
    r"""
    Base class which defines the attributes of the norm objects.

    Parameters
    ----------
    lamb : float
        regularization parameter :math:`\lambda`
    y : array_like, optional
        measurements. Default is 0.
    w : array_like, optional
        weights for a weighted norm. Default is 1.
    A : function, optional
        forward operator. Default is the identity, :math:`A(x)=x`.
    At : function, optional
        adjoint operator. Default is A, :math:`At(x)=A(x)`.
    tight : bool, optional
        ``True`` if `A` is a tight frame, ``False`` otherwise. Default is
        `True`.
    nu : float, optional
        bound on the norm of the operator `A`, i.e. :math:`||A(x)||^2 \leq \nu
        ||x||^2`. Default is 1.
    """

    def __init__(self, lamb, y=0, w=1, A=None, At=None,
                 tight=True, nu=1):
        self.lamb = lamb
        self.y = np.array(y)
        self.w = np.array(w)
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
    r"""
    L1-norm function object.
    """

    def eval(self, x):
        """
        L-1 norm evaluation.

        Parameters
        ----------
        x : array_like
            The evaluation point.

        Returns
        -------
        float
             The L-1 norm of the vector `x` :
             :math:`\lambda ||w\cdot(A(x)-y)||_1`

        Examples
        --------
        >>> import pyunlocbox
        >>> f1 = pyunlocbox.functions.norm_l1(1)
        >>> f1.eval([1, 2, 3, 4])
        10
        """
        sol = self.A(np.array(x)) - self.y
        sol = sum(abs(self.w * sol))
        return self.lamb * sol

    def prox(self, x, T):
        r"""
        L-1 norm proximal operator.

        Parameters
        ----------
        x : array_like
            The evaluation point.
        T : float
            The regularization parameter.

        Returns
        -------
        ndarray
            The L1-norm proximal operator evaluated at `x` :
            :math:`\min_z \frac{1}{2} ||x-z||_2^2 + \gamma
            ||w\cdot(A(z)-y)||_1` where :math:`\gamma = \lambda \cdot T`

        Examples
        --------
        >>> import pyunlocbox
        >>> f1 = pyunlocbox.functions.norm_l1(1)
        >>> f1.prox([1, 2, 3, 4], 1)
        0
        """
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lamb * T
        if self.tight:
            sol = self.A(x)
            sol = self.At(_solf_threshold(sol, gamma*self.nu*self.w) - sol)
            sol = x + sol / self.nu
        else:
            raise NotImplementedError('Not implemented for non tight frame.')
        return sol


class norm_l2(norm):
    r"""
    L2-norm function object.
    """

    def eval(self, x):
        r"""
        L-2 norm evaluation.

        Parameters
        ----------
        x : array_like
            The evaluation point.

        Returns
        -------
        float
            The squared L-2 norm of the vector `x` :
            :math:`\lambda ||w\cdot(A(x)-y)||_2^2`

        Examples
        --------
        >>> import pyunlocbox
        >>> f1 = pyunlocbox.functions.norm_l2(1)
        >>> f1.eval([1, 2, 3, 4])
        30
        """
        sol = self.A(np.array(x)) - self.y
        sol = np.sum((self.w * sol)**2)
        return self.lamb * sol

    def prox(self, x, T):
        r"""
        L-2 norm proximal operator.

        Parameters
        ----------
        x : array_like
            The evaluation point.
        T : float
            The regularization parameter.

        Returns
        -------
        ndarray
            The L2-norm proximal operator evaluated at `x` :
            :math:`\min_z \frac{1}{2} ||x-z||_2^2 + \gamma
            ||w\cdot(A(z)-y)||_2^2` where :math:`\gamma = \lambda \cdot T`

        Examples
        --------
        >>> import pyunlocbox
        >>> f1 = pyunlocbox.functions.norm_l2(1)
        >>> f1.prox([1, 2, 3, 4], 1)
        array([0.33333333, 0.66666667, 1., 1.33333333])
        """
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lamb * T
        if self.tight:
            sol = np.array(x) + 2. * gamma * self.At(self.y * self.w**2)
            sol /= 1. + 2. * gamma * self.nu * self.w**2
        else:
            raise NotImplementedError('Not implemented for non tight frame.')
        return sol

    def grad(self, x):
        r"""
        L-2 norm gradient.

        Parameters
        ----------
        x : array_like
            The evaluation point.

        Returns
        -------
        ndarray
            The L2-norm gradient evaluated at `x` :
            :math:`2 \lambda \cdot At( w\cdot(A(x)-y) )`

        Examples
        --------
        >>> import pyunlocbox
        >>> f1 = pyunlocbox.functions.norm_l2(1)
        >>> f1.grad([1, 2, 3, 4])
        array([2, 4, 6, 8])
        """
        sol = self.A(np.array(x)) - self.y
        return 2 * self.lamb * self.w * self.At(sol)
