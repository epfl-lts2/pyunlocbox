# -*- coding: utf-8 -*-

r"""
This module implements function objects which are then passed to solvers.  The
:class:`func` base class defines the interface whereas specialised classes who
inherit from it implement the methods. These classes include :

* :class:`norm_l1`: L1-norm which implements the :meth:`eval` and :meth:`prox`
  methods.
* :class:`norm_l2`: L2-norm which implements the :meth:`eval`, :meth:`prox`
  and :meth:`grad` methods.
"""

import numpy as np


def _soft_threshold(z, T, handle_complex=True):
    r"""
    Return the soft thresholded signal.

    Parameters
    ----------
    z : array_like
        Input signal (real or complex).
    T : float or array_like
        Threshold on the absolute value of `z`. There could be either a single
        threshold for the entire signal `z` or one threshold per dimension.
        Useful when you use weighted norms.
    handle_complex : bool
        Indicate that we should handle the thresholding of complex numbers,
        which may be slower. Default is True.

    Returns
    -------
    sz : ndarray
        Soft thresholded signal.

    Examples
    --------
    >>> import pyunlocbox
    >>> pyunlocbox.functions._soft_threshold([-2, -1, 0, 1, 2], 1)
    array([-1., -0.,  0.,  0.,  1.])
    """

    sz = np.maximum(np.abs(z)-T, 0)

    if not handle_complex:
        # This soft thresholding method only supports real signal.
        sz = np.sign(z) * sz

    else:
        # This soft thresholding method supports complex complex signal.
        # Transform to float to avoid integer division.
        # In our case 0 divided by 0 should be 0, not NaN, and is not an error.
        # It corresponds to 0 thresholded by 0, which is 0.
        old_err_state = np.seterr(invalid='ignore')
        sz = np.nan_to_num(np.float64(sz) / (sz+T) * z)
        np.seterr(**old_err_state)

    return sz


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

    Manual implementation of an L2 norm :
    >>> import pyunlocbox
    >>> import numpy as np
    >>> f = pyunlocbox.functions.func()
    >>> f.eval = lambda x : x**2
    >>> f.grad = lambda x : 2*x
    >>> x = np.array([1, 2, 3, 4])
    >>> f.eval(x)
    array([ 1,  4,  9, 16])
    >>> f.grad(x)
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
        y : float
            The objective function evaluated at `x`.

        Notes
        -----
        This method is required by the :func:`pyunlocbox.solvers.solve` solving
        function to evaluate the objective function.
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
        y : ndarray
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
        y : ndarray
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
    lambda_ : float, optional
        regularization parameter :math:`\lambda`. Default is 1.
    y : array_like, optional
        measurements. Default is 0.
    w : array_like, optional
        weights for a weighted norm. Default is 1.
    A : function or ndarray, optional
        The forward operator. Default is the identity, :math:`A(x)=x`. If `A`
        is an ndarray, it will be converted to the operator form.
    At : function or ndarray, optional
        The adjoint operator. If `A` is an ndarray, default is the transpose of
        `A`. If `A` is a function, default is `A`, :math:`At(x)=A(x)`.
    tight : bool, optional
        ``True`` if `A` is a tight frame, ``False`` otherwise. Default is
        ``True``.
    nu : float, optional
        bound on the norm of the operator `A`, i.e. :math:`||A(x)||^2 \leq \nu
        ||x||^2`. Default is 1.
    verbosity : {'low', 'high', 'none'}, optional
        The log level : 'none' for no log, 'low' for resume at convergence,
        'high' to for all steps. Default is 'low'.
    """

    def __init__(self, lambda_=1, y=0, w=1, A=None, At=None,
                 tight=True, nu=1, verbosity='low'):
        self.lambda_ = lambda_
        self.y = np.array(y)
        self.w = np.array(w)
        if A is None:
            self.A = lambda x: x
        else:
            if type(A) is np.ndarray:
                # Transform matrix form to operator.
                self.A = lambda x: np.dot(A, x)
            else:
                self.A = A
        if At is None:
            if type(A) is np.ndarray:
                self.At = lambda x: np.dot(np.transpose(A), x)
            else:
                self.At = self.A
        else:
            if type(At) is np.ndarray:
                # Transform matrix form to operator.
                self.At = lambda x: np.dot(At, x)
            else:
                self.At = At
        self.tight = tight
        self.nu = nu
        if verbosity not in ['none', 'low', 'high']:
            raise ValueError('Verbosity should be either none, low or high.')
        else:
            self.verbosity = verbosity


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
        y : float
             The L-1 norm of the vector `x` :
             :math:`\lambda ||w \cdot (A(x)-y)||_1`

        Examples
        --------
        >>> import pyunlocbox
        >>> f = pyunlocbox.functions.norm_l1()
        >>> f.eval([1, 2, 3, 4])
        10
        """
        sol = self.A(np.array(x)) - self.y
        sol = np.sum(np.abs(self.w * sol))
        if self.verbosity in ['low', 'high']:
            print('L1-norm evaluation : %e' % (sol,))
        return self.lambda_ * sol

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
        y : ndarray
            The L1-norm proximal operator evaluated at `x` :
            :math:`\min_z \frac{1}{2} ||x-z||_2^2 + \gamma
            ||w \cdot (A(z)-y)||_1` where :math:`\gamma = \lambda \cdot T`

        Examples
        --------
        >>> import pyunlocbox
        >>> f = pyunlocbox.functions.norm_l1()
        >>> f.prox([1, 2, 3, 4], 1)
        array([ 0.,  1.,  2.,  3.])
        """
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        if self.tight:
            sol = self.A(x) - self.y
            sol = _soft_threshold(sol, gamma*self.nu*self.w) - sol
            sol = x + self.At(sol) / self.nu
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
        y : float
            The squared L-2 norm of the vector `x` :
            :math:`\lambda ||w \cdot (A(x)-y)||_2^2`

        Examples
        --------
        >>> import pyunlocbox
        >>> f = pyunlocbox.functions.norm_l2()
        >>> f.eval([1, 2, 3, 4])
        30
        """
        sol = self.A(np.array(x)) - self.y
        sol = np.sum((self.w * sol)**2)
        if self.verbosity in ['low', 'high']:
            print('L2-norm evaluation : %e' % (sol,))
        return self.lambda_ * sol

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
        y : ndarray
            The L2-norm proximal operator evaluated at `x` :
            :math:`\min_z \frac{1}{2} ||x-z||_2^2 + \gamma
            ||w \cdot (A(z)-y)||_2^2` where :math:`\gamma = \lambda \cdot T`

        Examples
        --------
        >>> import pyunlocbox
        >>> f = pyunlocbox.functions.norm_l2()
        >>> f.prox([1, 2, 3, 4], 1)
        array([0.33333333, 0.66666667, 1., 1.33333333])
        """
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
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
        y : ndarray
            The L2-norm gradient evaluated at `x` :
            :math:`2 \lambda \cdot At(w \cdot (A(x)-y))`

        Examples
        --------
        >>> import pyunlocbox
        >>> f = pyunlocbox.functions.norm_l2()
        >>> f.grad([1, 2, 3, 4])
        array([2, 4, 6, 8])
        """
        sol = self.A(np.array(x)) - self.y
        return 2 * self.lambda_ * self.w * self.At(sol)
