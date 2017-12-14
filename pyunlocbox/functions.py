# -*- coding: utf-8 -*-

r"""
The :mod:`pyunlocbox.functions` module implements an interface for solvers to
access the functions to be optimized as well as common objective functions.

Interface
---------

The :class:`func` base class defines a common interface to all functions:

.. autosummary::

    func.cap
    func.eval
    func.prox
    func.grad

Functions
---------

Then, derived classes implement various common objective functions.

**Norm operators** (based on :class:`norm`)

.. autosummary::

    norm_l1
    norm_l2
    norm_nuclear
    norm_tv

**Projection operators** (based on :class:`proj`)

.. autosummary::

    proj_b2

**Miscellaneous**

.. autosummary::

    dummy

.. inheritance-diagram:: pyunlocbox.functions
    :parts: 2

"""

from __future__ import division

from time import time
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

from pyunlocbox import operators as op


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
    >>> from pyunlocbox import functions
    >>> functions._soft_threshold([-2, -1, 0, 1, 2], 1)
    array([-1,  0,  0,  0,  1])

    """
    sz = np.maximum(np.abs(z) - T, 0)

    if not handle_complex:
        # This soft thresholding method only supports real signal.
        sz[:] = np.sign(z) * sz

    else:
        # This soft thresholding method supports complex complex signal.
        # Transform to float to avoid integer division.
        # In our case 0 divided by 0 should be 0, not NaN, and is not an error.
        # It corresponds to 0 thresholded by 0, which is 0.
        old_err_state = np.seterr(invalid='ignore')
        sz[:] = np.nan_to_num(1. * sz / (sz + T) * z)
        np.seterr(**old_err_state)

    return sz


def _prox_star(func, z, T):
    r"""
    Proximity operator of the convex conjugate of a function.

    Notes
    -----
    Based on the Moreau decomposition of a vector w.r.t. a convex function.

    """
    return z - T * func.prox(z / T, 1 / T)


class func(object):
    r"""
    This class defines the function object interface.

    It is intended to be a base class for standard functions which will
    implement the required methods. It can also be instantiated by user code
    and dynamically modified for rapid testing.  The instanced objects are
    meant to be passed to the :func:`pyunlocbox.solvers.solve` solving
    function.

    Parameters
    ----------
    y : array_like, optional
        Measurements. Default is 0.
    A : function or ndarray, optional
        The forward operator. Default is the identity, :math:`A(x)=x`. If `A`
        is an ``ndarray``, it will be converted to the operator form.
    At : function or ndarray, optional
        The adjoint operator. If `At` is an ``ndarray``, it will be converted
        to the operator form. If `A` is an ``ndarray``, default is the
        transpose of `A`.  If `A` is a function, default is `A`,
        :math:`At(x)=A(x)`.
    tight : bool, optional
        ``True`` if `A` is a tight frame (semi-orthogonal linear transform),
        ``False`` otherwise. Default is ``True``.
    nu : float, optional
        Bound on the norm of the operator `A`, i.e. :math:`\|A(x)\|^2 \leq \nu
        \|x\|^2`. Default is 1.
    tol : float, optional
        The tolerance stopping criterion. The exact definition depends on the
        function object, please see the documentation of the considered
        function. Default is 1e-3.
    maxit : int, optional
        The maximum number of iterations. Default is 200.

    Examples
    --------

    Let's define a parabola as an example of the manual implementation of a
    function object :

    >>> from pyunlocbox import functions
    >>> f = functions.func()
    >>> f._eval = lambda x: x**2
    >>> f._grad = lambda x: 2*x
    >>> x = [1, 2, 3, 4]
    >>> f.eval(x)
    array([ 1,  4,  9, 16])
    >>> f.grad(x)
    array([2, 4, 6, 8])
    >>> f.cap(x)
    ['EVAL', 'GRAD']

    """

    def __init__(self, y=0, A=None, At=None, tight=True, nu=1, tol=1e-3,
                 maxit=200, **kwargs):

        if callable(y):
            self.y = lambda: np.asarray(y())
        else:
            self.y = lambda: np.asarray(y)

        if A is None:
            self.A = lambda x: x
        else:
            if callable(A):
                self.A = A
            else:
                # Transform matrix form to operator form.
                self.A = lambda x: A.dot(x)

        if At is None:
            if A is None:
                self.At = lambda x: x
            elif callable(A):
                self.At = A
            else:
                self.At = lambda x: A.T.dot(x)
        else:
            if callable(At):
                self.At = At
            else:
                self.At = lambda x: At.dot(x)

        self.tight = tight
        self.nu = nu
        self.tol = tol
        self.maxit = maxit

        # Should be initialized if called alone, updated by solve().
        self.verbosity = 'NONE'

    def eval(self, x):
        r"""
        Function evaluation.

        Parameters
        ----------
        x : array_like
            The evaluation point. If `x` is a matrix, the function gets
            evaluated for each column, as if it was a set of independent
            problems. Some functions, like the nuclear norm, are only defined
            on matrices.

        Returns
        -------
        z : float
            The objective function evaluated at `x`. If `x` is a matrix, the
            sum of the objectives is returned.

        Notes
        -----
        This method is required by the :func:`pyunlocbox.solvers.solve` solving
        function to evaluate the objective function. Each function class
        should therefore define it.

        """
        sol = self._eval(np.asarray(x))
        if self.verbosity in ['LOW', 'HIGH']:
            name = self.__class__.__name__
            print('    {} evaluation: {:e}'.format(name, sol))
        return sol

    def _eval(self, x):
        raise NotImplementedError("Class user should define this method.")

    def prox(self, x, T):
        r"""
        Function proximal operator.

        Parameters
        ----------
        x : array_like
            The evaluation point. If `x` is a matrix, the function gets
            evaluated for each column, as if it was a set of independent
            problems. Some functions, like the nuclear norm, are only defined
            on matrices.
        T : float
            The regularization parameter.

        Returns
        -------
        z : ndarray
            The proximal operator evaluated for each column of `x`.

        Notes
        -----
        The proximal operator is defined by
        :math:`\operatorname{prox}_{\gamma f}(x) = \operatorname{arg\,min}
        \limits_z \frac{1}{2} \|x-z\|_2^2 + \gamma f(z)`

        This method is required by some solvers.

        When the map A in the function construction is a tight frame
        (semi-orthogonal linear transformation), we can use property (x) of
        Table 10.1 in :cite:`combettes:2011iq` to compute the proximal
        operator of the composition of A with the base function. Whenever
        this is not the case, we have to resort to some iterative procedure,
        which may be very inefficient.

        """
        return self._prox(np.asarray(x), T)

    def _prox(self, x, T):
        raise NotImplementedError("Class user should define this method.")

    def grad(self, x):
        r"""
        Function gradient.

        Parameters
        ----------
        x : array_like
            The evaluation point. If `x` is a matrix, the function gets
            evaluated for each column, as if it was a set of independent
            problems. Some functions, like the nuclear norm, are only defined
            on matrices.

        Returns
        -------
        z : ndarray
            The objective function gradient evaluated for each column of `x`.

        Notes
        -----
        This method is required by some solvers.

        """
        return self._grad(np.asarray(x))

    def _grad(self, x):
        raise NotImplementedError("Class user should define this method.")

    def cap(self, x):
        r"""
        Test the capabilities of the function object.

        Parameters
        ----------
        x : array_like
            The evaluation point. Not really needed, but this function calls
            the methods of the object to test if they can properly execute
            without raising an exception. Therefore it needs some evaluation
            point with a consistent size.

        Returns
        -------
        cap : list of string
            A list of capabilities ('EVAL', 'GRAD', 'PROX').

        """
        tmp = self.verbosity
        self.verbosity = 'NONE'
        cap = ['EVAL', 'GRAD', 'PROX']
        try:
            self.eval(x)
        except NotImplementedError:
            cap.remove('EVAL')
        try:
            self.grad(x)
        except NotImplementedError:
            cap.remove('GRAD')
        try:
            self.prox(x, 1)
        except NotImplementedError:
            cap.remove('PROX')
        self.verbosity = tmp
        return cap


class dummy(func):
    r"""
    Dummy function which returns 0 (eval, prox, grad).

    This can be used as a second function object when there is only one
    function to minimize. It always evaluates as 0.

    Examples
    --------
    >>> from pyunlocbox import functions
    >>> f = functions.dummy()
    >>> x = [1, 2, 3, 4]
    >>> f.eval(x)
    0
    >>> f.prox(x, 1)
    array([1, 2, 3, 4])
    >>> f.grad(x)
    array([ 0.,  0.,  0.,  0.])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(dummy, self).__init__(**kwargs)

    def _eval(self, x):
        return 0

    def _prox(self, x, T):
        return x

    def _grad(self, x):
        return np.zeros(np.shape(x))


class norm(func):
    r"""
    Base class which defines the attributes of the `norm` objects.

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.func` base class.

    Parameters
    ----------
    lambda_ : float, optional
        Regularization parameter :math:`\lambda`. Default is 1.
    w : array_like, optional
        Weights for a weighted norm. Default is 1.

    """

    def __init__(self, lambda_=1, w=1, **kwargs):
        super(norm, self).__init__(**kwargs)
        self.lambda_ = lambda_
        self.w = np.asarray(w)


class norm_l1(norm):
    r"""
    L1-norm (eval, prox).

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.norm` base class. Note that the constructor
    takes keyword-only parameters.

    Notes
    -----
    * The L1-norm of the vector `x` is given by
      :math:`\lambda \|w \cdot (A(x)-y)\|_1`.
    * The L1-norm proximal operator evaluated at `x` is given by
      :math:`\operatorname{arg\,min}\limits_z \frac{1}{2} \|x-z\|_2^2 + \gamma
      \|w \cdot (A(z)-y)\|_1` where :math:`\gamma = \lambda \cdot T`. This is
      simply a soft thresholding.

    Examples
    --------
    >>> from pyunlocbox import functions
    >>> f = functions.norm_l1()
    >>> f.eval([1, 2, 3, 4])
    10
    >>> f.prox([1, 2, 3, 4], 1)
    array([0, 1, 2, 3])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(norm_l1, self).__init__(**kwargs)

    def _eval(self, x):
        sol = self.A(x) - self.y()
        return self.lambda_ * np.sum(np.abs(self.w * sol))

    def _prox(self, x, T):
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        if self.tight:
            # Nati: I've checked this code the use of 'y' seems correct
            sol = self.A(x) - self.y()
            sol[:] = _soft_threshold(sol, gamma * self.nu * self.w) - sol
            sol[:] = x + self.At(sol) / self.nu
        else:
            raise NotImplementedError('Not implemented for non-tight frame.')
        return sol


class norm_l2(norm):
    r"""
    L2-norm (eval, prox, grad).

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.norm` base class. Note that the constructor
    takes keyword-only parameters.

    Notes
    -----
    * The squared L2-norm of the vector `x` is given by
      :math:`\lambda \|w \cdot (A(x)-y)\|_2^2`.
    * The squared L2-norm proximal operator evaluated at `x` is given by
      :math:`\operatorname{arg\,min}\limits_z \frac{1}{2} \|x-z\|_2^2 + \gamma
      \|w \cdot (A(z)-y)\|_2^2` where :math:`\gamma = \lambda \cdot T`.
    * The squared L2-norm gradient evaluated at `x` is given by
      :math:`2 \lambda \cdot At(w \cdot (A(x)-y))`.

    Examples
    --------
    >>> from pyunlocbox import functions
    >>> f = functions.norm_l2()
    >>> x = [1, 2, 3, 4]
    >>> f.eval(x)
    30
    >>> f.prox(x, 1)
    array([ 0.33333333,  0.66666667,  1.        ,  1.33333333])
    >>> f.grad(x)
    array([2, 4, 6, 8])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(norm_l2, self).__init__(**kwargs)

    def _eval(self, x):
        sol = self.A(x) - self.y()
        return self.lambda_ * np.sum((self.w * sol)**2)

    def _prox(self, x, T):
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        if self.tight:
            sol = x + 2. * gamma * self.At(self.y() * self.w**2)
            sol /= 1. + 2. * gamma * self.nu * self.w**2
        else:
            res = minimize(fun=lambda z: 0.5 * np.sum((z - x)**2) + gamma *
                           np.sum((self.w * (self.A(z) - self.y()))**2),
                           x0=x,
                           method='BFGS',
                           jac=lambda z: z - x + 2. * gamma *
                           self.At((self.w**2) * (self.A(z) - self.y())))
            if res.success:
                sol = res.x
            else:
                raise RuntimeError('norm_l2.prox: ' + res.message)
        return sol

    def _grad(self, x):
        sol = self.A(x) - self.y()
        return 2 * self.lambda_ * self.At((self.w**2) * sol)


class norm_nuclear(norm):
    r"""
    Nuclear-norm (eval, prox).

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.norm` base class. Note that the constructor
    takes keyword-only parameters.

    Notes
    -----
    * The nuclear-norm of the matrix `x` is given by
      :math:`\lambda \| x \|_* = \lambda \operatorname{trace} (\sqrt{x^* x}) =
      \lambda \sum_{i=1}^N |e_i|` where `e_i` are the eigenvalues of `x`.
    * The nuclear-norm proximal operator evaluated at `x` is given by
      :math:`\operatorname{arg\,min}\limits_z \frac{1}{2} \|x-z\|_2^2 + \gamma
      \| x \|_*` where :math:`\gamma = \lambda \cdot T`, which is a
      soft-thresholding of the eigenvalues.

    Examples
    --------
    >>> from pyunlocbox import functions
    >>> f = functions.norm_nuclear()
    >>> f.eval([[1, 2],[2, 3]])  # doctest:+ELLIPSIS
    4.47213595...
    >>> f.prox([[1, 2],[2, 3]], 1)
    array([[ 0.89442719,  1.4472136 ],
           [ 1.4472136 ,  2.34164079]])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(norm_nuclear, self).__init__(**kwargs)

    def _eval(self, x):
        # TODO: take care of sparse matrices.
        _, s, _ = np.linalg.svd(x)
        return self.lambda_ * np.sum(np.abs(s))

    def _prox(self, x, T):
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        # TODO: take care of sparse matrices.
        U, s, V = np.linalg.svd(x)
        s = _soft_threshold(s, gamma)
        S = np.diag(s)
        return np.dot(U, np.dot(S, V))


class norm_tv(norm):
    r"""
    TV-norm (eval, prox).

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.norm` base class. Note that the constructor
    takes keyword-only parameters.

    Notes
    -----
    TODO

    See :cite:`beck2009fastTV` for details about the algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions
    >>> f = functions.norm_tv()
    >>> x = np.arange(0, 16)
    >>> x = x.reshape(4, 4)
    >>> f.eval(x)  # doctest:+ELLIPSIS
        norm_tv evaluation: 5.210795e+01
    52.10795063...

    """

    def __init__(self, dim=2, verbosity='LOW', **kwargs):
        super(norm_tv, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.dim = dim
        self.verbosity = verbosity

    def _eval(self, x):
        if self.dim >= 2:
            y = 0
            grads = []
            grads = op.grad(x, dim=self.dim, **self.kwargs)
            for g in grads:
                y += np.power(abs(g), 2)
            y = np.sqrt(y)
            return np.sum(y)

        if self.dim == 1:
            dx = op.grad(x, dim=self.dim, **self.kwargs)
            y = np.sum(np.abs(dx), axis=0)
            return np.sum(y)

    def _prox(self, x, T):
        # Time counter
        t_init = time()

        tol = self.tol
        maxit = self.maxit

        # TODO implement test_gamma
        # Initialization
        sol = x

        if self.dim == 1:
            r = op.grad(x * 0, dim=self.dim, **self.kwargs)
            rr = deepcopy(r)
        elif self.dim == 2:
            r, s = op.grad(x * 0, dim=self.dim, **self.kwargs)
            rr, ss = deepcopy(r), deepcopy(s)
        elif self.dim == 3:
            r, s, k = op.grad(x * 0, dim=self.dim, **self.kwargs)
            rr, ss, kk = deepcopy(r), deepcopy(s), deepcopy(k)
        elif self.dim == 4:
            r, s, k, u = op.grad(x * 0, dim=self.dim, **self.kwargs)
            rr, ss, kk, uu = deepcopy(r), deepcopy(s), deepcopy(k), deepcopy(u)

        if self.dim >= 1:
            pold = r
        if self.dim >= 2:
            qold = s
        if self.dim >= 3:
            kold = k
        if self.dim >= 4:
            uold = u

        told, prev_obj = 1., 0.

        # Initialization for weights
        if self.dim >= 1:
            try:
                wx = self.kwargs["wx"]
            except (KeyError, TypeError):
                wx = 1.
        if self.dim >= 2:
            try:
                wy = self.kwargs["wy"]
            except (KeyError, TypeError):
                wy = 1.
        if self.dim >= 3:
            try:
                wz = self.kwargs["wz"]
            except (KeyError, TypeError):
                wz = 1.
        if self.dim >= 4:
            try:
                wt = self.kwargs["wt"]
            except (KeyError, TypeError):
                wt = 1.

        if self.dim == 1:
            mt = wx
        elif self.dim == 2:
            mt = np.maximum(wx, wy)
        elif self.dim == 3:
            mt = np.maximum(wx, np.maximum(wy, wz))
        elif self.dim == 4:
            mt = np.maximum(np.maximum(wx, wy), np.maximum(wz, wt))

        if self.verbosity in ['LOW', 'HIGH', 'ALL']:
            print("Proximal TV Operator")

        iter = 0
        while iter <= maxit:
            # Current Solution
            if self.dim == 1:
                sol = x - T * op.div(rr, **self.kwargs)
            elif self.dim == 2:
                sol = x - T * op.div(rr, ss, **self.kwargs)
            elif self.dim == 3:
                sol = x - T * op.div(rr, ss, kk, **self.kwargs)
            elif self.dim == 4:
                sol = x - T * op.div(rr, ss, kk, uu, **self.kwargs)

            #  Objective function value
            obj = 0.5 * np.power(np.linalg.norm(x[:] - sol[:]), 2) + \
                T * np.sum(self._eval(sol), axis=0)
            rel_obj = np.abs(obj - prev_obj) / obj
            prev_obj = obj

            if self.verbosity in ['HIGH', 'ALL']:
                print("Iter: ", iter, " obj = ", obj, " rel_obj = ", rel_obj)

            # Stopping criterion
            if rel_obj < tol:
                crit = "TOL_EPS"
                break

            #  Update divergence vectors and project
            if self.dim == 1:
                dx = op.grad(sol, dim=self.dim, **self.kwargs)
                r -= 1. / (4 * T * mt**2) * dx
                weights = np.maximum(1, np.abs(r))

            elif self.dim == 2:
                dx, dy = op.grad(sol, dim=self.dim, **self.kwargs)
                r -= (1. / (8. * T * mt**2.)) * dx
                s -= (1. / (8. * T * mt**2.)) * dy
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2)))

            elif self.dim == 3:
                dx, dy, dz = op.grad(sol, dim=self.dim, **self.kwargs)
                r -= 1. / (12. * T * mt**2) * dx
                s -= 1. / (12. * T * mt**2) * dy
                k -= 1. / (12. * T * mt**2) * dz
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2) +
                                                np.power(np.abs(k), 2)))

            elif self.dim == 4:
                dx, dy, dz, dt = op.grad(sol, dim=self.dim, **self.kwargs)
                r -= 1. / (16 * T * mt**2) * dx
                s -= 1. / (16 * T * mt**2) * dy
                k -= 1. / (16 * T * mt**2) * dz
                u -= 1. / (16 * T * mt**2) * dt
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2) +
                                                np.power(np.abs(k), 2) +
                                                np.power(np.abs(u), 2)))

            # FISTA update
            t = (1 + np.sqrt(4 * told**2)) / 2.

            if self.dim >= 1:
                p = r / weights
                r = p + (told - 1) / t * (p - pold)
                pold = p
                rr = deepcopy(r)

            if self.dim >= 2:
                q = s / weights
                s = q + (told - 1) / t * (q - qold)
                ss = deepcopy(s)
                qold = q

            if self.dim >= 3:
                o = k / weights
                k = o + (told - 1) / t * (o - kold)
                kk = deepcopy(k)
                kold = o

            if self.dim >= 4:
                m = u / weights
                u = m + (told - 1) / t * (m - uold)
                uu = deepcopy(u)
                uold = m

            told = t
            iter += 1

        try:
            type(crit) == str
        except NameError:
            crit = "MAX_IT"

        t_end = time()
        exec_time = t_end - t_init

        if self.verbosity in ['HIGH', 'ALL']:
            print("Prox_TV: obj = {0}, rel_obj = {1}, {2}, iter = {3}".format(
                obj, rel_obj, crit, iter))
            print("exec_time = ", exec_time)
        return sol


class proj(func):
    r"""
    Base class which defines the attributes of the `proj` objects.

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.func` base class.

    Parameters
    ----------
    epsilon : float, optional
        The radius of the ball. Default is 1.
    method : {'FISTA', 'ISTA'}, optional
        The method used to solve the problem. It can be 'FISTA' or 'ISTA'.
        Default is 'FISTA'.

    Notes
    -----
    * All indicator functions (projections) evaluate to zero by definition.

    """

    def __init__(self, epsilon=1, method='FISTA', **kwargs):
        super(proj, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.method = method

    def _eval(self, x):
        # Matlab version returns a small delta to avoid division by 0 when
        # evaluating relative tolerance. Here the delta is added in the solve
        # function if the sum of the objective functions is zero.
        # np.spacing(1.0) is equivalent to matlab eps = eps(1.0)
        # return np.spacing(1.0)
        return 0


class proj_b2(proj):
    r"""
    Projection on the L2-ball (eval, prox).

    This function is the indicator function :math:`i_S(z)` of the set S which
    is zero if `z` is in the set and infinite otherwise. The set S is defined
    by :math:`\left\{z \in \mathbb{R}^N \mid \|A(z)-y\|_2 \leq \epsilon
    \right\}`.

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.proj` base class. Note that the constructor
    takes keyword-only parameters.

    Notes
    -----
    * The `tol` parameter is defined as the tolerance for the projection on the
      L2-ball. The algorithm stops if :math:`\frac{\epsilon}{1-tol} \leq
      \|y-A(z)\|_2 \leq \frac{\epsilon}{1+tol}`.
    * The evaluation of this function is zero.
    * The L2-ball proximal operator evaluated at `x` is given by
      :math:`\operatorname{arg\,min}\limits_z \frac{1}{2} \|x-z\|_2^2 + i_S(z)`
      which has an identical solution as
      :math:`\operatorname{arg\,min}\limits_z \|x-z\|_2^2` such that
      :math:`\|A(z)-y\|_2 \leq \epsilon`. It is thus a projection of the vector
      `x` onto an L2-ball of diameter `epsilon`.

    Examples
    --------
    >>> from pyunlocbox import functions
    >>> f = functions.proj_b2(y=[1, 1])
    >>> x = [3, 3]
    >>> f.eval(x)
    0
    >>> f.prox(x, 0)
    array([ 1.70710678,  1.70710678])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(proj_b2, self).__init__(**kwargs)

    def _prox(self, x, T):

        crit = None  # Stopping criterion.
        niter = 0    # Number of iterations.

        # Tight frame.
        if self.tight:
            tmp1 = self.A(x) - self.y()
            with np.errstate(divide='ignore', invalid='ignore'):
                # Avoid 'division by zero' warning
                scale = self.epsilon / np.sqrt(np.sum(tmp1 * tmp1, axis=0))
            tmp2 = tmp1 * np.minimum(1, scale)  # Scaling.
            sol = x + self.At(tmp2 - tmp1) / self.nu
            crit = 'TOL'
            u = np.nan

        # Non tight frame.
        else:

            # Initialization.
            sol = x
            u = np.zeros(np.shape(self.y()))
            if self.method is 'FISTA':
                v_last = u
                t_last = 1.
            elif self.method is not 'ISTA':
                raise ValueError('The method should be either FISTA or ISTA.')

            # Tolerance around the L2-ball.
            epsilon_low = self.epsilon / (1. + self.tol)
            epsilon_up = self.epsilon / (1. - self.tol)

            # Check if we are already in the L2-ball.
            norm_res = np.linalg.norm(self.y() - self.A(sol), 2)
            if norm_res <= epsilon_up:
                crit = 'INBALL'

            # Projection onto the L2-ball
            while not crit:

                niter += 1

                # Residual.
                res = self.A(sol) - self.y()
                norm_res = np.linalg.norm(res, 2)

                if self.verbosity is 'HIGH':
                    print('    proj_b2 iteration {:3d}: epsilon = {:.2e}, '
                          '||y-A(z)||_2 = {:.2e}'.format(niter, self.epsilon,
                                                         norm_res))

                # Scaling for projection.
                res += u * self.nu
                norm_proj = np.linalg.norm(res, 2)

                ratio = min(1, self.epsilon / norm_proj)
                v = 1. / self.nu * (res - res * ratio)

                if self.method is 'FISTA':
                    t = (1. + np.sqrt(1. + 4. * t_last**2.)) / 2.  # Time step.
                    u = v + (t_last - 1.) / t * (v - v_last)
                    v_last = v
                    t_last = t
                else:
                    u = v

                # Current estimation.
                sol = x - self.At(u)

                # Stopping criterion.
                if norm_res >= epsilon_low and norm_res <= epsilon_up:
                    crit = 'TOL'
                elif niter >= self.maxit:
                    crit = 'MAXIT'

            if self.verbosity in ['LOW', 'HIGH']:
                norm_res = np.linalg.norm(self.y() - self.A(sol), 2)
                print('    proj_b2: epsilon = {:.2e}, ||y-A(z)||_2 = {:.2e}, '
                      '{}, niter = {}'.format(self.epsilon, norm_res, crit,
                                              niter))

        return sol
