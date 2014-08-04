# -*- coding: utf-8 -*-

r"""
This module implements function objects which are then passed to solvers.  The
:class:`func` base class defines the interface whereas specialised classes who
inherit from it implement the methods. These classes include :

* :class:`dummy`: A dummy function object which returns 0 for the
  :meth:`_eval`, :meth:`_prox` and :meth:`_grad` methods.

* :class:`norm`: Norm operators base class.

  * :class:`norm_l1`: L1-norm who implements the :meth:`_eval` and
    :meth:`_prox` methods.
  * :class:`norm_l2`: L2-norm who implements the :meth:`_eval`, :meth:`_prox`
    and :meth:`_grad` methods.

* :class:`proj`: Projection operators base class.

  * :class:`proj_b2`: Projection on the L2-ball who implements the
    :meth:`_eval` and :meth:`_prox` methods.

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
        ``True`` if `A` is a tight frame, ``False`` otherwise. Default is
        ``True``.
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

    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.func()
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
                 maxit=200):

        self.y = np.array(y)

        if A is None:
            self.A = lambda x: x
        else:
            if type(A) is np.ndarray:
                # Transform matrix form to operator form.
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
                # Transform matrix form to operator form.
                self.At = lambda x: np.dot(At, x)
            else:
                self.At = At

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
            The evaluation point.

        Returns
        -------
        z : float
            The objective function evaluated at `x`.

        Notes
        -----
        This method is required by the :func:`pyunlocbox.solvers.solve` solving
        function to evaluate the objective function.
        """
        sol = self._eval(np.array(x))
        if self.verbosity in ['LOW', 'HIGH']:
            print('    %s evaluation : %e' % (self.__class__.__name__, sol))
        return sol

    def _eval(self, x):
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
        z : ndarray
            The proximal operator evaluated at `x`.

        Notes
        -----
        This method is required by some solvers.

        The proximal operator is defined by
        :math:`\operatorname{prox}_{f,\gamma}(x) = \operatorname{arg\,min}
        \limits_z \frac{1}{2} \|x-z\|_2^2 + \gamma f(z)`
        """
        return self._prox(np.array(x), T)

    def _prox(self, x, T):
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
        z : ndarray
            The objective function gradient evaluated at `x`.

        Notes
        -----
        This method is required by some solvers.
        """
        return self._grad(np.array(x))

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
    Dummy function object.

    This can be used as a second function object when there is only one
    function to minimize. It always evaluates as 0.

    Examples
    --------
    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.dummy()
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
        self.w = np.array(w)


class norm_l1(norm):
    r"""
    L1-norm function object.

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
    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.norm_l1()
    >>> f.eval([1, 2, 3, 4])
    10
    >>> f.prox([1, 2, 3, 4], 1)
    array([ 0.,  1.,  2.,  3.])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(norm_l1, self).__init__(**kwargs)

    def _eval(self, x):
        sol = self.A(np.array(x)) - self.y
        sol = self.lambda_ * np.sum(np.abs(self.w * sol))
        return sol

    def _prox(self, x, T):
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
    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.norm_l2()
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
        sol = self.A(np.array(x)) - self.y
        sol = self.lambda_ * np.sum((self.w * sol)**2)
        return sol

    def _prox(self, x, T):
        # Gamma is T in the matlab UNLocBox implementation.
        gamma = self.lambda_ * T
        if self.tight:
            sol = np.array(x) + 2. * gamma * self.At(self.y * self.w**2)
            sol /= 1. + 2. * gamma * self.nu * self.w**2
        else:
            raise NotImplementedError('Not implemented for non tight frame.')
        return sol

    def _grad(self, x):
        sol = self.A(np.array(x)) - self.y
        return 2 * self.lambda_ * self.w * self.At(sol)


class proj(func):
    r"""
    Base class which defines the attributes of the `proj` objects.

    See generic attributes descriptions of the
    :class:`pyunlocbox.functions.func` base class.

    Parameters
    ----------
    epsilon : float, optional
        The radius of the ball. Default is 1e-3.
    method : {'FISTA', 'ISTA'}, optional
        The method used to solve the problem. It can be 'FISTA' or 'ISTA'.
        Default is 'FISTA'.
    """

    def __init__(self, epsilon=1e-3, method='FISTA', **kwargs):
        super(proj, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.method = method


class proj_b2(proj):
    r"""
    L2-ball function object.

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
    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.proj_b2(y=[1, 2])
    >>> x = [3, 3]
    >>> f.eval(x)
    0
    >>> f.prox(x, 1)
    array([ 1.00089443,  2.00044721])

    """

    def __init__(self, **kwargs):
        # Constructor takes keyword-only parameters to prevent user errors.
        super(proj_b2, self).__init__(**kwargs)

    def _eval(self, x):
        # Matlab version returns a small delta to avoid division by 0 when
        # evaluating relative tolerance. Here the delta is added in the solve
        # function if the sum of the objective functions is zero.
        # np.spacing(1.0) is equivalent to matlab eps = eps(1.0)
        # return np.spacing(1.0)
        return 0

    def _prox(self, x, T):

        crit = None  # Stopping criterion.
        niter = 0    # Number of iterations.

        # Tight frame.
        if self.tight:
            tmp1 = self.A(x) - self.y
            tmp2 = tmp1 * min(1, self.epsilon/np.linalg.norm(tmp1))  # Scaling.
            sol = x + self.At(tmp2 - tmp1) / self.nu
            crit = 'TOL'
            u = np.nan

        # Non tight frame.
        else:

            # Initialization.
            sol = x
            u = np.zeros(np.size(self.y))
            if self.method is 'FISTA':
                v_last = u
                t_last = 1.
            elif self.method is not 'ISTA':
                raise ValueError('The method should be either FISTA or ISTA.')

            # Tolerance around the L2-ball.
            epsilon_low = self.epsilon / (1. + self.tol)
            epsilon_up = self.epsilon / (1. - self.tol)

            # Check if we are already in the L2-ball.
            norm_res = np.linalg.norm(self.y - self.A(sol), 2)
            if norm_res <= epsilon_up:
                crit = 'INBALL'

            # Projection onto the L2-ball
            while not crit:

                niter += 1

                # Residual.
                res = self.A(sol) - self.y
                norm_res = np.linalg.norm(res, 2)

                if self.verbosity is 'HIGH':
                    print('    proj_b2 iteration %3d : epsilon = %.2e, '
                          '||y-A(z)||_2 = %.2e'
                          % (niter, self.epsilon, norm_res))

                # Scaling for projection.
                res += u * self.nu
                norm_proj = np.linalg.norm(res, 2)

                ratio = min(1, self.epsilon/norm_proj)
                v = 1. / self.nu * (res - res*ratio)

                if self.method is 'FISTA':
                    t = (1. + np.sqrt(1.+4.*t_last**2.)) / 2.  # Time step.
                    u = v + (t_last-1.) / t * (v-v_last)
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
                norm_res = np.linalg.norm(self.y - self.A(sol), 2)
                print('    proj_b2 : epsilon = %.2e, ||y-A(z)||_2 = %.2e, '
                      '%s, niter = %d' % (self.epsilon, norm_res, crit, niter))

        return sol
