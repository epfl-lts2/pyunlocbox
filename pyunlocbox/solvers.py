# -*- coding: utf-8 -*-

r"""
This module implements solver objects who minimize an objective function. Call
:func:`solve` to solve your convex optimization problem using your instantiated
solver and functions objects. The :class:`solver` base class defines the
interface of all solver objects. The specialized solver objects inherit from
it and implement the class methods. The following solvers are included :

* :class:`forward_backward`: Forward-backward proximal splitting algorithm.
* :class:`douglas_rachford`: Douglas-Rachford proximal splitting algorithm.
* :class:`generalized_forward_backward`: Generalized Forward-Backward.

* :class:`primal_dual`: Primal-dual algorithms.

  * :class:`mlfbf`: Monotone+Lipschitz Forward-Backward-Forward primal-dual
    algorithm.
  * :class:`projection_based`: Projection-based primal-dual algorithm.
"""

import numpy as np
import time
from pyunlocbox.functions import dummy


def solve(functions, x0, solver=None, atol=None, dtol=None, rtol=1e-3,
          xtol=None, maxit=200, verbosity='LOW'):
    r"""
    Solve an optimization problem whose objective function is the sum of some
    convex functions.

    This function minimizes the objective function :math:`f(x) =
    \sum\limits_{k=0}^{k=K} f_k(x)`, i.e. solves
    :math:`\operatorname{arg\,min}\limits_x f(x)` for :math:`x \in
    \mathbb{R}^{n \times N}` where :math:`n` is the dimensionality of the data
    and :math:`N` the number of independent problems. It returns a dictionary
    with the found solution and some informations about the algorithm
    execution.

    Parameters
    ----------
    functions : list of objects
        A list of convex functions to minimize. These are objects who must
        implement the :meth:`pyunlocbox.functions.func.eval` method. The
        :meth:`pyunlocbox.functions.func.grad` and / or
        :meth:`pyunlocbox.functions.func.prox` methods are required by some
        solvers. Note also that some solvers can only handle two convex
        functions while others may handle more. Please refer to the
        documentation of the considered solver.
    x0 : array_like
        Starting point of the algorithm, :math:`x_0 \in \mathbb{R}^{n \times
        N}`. Note that if you pass a numpy array it will be modified in place
        during execution to save memory. It will then contain the solution. Be
        careful to pass data of the type (int, float32, float64) you want your
        computations to use.
    solver : solver class instance, optional
        The solver algorithm. It is an object who must inherit from
        :class:`pyunlocbox.solvers.solver` and implement the :meth:`_pre`,
        :meth:`_algo` and :meth:`_post` methods. If no solver object are
        provided, a standard one will be chosen given the number of convex
        function objects and their implemented methods.
    atol : float, optional
        The absolute tolerance stopping criterion. The algorithm stops when
        :math:`f(x^t) < atol` where :math:`f(x^t)` is the objective function at
        iteration :math:`t`. Default is None.
    dtol : float, optional
        Stop when the objective function is stable enough, i.e. when
        :math:`\left|f(x^t) - f(x^{t-1})\right| < dtol`. Default is None.
    rtol : float, optional
        The relative tolerance stopping criterion. The algorithm stops when
        :math:`\left|\frac{ f(x^t) - f(x^{t-1}) }{ f(x^t) }\right| < rtol`.
        Default is :math:`10^{-3}`.
    xtol : float, optional
        Stop when the variable is stable enough, i.e. when :math:`\frac{\|x^t -
        x^{t-1}\|_2}{\sqrt{n N}} < xtol`. Note that additional memory will be
        used to store :math:`x^{t-1}`. Default is None.
    maxit : int, optional
        The maximum number of iterations. Default is 200.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        The log level : ``'NONE'`` for no log, ``'LOW'`` for resume at
        convergence, ``'HIGH'`` for info at all solving steps, ``'ALL'`` for
        all possible outputs, including at each steps of the proximal operators
        computation. Default is ``'LOW'``.

    Returns
    -------
    sol : ndarray
        The problem solution.
    solver : str
        The used solver.
    crit : {'ATOL', 'DTOL', 'RTOL', 'XTOL', 'MAXIT'}
        The used stopping criterion. See above for definitions.
    niter : int
        The number of iterations.
    time : float
        The execution time in seconds.
    objective : ndarray
        The successive evaluations of the objective function at each iteration.

    Examples
    --------
    >>> import pyunlocbox
    >>> import numpy as np

    Define a problem:

    >>> y = [4, 5, 6, 7]
    >>> f = pyunlocbox.functions.norm_l2(y=y)

    Solve it:

    >>> x0 = np.zeros(len(y))
    >>> ret = pyunlocbox.solvers.solve([f], x0, atol=1e-2, verbosity='ALL')
    INFO: Dummy objective function added.
    INFO: Selected solver: forward_backward
        norm_l2 evaluation: 1.260000e+02
        dummy evaluation: 0.000000e+00
    INFO: Forward-backward method: FISTA
    Iteration 1 of forward_backward:
        norm_l2 evaluation: 1.400000e+01
        dummy evaluation: 0.000000e+00
        objective = 1.40e+01
    Iteration 2 of forward_backward:
        norm_l2 evaluation: 1.555556e+00
        dummy evaluation: 0.000000e+00
        objective = 1.56e+00
    Iteration 3 of forward_backward:
        norm_l2 evaluation: 3.293044e-02
        dummy evaluation: 0.000000e+00
        objective = 3.29e-02
    Iteration 4 of forward_backward:
        norm_l2 evaluation: 8.780588e-03
        dummy evaluation: 0.000000e+00
        objective = 8.78e-03
    Solution found after 4 iterations:
        objective function f(sol) = 8.780588e-03
        stopping criterion: ATOL

    Verify the stopping criterion (should be smaller than atol=1e-2):

    >>> np.linalg.norm(ret['sol'] - y)**2  # doctest:+ELLIPSIS
    0.00878058...

    Show the solution (should be close to y w.r.t. the L2-norm measure):

    >>> ret['sol']
    array([ 4.03339154,  5.04173943,  6.05008732,  7.0584352 ])

    Show the used solver:

    >>> ret['solver']
    'forward_backward'

    Show some information about the convergence:

    >>> ret['crit']
    'ATOL'
    >>> ret['niter']
    4
    >>> ret['time']  # doctest:+SKIP
    0.0012578964233398438
    >>> ret['objective']  # doctest:+NORMALIZE_WHITESPACE,+ELLIPSIS
    [[126.0, 0], [13.99999999..., 0], [1.55555555..., 0],
    [0.03293043..., 0], [0.00878058..., 0]]

    """

    if verbosity not in ['NONE', 'LOW', 'HIGH', 'ALL']:
        raise ValueError('Verbosity should be either NONE, LOW, HIGH or ALL.')

    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError('At least 1 convex function should be provided.')
    elif len(functions) == 1:
        functions.append(dummy())
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            print('INFO: Dummy objective function added.')

    # Choose a solver if none provided.
    if not solver:
        if len(functions) == 2:
            fb0 = 'GRAD' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            fb1 = 'GRAD' in functions[1].cap(x0) and \
                  'PROX' in functions[0].cap(x0)
            dg0 = 'PROX' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            if fb0 or fb1:
                solver = forward_backward()  # Need one prox and 1 grad.
            elif dg0:
                solver = douglas_rachford()  # Need two prox.
            else:
                raise ValueError('No suitable solver for the given functions.')
        elif len(functions) > 2:
            solver = generalized_forward_backward()
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('INFO: Selected solver: {}'.format(name))

    # Set solver and functions verbosity.
    translation = {'ALL': 'HIGH', 'HIGH': 'HIGH', 'LOW': 'LOW', 'NONE': 'NONE'}
    solver.verbosity = translation[verbosity]
    translation = {'ALL': 'HIGH', 'HIGH': 'LOW', 'LOW': 'NONE', 'NONE': 'NONE'}
    functions_verbosity = []
    for f in functions:
        functions_verbosity.append(f.verbosity)
        f.verbosity = translation[verbosity]

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)

    while not crit:

        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        if verbosity in ['HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('Iteration {} of {}:'.format(niter, name))

        # Solver iterative algorithm.
        solver.algo(objective, niter)

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol is not None and current < atol:
            crit = 'ATOL'
        if dtol is not None and np.abs(current - last) < dtol:
            crit = 'DTOL'
        if rtol is not None:
            div = current  # Prevent division by 0.
            if div == 0:
                if verbosity in ['LOW', 'HIGH', 'ALL']:
                    print('WARNING: objective function is equal to 0 !')
                if last != 0:
                    div = last
                else:
                    div = 1.0  # Result will be zero anyway.
            else:
                rtol_only_zeros = False
            relative = np.abs((current - last) / div)
            if relative < rtol and not rtol_only_zeros:
                crit = 'RTOL'
        if xtol is not None:
            err = np.linalg.norm(solver.sol - last_sol)
            err /= np.sqrt(last_sol.size)
            if err < xtol:
                crit = 'XTOL'
        if maxit is not None and niter >= maxit:
            crit = 'MAXIT'

        if verbosity in ['HIGH', 'ALL']:
            print('    objective = {:.2e}'.format(current))

    # Restore verbosity for functions. In case they are called outside solve().
    for k, f in enumerate(functions):
        f.verbosity = functions_verbosity[k]

    if verbosity in ['LOW', 'HIGH', 'ALL']:
        print('Solution found after {} iterations:'.format(niter))
        print('    objective function f(sol) = {:e}'.format(current))
        print('    stopping criterion: {}'.format(crit))

    # Returned dictionary.
    result = {'sol':       solver.sol,
              'solver':    solver.__class__.__name__,  # algo for consistency ?
              'crit':      crit,
              'niter':     niter,
              'time':      time.time() - tstart,
              'objective': objective}
    try:
        result['dual_sol'] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    return result


def _prox_star(func, z, T):
    r"""Proximity operator of the convex conjugate of a function."""
    return z - T * func.prox(z / T, 1 / T)


class solver(object):
    r"""
    Defines the solver object interface.

    This class defines the interface of a solver object intended to be passed
    to the :func:`pyunlocbox.solvers.solve` solving function. It is intended to
    be a base class for standard solvers which will implement the required
    methods. It can also be instantiated by user code and dynamically modified
    for rapid testing. This class also defines the generic attributes of all
    solver objects.

    Parameters
    ----------
    step : float
        The gradient-descent step-size. This parameter is bounded by 0 and
        :math:`\frac{2}{\beta}` where :math:`\beta` is the Lipschitz constant
        of the gradient of the smooth function (or a sum of smooth functions).
        Default is 1.
    post_step : function
        User defined function to post-process the step size. This function is
        called every iteration and permits the user to alter the solver
        algorithm. The user may start with a high step size and progressively
        lower it while the algorithm runs to accelerate the convergence. The
        function parameters are the following : `step` (current step size),
        `sol` (current problem solution), `objective` (list of successive
        evaluations of the objective function), `niter` (current iteration
        number). The function should return a new value for `step`. Default is
        to return an unchanged value.
    post_sol : function
        User defined function to post-process the problem solution. This
        function is called every iteration and permits the user to alter the
        solver algorithm. Same parameter as :func:`post_step`. Default is to
        return an unchanged value.
    """

    def __init__(self, step=1, post_step=None, post_sol=None):
        if step < 0:
            raise ValueError('Gamma should be a positive number.')
        self.step = step
        if post_step:
            self.post_step = post_step
        else:
            self.post_step = lambda step, sol, objective, niter: step
        if post_sol:
            self.post_sol = post_sol
        else:
            self.post_sol = lambda step, sol, objective, niter: sol

    def pre(self, functions, x0):
        """
        Solver specific initialization. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self.sol = np.asarray(x0)
        self._pre(functions, np.asarray(x0))

    def _pre(self, functions, x0):
        raise NotImplementedError("Class user should define this method.")

    def algo(self, objective, niter):
        """
        Call the solver iterative algorithm while allowing the user to alter
        it. This makes it possible to dynamically change the `step` step size
        while the algorithm is running.  See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._algo()
        self.step = self.post_step(self.step, self.sol, objective, niter)
        self.sol = self.post_sol(self.step, self.sol, objective, niter)

    def _algo(self):
        raise NotImplementedError("Class user should define this method.")

    def post(self):
        """
        Solver specific post-processing. Mainly used to delete references added
        during initialization so that the garbage collector can free the
        memory. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._post()
        del self.sol

    def _post(self):
        raise NotImplementedError("Class user should define this method.")


class forward_backward(solver):
    r"""
    Forward-backward proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    a smooth and a non-smooth function.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    method : {'FISTA', 'ISTA'}, optional
        The method used to solve the problem. It can be 'FISTA' or 'ISTA'. Note
        that while FISTA is much more time efficient, it is less memory
        efficient.  Default is 'FISTA'.
    lambda_ : float, optional
        The update term weight for ISTA. It should be between 0 and 1. Default
        is 1.

    Notes
    -----
    This algorithm requires one function to implement the
    :meth:`pyunlocbox.functions.func.prox` method and the other one to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

    See :cite:`beck2009FISTA` for details about the algorithm.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.forward_backward(method='FISTA', lambda_=1, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 12 iterations:
        objective function f(sol) = 4.135992e-06
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 3.99927529,  4.99909411,  5.99891293,  6.99873176])

    """

    def __init__(self, method='FISTA', lambda_=1, *args, **kwargs):
        super(forward_backward, self).__init__(*args, **kwargs)
        self.method = method
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.verbosity is 'HIGH':
            print('INFO: Forward-backward method: {}'.format(self.method))

        if self.lambda_ <= 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        if self.method is 'ISTA':
            self._algo = self._ista
        elif self.method is 'FISTA':
            self._algo = self._fista
            self.z = np.array(x0, copy=True)
            self.t = 1.
        else:
            raise ValueError('The method should be either FISTA or ISTA.')

        if len(functions) != 2:
            raise ValueError('Forward-backward requires two convex functions.')

        if 'PROX' in functions[0].cap(x0) and 'GRAD' in functions[1].cap(x0):
            self.f1 = functions[0]
            self.f2 = functions[1]
        elif 'PROX' in functions[1].cap(x0) and 'GRAD' in functions[0].cap(x0):
            self.f1 = functions[1]
            self.f2 = functions[0]
        else:
            raise ValueError('Forward-backward requires a function to '
                             'implement prox() and the other grad().')

    def _ista(self):
        x = self.sol - self.step * self.f2.grad(self.sol)
        self.sol[:] += self.lambda_ * (self.f1.prox(x, self.step) - self.sol)

    def _fista(self):
        x = self.z - self.step * self.f2.grad(self.z)
        x[:] = self.f1.prox(x, self.step)
        tn = (1. + np.sqrt(1. + 4. * self.t**2.)) / 2.
        self.z[:] = x + (self.t - 1.) / tn * (x - self.sol)
        self.t = tn
        self.sol[:] = x

    def _post(self):
        del self._algo, self.f1, self.f2
        if self.method is 'FISTA':
            del self.z, self.t


class generalized_forward_backward(solver):
    r"""
    Generalized forward-backward proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    any number of non-smooth (or smooth) functions.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    lambda_ : float, optional
        A relaxation parameter bounded by 0 and 1. Default is 1.

    Notes
    -----
    This algorithm requires each function to either implement the
    :meth:`pyunlocbox.functions.func.prox` method or the
    :meth:`pyunlocbox.functions.func.grad` method.

    See :cite:`raguet2013generalizedFB` for details about the algorithm.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = [0.01, 0.2, 8, 0.3, 0 , 0.03, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.norm_l1()
    >>> solver = solvers.generalized_forward_backward(lambda_=1, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver)
    Solution found after 2 iterations:
        objective function f(sol) = 1.463100e+01
        stopping criterion: RTOL
    >>> ret['sol']
    array([ 0. ,  0. ,  7.5,  0. ,  0. ,  0. ,  6.5])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(generalized_forward_backward, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.lambda_ <= 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        self.f = []  # Smooth functions.
        self.g = []  # Non-smooth functions.
        self.z = []
        for f in functions:
            if 'GRAD' in f.cap(x0):
                self.f.append(f)
            elif 'PROX' in f.cap(x0):
                self.g.append(f)
                self.z.append(np.array(x0, copy=True))
            else:
                raise ValueError('Generalized forward-backward requires each '
                                 'function to implement prox() or grad().')

        if self.verbosity is 'HIGH':
            print('INFO: Generalized forward-backward minimizing {} smooth '
                  'functions and {} non-smooth functions.'.format(len(self.f),
                                                                  len(self.g)))

    def _algo(self):

        # Smooth functions.
        grad = np.zeros(self.sol.shape)
        for f in self.f:
            grad += f.grad(self.sol)

        # Non-smooth functions.
        if not self.g:
            self.sol[:] -= self.step * grad  # Reduces to gradient descent.
        else:
            sol = np.zeros(self.sol.shape)
            for i, g in enumerate(self.g):
                tmp = 2 * self.sol - self.z[i] - self.step * grad
                tmp[:] = g.prox(tmp, self.step * len(self.g))
                self.z[i] += self.lambda_ * (tmp - self.sol)
                sol += 1. * self.z[i] / len(self.g)
            self.sol[:] = sol

    def _post(self):
        del self.f, self.g, self.z


class douglas_rachford(solver):
    r"""
    Douglas-Rachford proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    two non-smooth (or smooth) functions.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    lambda_ : float, optional
        The update term weight. It should be between 0 and 1. Default is 1.

    Notes
    -----
    This algorithm requires the two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method.

    See :cite:`combettes2007DR` for details about the algorithm.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.douglas_rachford(lambda_=1, step=1)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 8 iterations:
        objective function f(sol) = 2.927052e-06
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 3.99939034,  4.99923792,  5.99908551,  6.99893309])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(douglas_rachford, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.lambda_ <= 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        if len(functions) != 2:
            raise ValueError('Douglas-Rachford requires two convex functions.')

        self.f1 = functions[0]
        self.f2 = functions[1]

        self.z = np.array(x0, copy=True)

    def _algo(self):
        tmp = self.f1.prox(2 * self.sol - self.z, self.step)
        self.z[:] = self.z + self.lambda_ * (tmp - self.sol)
        self.sol[:] = self.f2.prox(self.z, self.step)

    def _post(self):
        del self.f1, self.f2, self.z


class primal_dual(solver):
    r"""
    Parent class of all primal-dual algorithms.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    L : function or ndarray, optional
        The transformation L that maps from the primal variable space to the
        dual variable space. Default is the identity, :math:`L(x)=x`. If `L` is
        an ``ndarray``, it will be converted to the operator form.
    Lt : function or ndarray, optional
        The adjoint operator. If `Lt` is an ``ndarray``, it will be converted
        to the operator form. If `L` is an ``ndarray``, default is the
        transpose of `L`. If `L` is a function, default is `L`,
        :math:`Lt(x)=L(x)`.
    d0: ndarray, optional
        Initialization of the dual variable.

    """

    def __init__(self, L=None, Lt=None, d0=None, *args, **kwargs):
        super(primal_dual, self).__init__(*args, **kwargs)

        if L is None:
            self.L = lambda x: x
        else:
            if callable(L):
                self.L = L
            else:
                # Transform matrix form to operator form.
                self.L = lambda x: np.dot(L, x)

        if Lt is None:
            if L is None:
                self.Lt = lambda x: x
            elif callable(L):
                self.Lt = L
            else:
                self.Lt = lambda x: np.dot(np.transpose(L), x)
        else:
            if callable(Lt):
                self.Lt = Lt
            else:
                self.Lt = lambda x: np.dot(Lt, x)

        self.d0 = d0

    def _pre(self, functions, x0):
        # Dual variable.
        if self.d0 is None:
            self.dual_sol = self.L(x0)
        else:
            self.dual_sol = self.d0

    def _post(self):
        del self.dual_sol, self.d0


class mlfbf(primal_dual):
    r"""
    Monotone + Lipschitz Forward-Backward-Forward primal-dual algorithm.

    This algorithm solves convex optimization problems with objective of the
    form :math:`f(x) + g(Lx) + h(x)`, where :math:`f` and :math:`g` are proper,
    convex, lower-semicontinuous functions with easy-to-compute proximity
    operators, and :math:`h` has Lipschitz-continuous gradient with constant
    :math:`\beta`.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.primal_dual` base class.

    Notes
    -----
    The order of the functions matters: set :math:`f` first on the list,
    :math:`g` second, and :math:`h` third.

    This algorithm requires the first two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method, and the third function to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

    The step-size should be in the interval :math:`\left] 0, \frac{1}{\beta +
    \|L\|_{2}}\right[`.

    See :cite:`komodakis2015primaldual`, Algorithm 6, for details.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = np.array([294, 390, 361])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> x0 = np.zeros(len(y))
    >>> f = functions.dummy()
    >>> f._prox = lambda x, T: np.maximum(np.zeros(len(x)), x)
    >>> g = functions.norm_l2(lambda_=0.5)
    >>> h = functions.norm_l2(y=y, lambda_=0.5)
    >>> max_step = 1/(1 + np.linalg.norm(L, 2))
    >>> solver = solvers.mlfbf(L=L, step=max_step/2.)
    >>> ret = solvers.solve([f, g, h], x0, solver, maxit=1000, rtol=0)
    Solution found after 1000 iterations:
        objective function f(sol) = 1.833865e+05
        stopping criterion: MAXIT
    >>> ret['sol']
    array([ 1.,  1.,  1.])

    """

    def _pre(self, functions, x0):
        super(mlfbf, self)._pre(functions, x0)

        if len(functions) != 3:
            raise ValueError('MLFBF requires 3 convex functions.')

        self.f = functions[0]
        self.g = functions[1]
        self.h = functions[2]

    def _algo(self):
        y1 = self.sol - self.step * (self.h.grad(self.sol) +
                                     self.Lt(self.dual_sol))
        y2 = self.dual_sol + self.step * self.L(self.sol)
        p1 = self.f.prox(y1, self.step)
        p2 = _prox_star(self.g, y2, self.step)
        q1 = p1 - self.step * (self.h.grad(p1) + self.Lt(p2))
        q2 = p2 + self.step * self.L(p1)
        self.sol[:] = self.sol - y1 + q1
        self.dual_sol[:] = self.dual_sol - y2 + q2

    def _post(self):
        super(mlfbf, self)._post()
        del self.f, self.g, self.h


class projection_based(primal_dual):
    r"""
    Projection-based primal-dual algorithm.

    This algorithm solves convex optimization problems with objective of the
    form :math:`f(x) + g(Lx)`, where :math:`f` and :math:`g` are proper,
    convex, lower-semicontinuous functions with easy-to-compute proximity
    operators.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.primal_dual` base class.

    Parameters
    ----------
    lambda_ : float, optional
        The update term weight. It should be between 0 and 2. Default is 1.

    Notes
    -----
    The order of the functions matters: set :math:`f` first on the list, and
    :math:`g` second.

    This algorithm requires the two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method.

    The step-size should be in the interval :math:`\left] 0, \infty \right[`.

    See :cite:`komodakis2015primaldual`, Algorithm 7, for details.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = np.array([294, 390, 361])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> x0 = np.array([500, 1000, -400])
    >>> f = functions.norm_l1(y=y)
    >>> g = functions.norm_l1()
    >>> solver = solvers.projection_based(L=L, step=1.)
    >>> ret = solvers.solve([f, g], x0, solver, maxit=1000, rtol=None, xtol=.1)
    Solution found after 996 iterations:
        objective function f(sol) = 1.045000e+03
        stopping criterion: XTOL
    >>> ret['sol']
    array([0, 0, 0])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(projection_based, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):
        super(projection_based, self)._pre(functions, x0)

        if self.lambda_ <= 0 or self.lambda_ > 2:
            raise ValueError('Lambda is bounded by 0 and 2.')

        if len(functions) != 2:
            raise ValueError('projection_based requires 2 convex functions.')

        self.f = functions[0]
        self.g = functions[1]

    def _algo(self):
        a = self.f.prox(self.sol - self.step *
                        self.Lt(self.dual_sol), self.step)
        l = self.L(self.sol)
        b = self.g.prox(l + self.step * self.dual_sol, self.step)
        s = (self.sol - a) / self.step + self.Lt(l - b) / self.step
        t = b - self.L(a)
        tau = np.sum(s**2) + np.sum(t**2)
        if tau == 0:
            self.sol[:] = a
            self.dual_sol[:] = self.dual_sol + (l - b) / self.step
        else:
            theta = self.lambda_ * (np.sum((self.sol - a)**2) / self.step +
                                    np.sum((l - b)**2) / self.step) / tau
            self.sol[:] = self.sol - theta * s
            self.dual_sol[:] = self.dual_sol - theta * t

    def _post(self):
        super(projection_based, self)._post()
        del self.f, self.g
