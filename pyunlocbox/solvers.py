# -*- coding: utf-8 -*-

r"""
The :mod:`pyunlocbox.solvers` module implements a solving function (which will
minimize your objective function) as well as common solvers.

Solving
-------

Call :func:`solve` to solve your convex optimization problem using your
instantiated solver and functions objects.

Interface
---------

The :class:`solver` base class defines a common interface to all solvers:

.. autosummary::

    solver.pre
    solver.algo
    solver.post

Solvers
-------

Then, derived classes implement various common solvers.

.. autosummary::

    gradient_descent
    forward_backward
    douglas_rachford
    generalized_forward_backward

**Primal-dual solvers** (based on :class:`primal_dual`)

.. autosummary::

    mlfbf
    projection_based

.. inheritance-diagram:: pyunlocbox.solvers
    :parts: 2

"""

import time

import numpy as np

from pyunlocbox.functions import dummy, _prox_star
from pyunlocbox import acceleration


def solve(functions, x0, solver=None, atol=None, dtol=None, rtol=1e-3,
          xtol=None, maxit=200, verbosity='LOW', inplace=False):
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
        N}`.
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
    inplace : bool, optional
        If True and x0 is a numpy array, then x0 will be modified in place
        during execution to save memory. It will then contain the solution. Be
        careful to pass data of the type (int, float32, float64) you want your
        computations to use.

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
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers

    Define a problem:

    >>> y = [4, 5, 6, 7]
    >>> f = functions.norm_l2(y=y)

    Solve it:

    >>> x0 = np.zeros(len(y))
    >>> ret = solvers.solve([f], x0, atol=1e-2, verbosity='ALL')
    INFO: Dummy objective function added.
    INFO: Selected solver: forward_backward
    INFO: Forward-backward method
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 1.260000e+02
    Iteration 1 of forward_backward:
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 1.400000e+01
        objective = 1.40e+01
    Iteration 2 of forward_backward:
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 2.963739e-01
        objective = 2.96e-01
    Iteration 3 of forward_backward:
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 7.902529e-02
        objective = 7.90e-02
    Iteration 4 of forward_backward:
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 5.752265e-02
        objective = 5.75e-02
    Iteration 5 of forward_backward:
        dummy evaluation: 0.000000e+00
        norm_l2 evaluation: 5.142032e-03
        objective = 5.14e-03
    Solution found after 5 iterations:
        objective function f(sol) = 5.142032e-03
        stopping criterion: ATOL

    Verify the stopping criterion (should be smaller than atol=1e-2):

    >>> np.linalg.norm(ret['sol'] - y)**2  # doctest:+ELLIPSIS
    0.00514203...

    Show the solution (should be close to y w.r.t. the L2-norm measure):

    >>> ret['sol']
    array([4.02555301, 5.03194126, 6.03832952, 7.04471777])

    Show the used solver:

    >>> ret['solver']
    'forward_backward'

    Show some information about the convergence:

    >>> ret['crit']
    'ATOL'
    >>> ret['niter']
    5
    >>> ret['time']  # doctest:+SKIP
    0.0012578964233398438
    >>> ret['objective']  # doctest:+NORMALIZE_WHITESPACE,+ELLIPSIS
    [[126.0, 0], [13.99999999..., 0], [0.29637392..., 0], [0.07902528..., 0],
    [0.05752265..., 0], [0.00514203..., 0]]

    """
    # to prevent any modification of the input
    if not(inplace):
        x0 = x0.copy()

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
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)

    # Evaluate the objective function at the begining
    objective = [solver.objective(x0)]

    while not crit:

        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        if verbosity in ['HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('Iteration {} of {}:'.format(niter, name))

        # Solver iterative algorithm.
        solver.algo(objective, niter)

        objective.append(solver.objective(solver.sol))
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
                    print('WARNING: (rtol) objective function is equal to 0 !')
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
        # Update dictionary for primal-dual solvers
        result['dual_sol'] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    return result


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
    accel : pyunlocbox.acceleration.accel
        User-defined object used to adaptively change the current step size
        and solution while the algorithm is running. Default is a dummy
        object that returns unchanged values.

    """

    def __init__(self, step=1., accel=None):
        if step < 0:
            raise ValueError('Step should be a positive number.')
        self.step = step
        self.accel = acceleration.dummy() if accel is None else accel

    def pre(self, functions, x0):
        """
        Solver-specific pre-processing. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.

        Notes
        -----
        When preprocessing the functions, the solver should split them into
        two lists:
        * `self.smooth_funs`, for functions involved in gradient steps.
        * `self.non_smooth_funs`, for functions involved proximal steps.
        This way, any method that takes in the solver as argument, such as the
        methods in :class:`pyunlocbox.acceleration.accel`, can have some
        context as to how the solver is using the functions.

        """
        self.sol = np.asarray(x0)
        self.smooth_funs = []
        self.non_smooth_funs = []
        self._pre(functions, self.sol)
        self.accel.pre(functions, self.sol)

    def _pre(self, functions, x0):
        raise NotImplementedError("Class user should define this method.")

    def algo(self, objective, niter):
        """
        Call the solver iterative algorithm and the provided acceleration
        scheme. See parameters documentation in
        :func:`pyunlocbox.solvers.solve`

        Notes
        -----
        The method :meth:`self.accel.update_sol` is called before
        :meth:`self._algo` because the acceleration schemes usually involves
        some sort of averaging of previous solutions, which can add some
        unwanted artifacts on the output solution. With this ordering, we
        guarantee that the output of solver.algo is not corrupted by the
        acceleration scheme.

        Similarly, the method :meth:`self.accel.update_step` is called after
        :meth:`self._algo` to allow the step update procedure to act directly
        on the solution output by the underlying algorithm, and not on the
        intermediate solution output by the acceleration scheme in
        :meth:`self.accel.update_sol`.

        """
        self.sol[:] = self.accel.update_sol(self, objective, niter)
        self.step = self.accel.update_step(self, objective, niter)
        self._algo()

    def _algo(self):
        raise NotImplementedError("Class user should define this method.")

    def post(self):
        """
        Solver-specific post-processing. Mainly used to delete references added
        during initialization so that the garbage collector can free the
        memory. See parameters documentation in
        :func:`pyunlocbox.solvers.solve`.

        """
        self._post()
        self.accel.post()
        del self.sol, self.smooth_funs, self.non_smooth_funs

    def _post(self):
        raise NotImplementedError("Class user should define this method.")

    def objective(self, x):
        """
        Return the objective function at x.

        Necessitate `solver._pre(...)` to be run first.
        """
        return self._objective(x)

    def _objective(self, x):
        obj_smooth = [f.eval(x) for f in self.smooth_funs]
        obj_nonsmooth = [f.eval(x) for f in self.non_smooth_funs]
        return obj_nonsmooth + obj_smooth


class gradient_descent(solver):
    r"""
    Gradient descent algorithm.

    This algorithm solves optimization problems composed of the sum of
    any number of smooth functions.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Notes
    -----
    This algorithm requires each function implement the
    :meth:`pyunlocbox.functions.func.grad` method.

    See :class:`pyunlocbox.acceleration.regularized_nonlinear` for a very
    efficient acceleration scheme for this method.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> dim = 25
    >>> np.random.seed(0)
    >>> xstar = np.random.rand(dim)  # True solution
    >>> x0 = np.random.rand(dim)
    >>> x0 = xstar + 5*(x0 - xstar) / np.linalg.norm(x0 - xstar)
    >>> A = np.random.rand(dim, dim)
    >>> step = 1 / np.linalg.norm(np.dot(A.T, A))
    >>> f = functions.norm_l2(lambda_=0.5, A=A, y=np.dot(A, xstar))
    >>> fd = functions.dummy()
    >>> solver = solvers.gradient_descent(step=step)
    >>> params = {'rtol':0, 'maxit':14000, 'verbosity':'NONE'}
    >>> ret = solvers.solve([f, fd], x0, solver, **params)
    >>> pctdiff = 100 * np.sum((xstar - ret['sol'])**2) / np.sum(xstar**2)
    >>> print('Difference: {0:.1f}%'.format(pctdiff))
    Difference: 1.3%

    """

    def __init__(self, **kwargs):
        super(gradient_descent, self).__init__(**kwargs)

    def _pre(self, functions, x0):

        for f in functions:
            if 'GRAD' in f.cap(x0):
                self.smooth_funs.append(f)
            else:
                raise ValueError('Gradient descent requires each function to '
                                 'implement grad().')

        if self.verbosity == 'HIGH':
            print('INFO: Gradient descent minimizing {} smooth '
                  'functions.'.format(len(self.smooth_funs)))

    def _algo(self):
        """
        x^{k+1} = x^k-λ ∇x^k
        """
        grad = np.zeros_like(self.sol)
        for f in self.smooth_funs:
            grad += f.grad(self.sol)
        self.sol[:] -= self.step * grad

    def _post(self):
        pass


class forward_backward(solver):
    r"""
    Forward-backward proximal splitting (FISTA and ISTA) algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    a smooth and a non-smooth function.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    accel : :class:`pyunlocbox.acceleration.accel`
        Acceleration scheme to use.
        Default is :meth:`pyunlocbox.acceleration.fista`, which corresponds
        to the 'FISTA' solver. Passing :meth:`pyunlocbox.acceleration.dummy`
        instead results in the ISTA solver. Note that while FISTA is much more
        time-efficient, it is less memory-efficient.

    Notes
    -----
    This algorithm requires one function to implement the
    :meth:`pyunlocbox.functions.func.prox` method and the other one to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

    See :cite:`beck2009FISTA` for details about the algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.forward_backward(step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 15 iterations:
        objective function f(sol) = 4.957288e-07
        stopping criterion: ATOL
    >>> ret['sol']
    array([4.0002509 , 5.00031362, 6.00037635, 7.00043907])

    """

    def __init__(self, accel=acceleration.fista(), **kwargs):
        super(forward_backward, self).__init__(accel=accel, **kwargs)

    def _pre(self, functions, x0):

        if self.verbosity == 'HIGH':
            print('INFO: Forward-backward method')

        if len(functions) != 2:
            raise ValueError('Forward-backward requires two convex functions.')

        if 'PROX' in functions[0].cap(x0) and 'GRAD' in functions[1].cap(x0):
            self.smooth_funs.append(functions[1])
            self.non_smooth_funs.append(functions[0])
        elif 'PROX' in functions[1].cap(x0) and 'GRAD' in functions[0].cap(x0):
            self.smooth_funs.append(functions[0])
            self.non_smooth_funs.append(functions[1])
        else:
            raise ValueError('Forward-backward requires a function to '
                             'implement prox() and the other grad().')

    def _algo(self):
        # Forward step
        x = self.sol - self.step * self.smooth_funs[0].grad(self.sol)
        # Backward step
        self.sol[:] = self.non_smooth_funs[0].prox(x, self.step)

    def _post(self):
        pass


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
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
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
    array([0. , 0. , 7.5, 0. , 0. , 0. , 6.5])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(generalized_forward_backward, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.lambda_ <= 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        self.z = []
        for f in functions:
            if 'GRAD' in f.cap(x0):
                self.smooth_funs.append(f)
            elif 'PROX' in f.cap(x0):
                self.non_smooth_funs.append(f)
                self.z.append(np.array(x0, copy=True))
            else:
                raise ValueError('Generalized forward-backward requires each '
                                 'function to implement prox() or grad().')

        if self.verbosity == 'HIGH':
            print('INFO: Generalized forward-backward minimizing {} smooth '
                  'functions and {} non-smooth functions.'.format(
                      len(self.smooth_funs), len(self.non_smooth_funs)))

    def _algo(self):

        # Smooth functions.
        grad = np.zeros_like(self.sol)
        for f in self.smooth_funs:
            grad += f.grad(self.sol)

        # Non-smooth functions.
        if not self.non_smooth_funs:
            self.sol[:] -= self.step * grad  # Reduces to gradient descent.
        else:
            sol = np.zeros_like(self.sol)
            for i, g in enumerate(self.non_smooth_funs):
                tmp = 2 * self.sol - self.z[i] - self.step * grad
                tmp[:] = g.prox(tmp, self.step * len(self.non_smooth_funs))
                self.z[i] += self.lambda_ * (tmp - self.sol)
                sol += 1. * self.z[i] / len(self.non_smooth_funs)
            self.sol[:] = sol

    def _post(self):
        del self.z


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
    A       : numpy array, optional
        Matrix implementing a linear transformation of x in g() as : minimize f(x) + g(Ax)

    Notes
    -----
    This algorithm requires the two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method.

    See :cite:`combettes2007DR` for details about the algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.douglas_rachford(step=1)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 8 iterations:
        objective function f(sol) = 2.927052e-06
        stopping criterion: ATOL
    >>> ret['sol']
    array([3.99939034, 4.99923792, 5.99908551, 6.99893309])

    --- Linearized ADMM ---
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = np.array([4,-9,-13,-4])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> max_step = 0.5/(1 + np.linalg.norm(L, 2))
    >>> x0 = np.zeros(3)
    >>> f1 = functions.norm_l1()
    >>> f2 = functions.norm_l1(y=y)
    >>> solver = solvers.douglas_rachford(step=max_step*50, A=L)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-1, maxit=1000, rtol=1e-5)
    Solution found after 993 iterations:
        objective function f(sol) = 8.008191e+00
        stopping criterion: RTOL
    >>> ret['sol']
    array([-4.00133346  3.00096956 -0.99996531])

    """

    def __init__(self, A=None, mu=None, *args, **kwargs):
        super(douglas_rachford, self).__init__(*args, **kwargs)

        if A is None:
            self.A = lambda x: x
            self.At = lambda x: x 
        else:
            # Transform matrix form to operator form.
            self.A = lambda x: A.dot(x)
            self.At = lambda x: A.T.dot(x)

        self.mu=0.5
        if (mu is None and A is not None):
            self.mu = self.step/(np.linalg.norm(A,2)**2)

    def _pre(self, functions, x0):

        if self.mu <= 0 or self.mu > 1:
            raise ValueError('Mu is bounded by 0 and 1.')

        if len(functions) != 2:
            raise ValueError('Douglas-Rachford requires two convex functions.')

        for f in functions:
            x1 = np.copy(x0)

            try :
                f.cap(x1)
            except ValueError:
                x1 = self.A(x0)

            if 'PROX' in f.cap(x1):
                self.non_smooth_funs.append(f)
            else:
                raise ValueError('Douglas-Rachford requires each '
                                'function to implement prox().')

        self.z = np.array(self.A(x0), copy=True)
        self.u = np.array(self.A(x0), copy=True)

    def _algo(self):
        """
        Default:
            x^{k+1} = prox_{λf} (z^k)
            z^{k+1} = z^k + prox_{λg}(2x^{k+1}−z^k) − x^{k+1}
        Or equivalently:
            z^{k+1} = prox_{λg} (x^k+u^k)
            x^{k+1} = prox_{λf} (z^{k+1}−u^k)
            u^{k+1} = u^k+x^{k+1}−z^{k+1}

        If linearized:
            z^{k+1} = prox_{λg} (Ax^k+u^k)
            x^{k+1} = prox_{µf} (x^k−(µ/λ)A^T(Ax^k−z^{k+1}+u^k))
            u^{k+1} = u^k+Ax^{k+1}−z^{k+1}

        """
        # if (self.A is None):
        #     tmp = self.non_smooth_funs[0].prox(2 * self.sol - self.z, self.step)
        #     self.z[:] = self.z + self.lambda_ * (tmp - self.sol)        # prox_{λg}(y) != λ prox_{g}(y)
        #     self.sol[:] = self.non_smooth_funs[1].prox(self.z, self.step)

        #     # self.z[:] = self.non_smooth_funs[0].prox(self.sol + self.u, self.step)
        #     # self.sol[:] = self.non_smooth_funs[1].prox(self.z-self.u, self.step)
        #     # self.u[:] = self.u + self.sol - self.z

        # else : # See "Proximal Algorithms. N. Parikh and S. Boyd. Foundations and Trends in Optimization, 1(3):123-231, 2014." 
        self.z[:] = self.non_smooth_funs[1].prox(self.A(self.sol) + self.u, self.step)
        self.sol[:] = self.non_smooth_funs[0].prox(self.sol-(self.mu/self.step)*self.At(self.A(self.sol)-self.z+self.u), self.mu)
        self.u[:] = self.u + self.A(self.sol) - self.z

    def _objective(self, x):
        obj_smooth = [f.eval(x) for f in self.smooth_funs]
        obj_nonsmooth = [self.non_smooth_funs[0].eval(x), 
                         self.non_smooth_funs[1].eval(self.A(x))]
        return obj_nonsmooth + obj_smooth

    def _post(self):
        del self.z
        del self.u


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
                self.L = lambda x: L.dot(x)

        if Lt is None:
            if L is None:
                self.Lt = lambda x: x
            elif callable(L):
                self.Lt = L
            else:
                self.Lt = lambda x: L.T.dot(x)
        else:
            if callable(Lt):
                self.Lt = Lt
            else:
                self.Lt = lambda x: Lt.dot(x)

        self.d0 = d0

    def _pre(self, functions, x0):
        # Dual variable.
        if self.d0 is None:
            # The copy is necessary in case `L = lambda x: x`.
            self.dual_sol = self.L(np.asarray(x0).copy())
        else:
            self.dual_sol = self.d0

    def _post(self):
        self.d0 = None
        del self.dual_sol

    def _objective(self, x):
        obj_smooth = [f.eval(x) for f in self.smooth_funs]
        obj_nonsmooth = [self.non_smooth_funs[0].eval(x),
                         self.non_smooth_funs[1].eval(self.L(x))]
        return obj_nonsmooth + obj_smooth


class mlfbf(primal_dual):
    r"""
    Monotone+Lipschitz forward-backward-forward primal-dual algorithm.

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
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
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
        objective function f(sol) = 1.839060e+05
        stopping criterion: MAXIT
    >>> ret['sol']
    array([1., 1., 1.])

    """

    def _pre(self, functions, x0):
        super(mlfbf, self)._pre(functions, x0)

        if len(functions) != 3:
            raise ValueError('MLFBF requires 3 convex functions.')

        self.non_smooth_funs.append(functions[0])   # f
        self.non_smooth_funs.append(functions[1])   # g
        self.smooth_funs.append(functions[2])       # h

    def _algo(self):
        """
            y1 = x^k - λ ∇h(x^k) + L^T(z^k)
            y2 = z^k + λ L(x^k)

            p1 = prox_{λf} y1
            p2 = prox_{λg^*} y2

            q1 = p1 - λ ∇h(p1) + L^T(p2)
            q2 = p2 + λ L(p1)

            with x^k (z^k) the solution (dual-solution) at iteration k.
        """
        # Forward steps (in both primal and dual spaces)
        y1 = self.sol - self.step * (self.smooth_funs[0].grad(self.sol) +
                                     self.Lt(self.dual_sol))
        y2 = self.dual_sol + self.step * self.L(self.sol)

        # Backward steps (in both primal and dual spaces)
        p1 = self.non_smooth_funs[0].prox(y1, self.step)
        p2 = _prox_star(self.non_smooth_funs[1], y2, self.step)

        # Forward steps (in both primal and dual spaces)
        q1 = p1 - self.step * (self.smooth_funs[0].grad(p1) + self.Lt(p2))
        q2 = p2 + self.step * self.L(p1)

        # Update solution (in both primal and dual spaces)
        self.sol[:] = self.sol - y1 + q1
        self.dual_sol[:] = self.dual_sol - y2 + q2


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
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
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

    def __init__(self, lambda_=1., *args, **kwargs):
        super(projection_based, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):
        super(projection_based, self)._pre(functions, x0)

        if self.lambda_ <= 0 or self.lambda_ > 2:
            raise ValueError('Lambda is bounded by 0 and 2.')

        if len(functions) != 2:
            raise ValueError('projection_based requires 2 convex functions.')

        self.non_smooth_funs.append(functions[0])   # f
        self.non_smooth_funs.append(functions[1])   # g

    def _algo(self):
        a = self.non_smooth_funs[0].prox(self.sol - self.step *
                                         self.Lt(self.dual_sol), self.step)
        ell = self.L(self.sol)
        b = self.non_smooth_funs[1].prox(ell + self.step * self.dual_sol,
                                         self.step)
        s = (self.sol - a) / self.step + self.Lt(ell - b) / self.step
        t = b - self.L(a)
        tau = np.sum(s**2) + np.sum(t**2)
        if tau == 0:
            self.sol[:] = a
            self.dual_sol[:] = self.dual_sol + (ell - b) / self.step
        else:
            theta = self.lambda_ * (np.sum((self.sol - a)**2) / self.step +
                                    np.sum((ell - b)**2) / self.step) / tau
            self.sol[:] = self.sol - theta * s
            self.dual_sol[:] = self.dual_sol - theta * t


class chambolle_pock(primal_dual):
    r"""
    Primal-Dual Proximal Splitting.

    This algorithm solves convex optimization problems with objective of the
    form :math:`G(x) + F(Lx)`, where :math:`F` and :math:`G` are proper,
    convex, lower-semicontinuous functions with easy-to-compute proximity
    operators, and :math:`L` is a linear operator.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.primal_dual` base class.

    Notes
    -----
    The order of the functions matters: set :math:`G` first on the list and
    :math:`F` second.

    This algorithm requires the two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method.

    The step-size should be in the interval :math:`\left] 0, \frac{1}{\beta + \|L\|_{2}}\right[`.

    See :cite:`Antonin Chambolle and Thomas Pock: A First-order primal-dual algorithm for convex problems
     with application to imaging, Journal of Mathematical Imaging and Vision, Volume 40, Number 1 (2011), 120-145" for details.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = np.array([4,-9,-13,-4])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> max_step = 1/(1 + np.linalg.norm(L, 2))
    >>> x0 = np.array([0,0,0])
    >>> f = functions.norm_l1(y=y)
    >>> g = functions.norm_l1()
    >>> solver = solvers.chambolle_pock(L=L, sigma=max_step/2., theta=max_step/2., tau=max_step/2.)
    >>> ret = solvers.solve([g, f], x0, solver, maxit=1000, rtol=None, xtol=None)
    >>> print ('Chambolle-Pock solution : ', ret['sol'])
    Solution found after 1000 iterations:
        objective function f(sol) = 8.000000e+00
        stopping criterion: MAXIT
    Chambolle-Pock solution :  [-4  3 -1]

    """
    def __init__(self, sigma=1., tau=1., theta=1., accel=None, *args, **kwargs):
        super(chambolle_pock, self).__init__(*args, **kwargs)

        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        # self.accel = acceleration.dummy() if accel is None else accel

    def _pre(self, functions, x0):
        super(chambolle_pock, self)._pre(functions, x0)

        if self.tau <= 0 or self.tau > 2:
            raise ValueError('tau is bounded by 0 and 2.')
        if self.sigma <= 0 or self.sigma > 2:
            raise ValueError('sigma is bounded by 0 and 2.')
        if self.theta <= 0 or self.theta > 2:
            raise ValueError('theta is bounded by 0 and 2.')

        if len(functions) != 2:
            raise ValueError('Chambolle-Pock requires 2 functions.')

        self.non_smooth_funs.append(functions[0])   # F
        self.non_smooth_funs.append(functions[1])   # G

        # Initializations
        self.f = np.array(x0, copy=True)
        self.g = np.array(self.L(x0), copy=True)

    def _algo(self):
        """
        g_{k+1} = prox_{\sigma F^∗} (g_k+ \sigma*L(\tilde{f}_k)) 
        f_{k+1} = prox_{\tau G} (f_k−\tau L^∗(g_{k+1}) ) 
        \tilde{f}_{k+1} = f{k+1}+ \theta*(f_{k+1} − f_k)
        """
        # Backward steps 
        self.g = _prox_star(self.non_smooth_funs[1], self.g + self.sigma*self.L(self.sol), self.sigma) 
        self.fp1 = self.non_smooth_funs[0].prox(self.f - self.tau*self.Lt(self.g), self.tau)

        # Update solution 
        self.sol[:] = self.fp1 + self.theta*( self.fp1 - self.f )

        # update
        self.f = np.copy(self.fp1)

    def _post(self):
        del self.g, self.fp1, self.f