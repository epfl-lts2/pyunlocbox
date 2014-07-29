# -*- coding: utf-8 -*-

r"""
This module implements solver objects who minimize an objective function. Call
:func:`solve` to solve your convex optimization problem using your instantiated
solver and functions objects. The :class:`solver` base class defines the
interface of all solver objects. The specialized solver objects inherit from
it and implement the class methods. The following solvers are included :

* :class:`forward_backward`: Forward-backward proximal splitting algorithm.
* :class:`douglas_rachford`: Douglas-Rachford proximal splitting algorithm.
"""

import numpy as np
import time
from pyunlocbox.functions import dummy


def solve(functions, x0, solver=None, rtol=1e-3, atol=float('-inf'),
          convergence_speed=float('-inf'), maxit=200, verbosity='low'):
    r"""
    Solve an optimization problem whose objective function is the sum of some
    convex functions.

    This function minimizes the objective function :math:`f(x) =
    \sum\limits_{m=0}^{m=M} f_m(x)`, i.e. solves
    :math:`\operatorname{arg\,min}\limits_x f(x)` for
    :math:`x \in \mathbb{R}^N` using whatever algorithm. It returns a
    dictionary with the found solution and some informations about the
    algorithm execution.

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
        Starting point of the algorithm, :math:`x_0 \in \mathbb{R}^N`.
    solver : solver class instance, optional
        The solver algorithm. It is an object who must inherit from
        :class:`pyunlocbox.solvers.solver` and implement the :meth:`_pre`,
        :meth:`_algo` and :meth:`_post` methods. If no solver object are
        provided, a standard one will be chosen given the number of convex
        function objects and their implemented methods.
    rtol : float, optional
        The convergence (relative tolerance) stopping criterion. The algorithm
        stops if :math:`\left|\frac{n(k-1)-n(k)}{n(k)}\right|<rtol` where
        :math:`n(k)=f(x)` is the objective function at iteration :math:`k`.
        Default is :math:`10^{-3}`.
    atol : float, optional
        The absolute tolerance stopping criterion. The algorithm stops if
        :math:`n(k)<atol`. Default is minus infinity.
    convergence_speed : float, optional
        The minimum tolerable convergence speed of the objective function. The
        algorithm stops if n(k-1) - n(k) < `convergence_speed`. Default is
        minus infinity (i.e. the objective function may even increase).
    maxit : int, optional
        The maximum number of iterations. Default is 200.
    verbosity : {'low', 'high', 'none'}, optional
        The log level : ``'none'`` for no log, ``'low'`` for resume at
        convergence, ``'high'`` for info at all steps. Default is ``'low'``.

    Returns
    -------
    sol : ndarray
        problem solution
    solver : str
        used solver
    niter : int
        number of iterations
    time : float
        execution time in seconds
    eval : float
        final evaluation of the objective function :math:`f(x)`
    crit : {'MAX_IT', 'ABS_TOL', 'REL_TOL', 'CONV_SPEED'}
        Used stopping criterion. 'MAX_IT' if the maximum number of iterations
        `maxIter` is reached, 'ABS_TOL' if the objective function value is
        smaller than `absTol`, 'REL_TOL' if the relative objective function
        improvement was smaller than `relTol` (i.e. the algorithm converged),
        'CONV_SPEED' if the objective function improvement is smaller than
        `convergence_speed`.
    rel : float
        relative objective improvement at convergence
    objective : ndarray
        successive evaluations of the objective function at each iteration

    Examples
    --------

    Simple example showing the automatic selection of a solver (and a second
    function) :

    >>> import pyunlocbox
    >>> f = pyunlocbox.functions.norm_l2(y=[4, 5, 6, 7])
    >>> ret = pyunlocbox.solvers.solve([f], [0, 0, 0, 0], atol=1e-5)
    INFO: Dummy objective function added.
    INFO: Selected solver : forward_backward
    Solution found after 10 iterations :
        objective function f(sol) = 7.460428e-09
        last relative objective improvement : 1.624424e+03
        stopping criterion : ABS_TOL
    >>> ret['sol']
    array([ 3.99996922,  4.99996153,  5.99995383,  6.99994614])

    """

    if rtol < 0 or maxit < 0:
        raise ValueError('Parameters should be positive numbers.')
    if verbosity not in ['none', 'low', 'high']:
        raise ValueError('Verbosity should be either none, low or high.')

    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError('At least 1 convex function should be provided.')
    elif len(functions) == 1:
        functions.append(dummy())
        if verbosity in ['low', 'high']:
            print('INFO: Dummy objective function added.')

    # Choose a solver if none provided.
    if not solver:
        fb0 = 'GRAD' in functions[0].cap(x0) and 'PROX' in functions[1].cap(x0)
        fb1 = 'GRAD' in functions[1].cap(x0) and 'PROX' in functions[0].cap(x0)
        dg0 = 'PROX' in functions[0].cap(x0) and 'PROX' in functions[1].cap(x0)
        if len(functions) == 2:
            if fb0 or fb1:
                solver = forward_backward()  # Need one prox and 1 grad.
            elif dg0:
                solver = douglas_rachford()  # Need two prox.
            else:
                raise ValueError('No suitable solver for the given functions.')
        elif len(functions) > 2:
            raise NotImplementedError('No solver able to minimize more than 2 '
                                      'functions for now.')
        if verbosity in ['low', 'high']:
            print('INFO: Selected solver : %s' % (solver.__class__.__name__,))

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]

    # Solver specific initialization.
    solver.pre(functions, x0, verbosity)

    while not crit:

        niter += 1

        if verbosity is 'high':
            print('Iteration %d of %s :' % (niter, solver.__class__.__name__))

        # Solver iterative algorithm.
        solver.algo(objective, niter)

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Prevent division by 0.
        div = current
        if div == 0:
            if verbosity in ['low', 'high']:
                print('WARNING: objective function is equal to 0 !')
            if last != 0:
                div = last
            else:
                div = 1.0  # Result will be zero anyway.

        relative = np.abs((last - current) / div)

        # Verify stopping criteria.
        if current < atol:
            crit = 'ABS_TOL'
        elif relative < rtol:
            crit = 'REL_TOL'
        elif niter >= maxit:
            crit = 'MAX_IT'
        elif last - current < convergence_speed:
            crit = 'CONV_SPEED'

        if verbosity is 'high':
            print('    objective = %.2e, relative = %.2e'
                  % (current, relative))

    # Solver specific post-processing.
    solver.post(verbosity)

    if verbosity in ['low', 'high']:
        print('Solution found after %d iterations :' % (niter,))
        print('    objective function f(sol) = %e' % (current,))
        print('    last relative objective improvement : %e' % (relative,))
        print('    stopping criterion : %s' % (crit,))

    # Returned dictionary.
    result = {'sol':       solver.sol,
              'solver':    solver.__class__.__name__,  # algo for consistency ?
              'niter':     niter,
              'time':      time.time() - tstart,
              'eval':      current,
              'objective': objective,
              'crit':      crit,
              'rel':       relative}

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
    gamma : float
        The step size. This parameter is upper bounded by
        :math:`\frac{1}{\beta}` where the second convex function (gradient ?)
        is :math:`\beta` Lipschitz continuous. Default is 1.
    post_gamma : function
        User defined function to post-process the step size. This function is
        called every iteration and permits the user to alter the solver
        algorithm. The user may start with a high step size and progressively
        lower it while the algorithm runs to accelerate the convergence. The
        function parameters are the following : `gamma` (current step size),
        `sol` (current problem solution), `objective` (list of successive
        evaluations of the objective function), `niter` (current iteration
        number). The function should return a new value for `gamma`. Default is
        to return an unchanged value.
    post_sol : function
        User defined function to post-process the problem solution. This
        function is called every iteration and permits the user to alter the
        solver algorithm. Same parameter as :func:`post_gamma`. Default is to
        return an unchanged value.
    """

    def __init__(self, gamma=1, post_gamma=None, post_sol=None):
        if gamma < 0:
            raise ValueError('Gamma should be a positive number.')
        self.gamma = gamma
        if post_gamma:
            self.post_gamma = post_gamma
        else:
            self.post_gamma = lambda gamma, sol, objective, niter: gamma
        if post_sol:
            self.post_sol = post_sol
        else:
            self.post_sol = lambda gamma, sol, objective, niter: sol

    def pre(self, functions, x0, verbosity):
        """
        Solver specific initialization. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._pre(functions, x0, verbosity)

    def _pre(self, x0, verbosity):
        raise NotImplementedError("Class user should define this method.")

    def algo(self, objective, niter):
        """
        Call the solver iterative algorithm while allowing the user to alter
        it. This makes it possible to dynamically change the `gamma` step size
        while the algorithm is running.  See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._algo()
        self.gamma = self.post_gamma(self.gamma, self.sol, objective, niter)
        self.sol = self.post_sol(self.gamma, self.sol, objective, niter)

    def _algo(self):
        raise NotImplementedError("Class user should define this method.")

    def post(self, verbosity):
        """
        Solver specific post-processing. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._post(verbosity)

    def _post(self, verbosity):
        # Do not need to be necessarily implemented by class user.
        pass


class forward_backward(solver):
    r"""
    Forward-backward proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    two objective functions.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    method : {'FISTA', 'ISTA'}, optional
        the method used to solve the problem. It can be 'FISTA' or 'ISTA'.
        Default is 'FISTA'.
    lambda_ : float, optional
        the update term weight for ISTA. It should be between 0 and 1. Default
        is 1.

    Notes
    -----
    This algorithm requires one function to implement the
    :meth:`pyunlocbox.functions.func.prox` method and the other one to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.forward_backward(method='FISTA', lambda_=1, gamma=1)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 10 iterations :
        objective function f(sol) = 7.460428e-09
        last relative objective improvement : 1.624424e+03
        stopping criterion : ABS_TOL
    >>> ret['sol']
    array([ 3.99996922,  4.99996153,  5.99995383,  6.99994614])

    """

    def __init__(self, method='FISTA', lambda_=1, *args, **kwargs):
        super(forward_backward, self).__init__(*args, **kwargs)
        self.method = method
        self.lambda_ = lambda_

    def _pre(self, functions, x0, verbosity):

        if verbosity is 'high':
            print('INFO: Forward-backward method : %s' % (self.method,))

        if self.lambda_ < 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        # ISTA and FISTA initialization.
        self.sol = np.array(x0)

        if self.method is 'ISTA':
            self._algo = self._ista
        elif self.method is 'FISTA':
            self._algo = self._fista
            self.un = np.array(x0)
            self.tn = 1.
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
        yn = self.sol - self.gamma * self.f2.grad(self.sol)
        self.sol += self.lambda_ * (self.f1.prox(yn, self.gamma) - self.sol)

    def _fista(self):
        xn = self.un - self.gamma * self.f2.grad(self.un)
        xn = self.f1.prox(xn, self.gamma)
        tn1 = (1. + np.sqrt(1.+4.*self.tn**2.)) / 2.
        self.un = xn + (self.tn-1) / tn1 * (xn-self.sol)
        self.tn = tn1
        self.sol = xn


class douglas_rachford(solver):
    r"""
    Douglas-Rachford proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    two objective functions.

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

    Examples
    --------
    >>> from pyunlocbox import functions, solvers
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> solver = solvers.douglas_rachford(lambda_=1, gamma=1)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 8 iterations :
        objective function f(sol) = 2.927052e-06
        last relative objective improvement : 8.000000e+00
        stopping criterion : ABS_TOL
    >>> ret['sol']
    array([ 3.99939034,  4.99923792,  5.99908551,  6.99893309])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(douglas_rachford, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0, verbosity):

        if self.lambda_ < 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        if len(functions) != 2:
            raise ValueError('Douglas-Rachford requires two convex functions.')

        self.f1 = functions[0]
        self.f2 = functions[1]

        self.yn = np.array(x0)
        self.sol = np.array(x0)

    def _algo(self):
        tmp = self.f1.prox(2 * self.sol - self.yn, self.gamma)
        self.yn += self.lambda_ * (tmp - self.sol)
        self.sol = self.f2.prox(self.yn, self.gamma)
