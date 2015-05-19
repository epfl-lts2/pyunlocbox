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
        x^{t-1}\|_2}{n N} < xtol`. Default is None.
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
    INFO: Selected solver : forward_backward
        norm_l2 evaluation : 1.260000e+02
        dummy evaluation : 0.000000e+00
    INFO: Forward-backward method : FISTA
    Iteration 1 of forward_backward :
        norm_l2 evaluation : 1.400000e+01
        dummy evaluation : 0.000000e+00
        objective = 1.40e+01
    Iteration 2 of forward_backward :
        norm_l2 evaluation : 1.555556e+00
        dummy evaluation : 0.000000e+00
        objective = 1.56e+00
    Iteration 3 of forward_backward :
        norm_l2 evaluation : 3.293044e-02
        dummy evaluation : 0.000000e+00
        objective = 3.29e-02
    Iteration 4 of forward_backward :
        norm_l2 evaluation : 8.780588e-03
        dummy evaluation : 0.000000e+00
        objective = 8.78e-03
    Solution found after 4 iterations :
        objective function f(sol) = 8.780588e-03
        stopping criterion : ATOL

    Verify the stopping criterion (should be smaller than atol=1e-2):

    >>> np.linalg.norm(ret['sol'] - y)**2
    0.008780587752251795

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
    >>> ret['objective']  # doctest:+NORMALIZE_WHITESPACE
    [[126.0, 0], [13.999999999999998, 0], [1.5555555555555558, 0],
    [0.032930436204105726, 0], [0.0087805877522517933, 0]]

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
            fb0 = 'GRAD' in functions[0].cap(x0) and 'PROX' in functions[1].cap(x0)
            fb1 = 'GRAD' in functions[1].cap(x0) and 'PROX' in functions[0].cap(x0)
            dg0 = 'PROX' in functions[0].cap(x0) and 'PROX' in functions[1].cap(x0)
            if fb0 or fb1:
                solver = forward_backward()  # Need one prox and 1 grad.
            elif dg0:
                solver = douglas_rachford()  # Need two prox.
            else:
                raise ValueError('No suitable solver for the given functions.')
        elif len(functions) > 2:
            solver = generalized_forward_backward()
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            print('INFO: Selected solver : %s' % (solver.__class__.__name__,))

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

        if xtol != None:
            last_sol = solver.sol

        if verbosity in ['HIGH', 'ALL']:
            print('Iteration %d of %s :' % (niter, solver.__class__.__name__))

        # Solver iterative algorithm.
        solver.algo(objective, niter)

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol != None and current < atol:
            crit = 'ATOL'
        if dtol != None and np.abs(current - last) < dtol:
            crit = 'DTOL'
        if rtol != None:
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
        if xtol != None:
            err = np.linalg.norm(solver.sol - last_sol) / last_sol.size
            if err < xtol:
                crit = 'XTOL'
        if maxit != None and niter >= maxit:
            crit = 'MAXIT'

        if verbosity in ['HIGH', 'ALL']:
            print('    objective = %.2e' % current)

    # Solver specific post-processing.
    solver.post()

    # Restore verbosity for functions. In case they are called outside solve().
    for k, f in enumerate(functions):
        f.verbosity = functions_verbosity[k]

    if verbosity in ['LOW', 'HIGH', 'ALL']:
        print('Solution found after %d iterations :' % niter)
        print('    objective function f(sol) = %e' % current)
        print('    stopping criterion : %s' % crit)

    # Returned dictionary.
    result = {'sol':       solver.sol,
              'solver':    solver.__class__.__name__,  # algo for consistency ?
              'crit':      crit,
              'niter':     niter,
              'time':      time.time() - tstart,
              'objective': objective}


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
        The step size. This parameter is upper bounded by
        :math:`\frac{1}{\beta}` where the second convex function (gradient ?)
        is :math:`\beta` Lipschitz continuous. Default is 1.
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
        self._pre(functions, x0)

    def _pre(self, x0):
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
        Solver specific post-processing. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.
        """
        self._post()

    def _post(self):
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
        The method used to solve the problem. It can be 'FISTA' or 'ISTA'.
        Default is 'FISTA'.
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
    Solution found after 12 iterations :
        objective function f(sol) = 4.135992e-06
        stopping criterion : ATOL
    >>> ret['sol']
    array([ 3.99927529,  4.99909411,  5.99891293,  6.99873176])

    """

    def __init__(self, method='FISTA', lambda_=1, *args, **kwargs):
        super(forward_backward, self).__init__(*args, **kwargs)
        self.method = method
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.verbosity is 'HIGH':
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
        yn = self.sol - self.step * self.f2.grad(self.sol)
        self.sol += self.lambda_ * (self.f1.prox(yn, self.step) - self.sol)

    def _fista(self):
        xn = self.un - self.step * self.f2.grad(self.un)
        xn = self.f1.prox(xn, self.step)
        tn1 = (1. + np.sqrt(1.+4.*self.tn**2.)) / 2.
        self.un = xn + (self.tn-1) / tn1 * (xn-self.sol)
        self.tn = tn1
        self.sol = xn


class generalized_forward_backward(solver):
    r"""
    Forward-backward proximal splitting algorithm.

    This algorithm solves convex optimization problems composed of the sum of
    N objective functions.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    lambda_ : float, optional
        The update term weight for ISTA. It should be between 0 and 1. Default
        is 1.

    Notes
    -----
    This algorithm requires one function to implement the
    :meth:`pyunlocbox.functions.func.prox` method and the other one to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

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
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 2 iterations :
        objective function f(sol) = 1.463100e+01
        stopping criterion : RTOL
    >>> ret['sol']
    array([ 0. ,  0. ,  7.5,  0. ,  0. ,  0. ,  6.5])

    """

    def __init__(self, lambda_=1, weight=[], *args, **kwargs):
        super(generalized_forward_backward, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.weight = weight

    def _pre(self, functions, x0):

        if self.verbosity is 'HIGH':
            print('INFO: Generalized forward-backward\
                  method minimizing %i functions')

        if self.lambda_ < 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        # Initialization.
        self.sol = np.array(x0)

        self._algo = self._gista
        self.f1 = []
        self.f2 = []
        self.z = []
        for ii in range(0, len(functions)):
            if 'GRAD' in functions[ii].cap(x0):
                self.f2.append(functions[ii])
            elif 'PROX' in functions[ii].cap(x0):
                self.f1.append(functions[ii])
                self.z.append(x0)
            else:
                raise ValueError('SOLVER: There is a function without grad\
                                 and prox')

        if len(self.weight) == 0:
            if len(self.f1):
                self.weight = np.repeat(1./len(self.f1), len(self.f1))
        elif len(self.weight) != len(self.f1):
            raise ValueError('GENERALIZED FORWARD BACKWARD: The number of\
                             element in weight is wrong')

        #if len(self.f2) == 0:
        #    raise ValueError('GENERALIZED FORWARD BACKWARD: I need at least a function with at gradient!')

    def _gista(self):
        grad_eval = np.zeros(np.shape(self.sol))
        for ii in range(0, len(self.f2)):
            grad_eval = grad_eval + self.f2[ii].grad(self.sol)

        for ii in range(0, len(self.f1)):
            self.z[ii] += self.lambda_ *\
                (self.f1[ii].prox(2 * self.sol - self.z[ii] - self.step *
                                  grad_eval,
                                  self.step/self.weight[ii]) - self.sol)

        if len(self.f1):
            self.sol = np.zeros(np.shape(self.sol))
            for ii in range(0, len(self.f1)):
                self.sol += self.weight[ii] * self.z[ii]
        else:
            self.sol -= self.step * grad_eval


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
    Solution found after 8 iterations :
        objective function f(sol) = 2.927052e-06
        stopping criterion : ATOL
    >>> ret['sol']
    array([ 3.99939034,  4.99923792,  5.99908551,  6.99893309])

    """

    def __init__(self, lambda_=1, *args, **kwargs):
        super(douglas_rachford, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):

        if self.lambda_ < 0 or self.lambda_ > 1:
            raise ValueError('Lambda is bounded by 0 and 1.')

        if len(functions) != 2:
            raise ValueError('Douglas-Rachford requires two convex functions.')

        self.f1 = functions[0]
        self.f2 = functions[1]

        self.yn = np.array(x0)
        self.sol = np.array(x0)

    def _algo(self):
        tmp = self.f1.prox(2 * self.sol - self.yn, self.step)
        self.yn = self.yn + self.lambda_ * (tmp - self.sol)
        self.sol = self.f2.prox(self.yn, self.step)
