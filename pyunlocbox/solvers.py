#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements solvers which minimize functions output. Call
:func:`solve` to solve your convex optimization problem using the solver
and functions of your choice. The :class:`solver` class is a base class for all
solvers. The following solvers are included :

* :class:`forward_backward`: Forward-backward splitting algorithm
"""

import numpy as np


def solve(solver, f1, f2, x0):
    """
    Parameters :

    * *solver* : solver. It is an object who must implement the :func:`pre`,
      :func:`algo` and :func:`post` methods.
    * *f1* : first convex function to minimize. It is an object who must
      implement the :func:`eval` method. The :func:`grad` and / or :func:`prox`
      methods are required by some solvers. Please refer to the documentation
      of the solver.
    * *f2* : second convex function to minimize, with a :math:`\\beta`
      Lipschitz continuous gradient. It is an object who must implement the
      :func:`eval` method. The :func:`grad` and / or :func:`prox` methods are
      required by some solvers. Please refer to the documentation of
      the solver.
    * *x0* : starting point of the algorithm

    **Returns**

    * *sol* : solution
    * *info* : informations about the algorithm execution
    """

    # Common initialization.
    curNorm = f1.eval(x0) + f2.eval(x0)
# convergence test
    stop = False
    niter = 0
    relNorm = 0
    crit = 'MAXIT'

    # Solver specific initialization.
    solver.pre(x0)

    while not stop:

        if solver.verbosity == 'high':
            print('Iteration ' + str(niter))

        # Solver iterative algorithm.
        solver.algo(f1, f2)

        # Verify stopping criterion.
        curNorm = f1.eval(solver.sol) + f2.eval(solver.sol)

# convergence test
# post process
        if solver.verbosity == 'high':
            print('||f1 + f2|| = ' + str(curNorm) +
                  ', relative norm : ' + str(relNorm))

    # Solver specific post-processing.
    solver.post()

    # Final logging.
    if solver.verbosity in ['high', 'low']:
        print('Solution found (in ' + str(niter) + ' iterations) : ||f|| = ' +
              str(curNorm) + ', relative norm : ' + str(relNorm) +
              ', stopping criterion : ' + str(crit))

    return solver.sol


class solver:
    """
    This class defines a solver object to be passed to :func:`solve`. It is
    intended to be a base class for standard solvers which will implement
    the required methods. It can also be instantiated by user code and
    dynamically modified for rapid testing.

    This class also defines the generic attributes of the solver objects.

    **Attributes :**

    * *gamma* : the step size. This parameter is upper bounded by
      :math:`\\frac{1}{\\beta}` where :math:`f_2` is :math:`\\beta` Lipschitz
      continuous. Default is 1.
    * *reltol* : the relative tolerance stopping criterion. The algorithm stops
      when :math:`\\frac{n(t)-n(t-1)}{n(t)}<reltol` where
      :math:`n(t)=f_1(x)+f_2(x)` is the objective function at iteration
      :math:`t`. Default is :math:`10^{-3}`.
    * *abstol* : the absolute tolerance stopping criterion. The algorithm stops
      when :math:`n(t)<abstol`. Default is 0.
    * *maxit* : maximum number of iterations. Default is 200.
    * *verbosity* : 'none' for no log, 'low' to print main steps, 'high' to
      print all steps.
    """

    def __init__(self, gamma=1, reltol=10**-3, abstol=0, maxit=200,
                 verbosity='low'):
        self.gamma = gamma
        self.reltol = reltol
        self.abstol = abstol
        self.maxit = maxit
        if verbosity not in ['none', 'low', 'high']:
            raise ValueError('verbosity should be either none, low or high')
        self.verbosity = verbosity

    def pre(self, x0):
        """
        Solver specific initialization.
        """
        raise NotImplementedError("Class user should define this method.")

    def algo(self, f1, f2):
        """
        Solver iterative algorithm.
        """
        raise NotImplementedError("Class user should define this method.")

    def post(self):
        """
        Solver specific post-processing.
        """
        raise NotImplementedError("Class user should define this method.")


class forward_backward(solver):
    """
    Forward-backward splitting algorithm.

    This algorithm requires ``f1`` to implement the :func:`prox` method and
    ``f2`` to implement the :func:`grad` method.

    **Usage**

    ``solver1 = solvers.forward_backward(...)
    sol = solvers.solve(solver1, f1, f2, x0)``
    TODO: using doctest

    **Attributes**

    * *method* : the method used to solve the problem. It can be 'FISTA' or
      'ISTA'. Default is 'FISTA'.
    * *lamb* : the update term weight for ISTA. It should be between 0 and 1.
      Default is 1.

    See generic attributes descriptions of the base class :class:`solver`.

    **Description**

    ``forward_backward`` solves :math:`arg \min_x f_1(x) + f_2(x)`
    for :math:`x \in \mathbb{R}^N` where :math:`x` is the variable.
    """

    def __init__(self, method='FISTA', lamb=1, *args, **kwargs):
        solver.__init__(self, *args, **kwargs)

        if method != 'FISTA' or method != 'ISTA':
            raise ValueError('method should be FISTA or ISTA')
        self.method = method
        if lamb < 0 or lamb > 1:
            raise ValueError('lamb is bounded by 0 and 1')
        self.lamb = lamb

    def pre(self, x0):
        """
        Algorithm initialization.
        """
        if self.verbosity == 'high':
            print('Selected algorithm : ' + str(self.method))

        # ISTA and FISTA initialization.
        self.sol = np.array(x0)

        # FISTA initialization.
        self.un = np.array(x0)
        self.tn = 1.

    def algo(self, f1, f2):
        """
        Iterative ISTA or FISTA algorithm.
        """
        if self.method == 'ISTA':
            yn = self.sol - self.gamma * f2.grad(self.sol)
#FIXME gamma is not the same as in matlab
            self.sol += self.lamb * (f1.prox(yn, self.gamma) - self.sol)
        elif self.method == 'FISTA':
            xn = f1.prox(self.un - self.gamma * f2.grad(self.un), self.gamma)
            tn1 = (1. + np.sqrt(1.+4.*self.tn**2.)) / 2.
            self.un = xn + (self.tn-1) / tn1 * (xn-self.sol)
            self.sol = xn
            self.tn = tn1
        else:
            raise ValueError('method should be FISTA or ISTA')

    def post(self):
        """
        No post-processing.
        """
