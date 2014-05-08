#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements solver objects which minimize function objects output.
It includes the following solvers :

* :func:`forward_backward`: Forward-backward splitting algorithm
"""

import numpy as np


def forward_backward(x0, f1, f2, gamma=1, reltol=10**-3, abstol=0,
                     method='FISTA', lamb=1, maxit=200, verbosity='low'):
    """
    Forward-backward splitting algorithm.

    **Usage**

    ``sol = forward_backward(x0, f1, f2)``
    TODO: using doctest

    **Arguments**

    * *x0* : starting point of the algorithm
    * *f1* : first convex function to minimize. It is an object who must
      implement the :func:`eval` and :func:`prox` methods.
    * *f2* : second convex function to minimize, with a :math:`\\beta`
      Lipschitz continuous gradient. It is an object who must implement the
      :func:`grad` and :func:`prox` methods.
    * *gamma* : the step size. This parameter is upper bounded by
      :math:`\\frac{1}{\\beta}` where :math:`f_2` is :math:`\\beta` Lipschitz
      continuous. Default is 1.
    * *reltol* : the relative tolerance stopping criterion. The algorithm stops
      when :math:`\\frac{n(t)-n(t-1)}{n(t)}<reltol` where
      :math:`n(t)=f_1(x)+f_2(x)` is the objective function at iteration
      :math:`t`. Default is :math:`10^{-3}`.
    * *abstol* : the absolute tolerance stopping criterion. The algorithm stops
      when :math:`n(t)<abstol`. Default is 0.
    * *method* : the method used to solve the problem. It can be 'FISTA' or
      'ISTA'. Default is 'FISTA'.
    * *lamb* : the update term weight for ISTA. It should be between 0 and 1.
      Default is 1.
    * *maxit* : maximum number of iterations. Default is 200.
    * *verbosity* : 'none' for no log, 'low' to print main steps, 'high' to
      print all steps.

    **Returns**

    * *sol* : solution

    **Description**

    ``forward_backward`` solves :math:`arg \min_x f_1(x) + f_2(x)`
    for :math:`x \in \mathbb{R}^N` where :math:`x` is the variable
    """
    if method != 'FISTA' or method != 'ISTA':
        raise ValueError('method should be FISTA or ISTA')
    if lamb < 0 or lamb > 1:
        raise ValueError('lamb is bounded by 0 and 1')
    if verbosity != 'none' or verbosity != 'low' or verbosity != 'high':
        raise ValueError('verbosity should be either none, low or high')

    curNorm = f1.eval(x0) + f2.eval(x0)
# convergence test
    stop = False

    if verbosity == 'high':
        print('Selected algorithm' + str(method))

    # ISTA initialization.
    xn = x0

    # FISTA initialization.
    un = x0
    sol = x0
    tn = 1.

    # Iterative ISTA or FISTA algorithm.
    while !stop:

        if verbosity == 'high':
            print('Iteration ' + str(iter))

        if method == 'ISTA':
            yn = xn - gamma * f2.grad(xn)
            sol = xn + lamb * (f1.prox(yn, gamma) - xn)
            xn = sol
        elif method == 'FISTA':
            xn = f1.prox(un - gamma * f2.grad(un), gamma)
            tn1 = (1. + np.sqrt(1.+4.*tn^2.)) / 2.
            un = xn + (tn-1) / tn1 * (xn-sol)
            sol = xn
            tn = tn1
        else:
            raise ValueError('method should be FISTA or ISTA')

        curNorm = f1.eval(sol) + f2.eval(sol)
# convergence test
# post process
        if verbosity == 'high':
            print('||f1 + f2|| = ' + str(curNorm) +
                  ', relative norm : ' + str(relNorm))

    # Final logging.
    if verbosity == 'high' or verbosity == 'low':
        print('Solution found (in ' + str(iter) + ' iterations) : ||f|| = ' +
              str(curNorm) + ', relative norm : ' + str(relNorm) +
              ', stopping criterion : ' + str(crit))

# info struct
