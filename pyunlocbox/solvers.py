#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements solver objects which minimize function objects output.
It includes the following solvers :

* :func:`forward_backward`: Forward-backward splitting algorithm
"""


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
