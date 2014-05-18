# -*- coding: utf-8 -*-

"""
PyUNLockBox is a convex optimization toolbox using proximal splitting methods.
It is a port of the Matlab UNLocBox toolbox

The toolbox is organized around two classes hierarchy : the functions and the
solvers. Instanced functions represent convex functions to optimize. Instanced
solvers represent solving algorithms. The :func:`pyunlocbox.solvers.solve`
solving function takes as parameters a solver object and some function objects
to actually solve the optimization problem.

The :mod:`pyunlockbox` package is divided into the following modules :

* :mod:`pyunlocbox.solvers`: problem solvers, implement the solvers class
  hierarchy and the solving function
* :mod:`pyunlocbox.functions`: functions to be passed to the solvers, implement
  the functions class hierarchy
* :mod:`pyunlocbox.demos`: some problem solving demonstrations using the
  toolbox

Following is a typical usage example who solves an optimization problem
composed by the sum of two convex functions. The functions and solver objects
are first instantiated with the desired parameters. The problem is then solved
by a call to the solving algorithm.

>>> import pyunlocbox
>>> f1 = pyunlocbox.functions.norm_l1(lamb=1)
>>> y = [4, 5, 6, 7]
>>> f2 = pyunlocbox.functions.norm_l2(lamb=1, y=y)
>>> solver = pyunlocbox.solvers.forward_backward()
>>> x0 = [0, 0, 0, 0]
>>> sol, info, objective = pyunlocbox.solvers.solve(solver, f1, f2, x0)
"""

__author__ = 'EPFL LTS2'
__email__ = 'nathanael.perraudin@epfl.ch'
__version__ = '0.1.0'
