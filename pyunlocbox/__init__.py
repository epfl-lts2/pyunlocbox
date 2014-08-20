# -*- coding: utf-8 -*-

"""
PyUNLocBoX is a convex optimization toolbox using proximal splitting methods.
It is a port of the Matlab UNLocBoX toolbox.

The toolbox is organized around two classes hierarchies : the functions and the
solvers. Instantiated functions represent convex functions to optimize.
Instantiated solvers represent solving algorithms. The
:func:`pyunlocbox.solvers.solve` solving function takes as parameters a
solver object and some function objects to actually solve the optimization
problem.

The :mod:`pyunlocbox` package is divided into the following modules :

* :mod:`pyunlocbox.solvers`: problem solvers, implement the solvers class
  hierarchy and the solving function
* :mod:`pyunlocbox.functions`: functions to be passed to the solvers, implement
  the functions class hierarchy

Following is a typical usage example who solves an optimization problem
composed by the sum of two convex functions. The functions and solver objects
are first instantiated with the desired parameters. The problem is then solved
by a call to the solving function.

>>> import pyunlocbox
>>> f1 = pyunlocbox.functions.norm_l2(y=[4, 5, 6, 7])
>>> f2 = pyunlocbox.functions.dummy()
>>> solver = pyunlocbox.solvers.forward_backward()
>>> ret = pyunlocbox.solvers.solve([f1, f2], [0, 0, 0, 0], solver, atol=1e-5)
Solution found after 10 iterations :
    objective function f(sol) = 7.460428e-09
    last relative objective improvement : 1.624424e+03
    stopping criterion : ATOL
>>> ret['sol']
array([ 3.99996922,  4.99996153,  5.99995383,  6.99994614])

"""

# When importing the toolbox, you surely want these modules.
from pyunlocbox import functions
from pyunlocbox import solvers

# Silence the code checker warning about unused symbols.
assert functions
assert solvers

__version__ = '0.2.1'
__release_date__ = '2014-08-20'
