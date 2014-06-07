# -*- coding: utf-8 -*-

"""
PyUNLocBoX is a convex optimization toolbox using proximal splitting methods.
It is a port of the Matlab UNLocBoX toolbox.

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

Following is a typical usage example who solves an optimization problem
composed by the sum of two convex functions. The functions and solver objects
are first instantiated with the desired parameters. The problem is then solved
by a call to the solving function.

>>> import pyunlocbox
>>> y = [4, 5, 6, 7]
>>> f1 = pyunlocbox.functions.norm_l2(y=y)
>>> f1.eval([0, 0, 0, 0])
126
>>> f1.grad([0, 0, 0, 0])
array([ -8, -10, -12, -14])
>>> f2 = pyunlocbox.functions.func()
>>> f2.eval = lambda x: 0
>>> f2.grad = lambda x: 0
>>> solver = pyunlocbox.solvers.forward_backward()
>>> ret = pyunlocbox.solvers.solve([f1, f2], [0, 0, 0, 0], solver, absTol=1e-5)
Solution found in 10 iterations :
    objective function f(sol) = 7.460428e-09
    last relative objective improvement : 1.624424e+03
    stopping criterion : ABS_TOL
>>> ret['sol']
array([ 3.99996922,  4.99996153,  5.99995383,  6.99994614])

"""

# When importing the toolbox, you surely want these modules.
from pyunlocbox import functions
from pyunlocbox import solvers

# Silence the code checker warning about unused symbols.
assert functions
assert solvers

__name__ = 'PyUNLocBoX'
__version__ = '1.0'
__release_date__ = '2014-06-07'

__docformat__ = "restructuredtext en"

__author__ = 'EPFL LTS2 Michaël Defferrard and Nathanaël Perraudin'
__copyright__ = '2014, EPFL LTS2'
__email__ = 'michael.defferrard@epfl.ch, nathanael.perraudin@epfl.ch'
