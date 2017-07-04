# -*- coding: utf-8 -*-

"""
The toolbox is organized around two class hierarchies: the functions and the
solvers. Instantiated functions represent convex functions to optimize.
Instantiated solvers represent solving algorithms. The
:func:`pyunlocbox.solvers.solve` solving function takes as parameters a solver
object and some function objects to actually solve the optimization problem.
See this function's documentation for a typical usage example.

The :mod:`pyunlocbox` package is divided into the following modules:

* :mod:`pyunlocbox.solvers`: problem solvers, implement the solvers class
  hierarchy and the solving function
* :mod:`pyunlocbox.functions`: functions to be passed to the solvers, implement
  the functions class hierarchy
* :mod:`pyunlocbox.operators`: useful operators to be passed to the functions
* :mod:`pyunlocbox.acceleration`: acceleration schemes to be passed to the
  solvers, implement the acceleration class hierarchy

"""

# When importing the toolbox, you surely want these modules.
from pyunlocbox import functions
from pyunlocbox import solvers
from pyunlocbox import operators
from pyunlocbox import acceleration

# Silence the code checker warning about unused symbols.
assert functions
assert solvers
assert operators
assert acceleration

__version__ = '0.5.1'
__release_date__ = '2017-07-04'
