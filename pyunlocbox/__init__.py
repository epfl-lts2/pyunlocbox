# -*- coding: utf-8 -*-

r"""
The package is mainly organized around two class hierarchies: the functions and
the solvers. Instantiated functions represent convex functions to optimize.
Instantiated solvers represent solving algorithms. The
:func:`pyunlocbox.solvers.solve` solving function takes as parameters a solver
object and some function objects to actually solve the optimization problem.
See this function's documentation for a typical usage example.

The :mod:`pyunlocbox` package is divided into the following modules:

* :mod:`.functions`: objective functions to define an optimization problem,
* :mod:`.solvers`: the main solving function and common solvers,
* :mod:`.acceleration`: general acceleration schemes for various solvers,
* :mod:`.operators`: some operators.

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

__version__ = '0.5.2'
__release_date__ = '2017-12-15'
