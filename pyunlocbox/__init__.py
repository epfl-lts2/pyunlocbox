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

from pyunlocbox import acceleration  # noqa: F401
from pyunlocbox import functions  # noqa: F401
from pyunlocbox import operators  # noqa: F401
from pyunlocbox import solvers  # noqa: F401

__version__ = "0.5.2"
__release_date__ = "2017-12-15"


def test():  # pragma: no cover
    """Run the test suite."""
    import unittest

    # Lazy as it might be slow and require additional dependencies.
    from pyunlocbox.tests import suite

    unittest.TextTestRunner(verbosity=2).run(suite)
