==============================================
PyUNLocBoX: Optimization by Proximal Splitting
==============================================

+-------------------------------------------+
| |doc|  |pypi|  |license|  |pyversions|    |
+-------------------------------------------+
| |binder|  |travis|  |coveralls|  |github| |
+-------------------------------------------+

.. |doc| image:: https://readthedocs.org/projects/pyunlocbox/badge/?version=latest
   :target: https://pyunlocbox.readthedocs.io
.. |pypi| image:: https://img.shields.io/pypi/v/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox
.. |license| image:: https://img.shields.io/pypi/l/pyunlocbox.svg
   :target: https://github.com/epfl-lts2/pyunlocbox/blob/master/LICENSE.txt
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox
.. |travis| image:: https://img.shields.io/travis/epfl-lts2/pyunlocbox.svg
   :target: https://travis-ci.org/epfl-lts2/pyunlocbox
.. |coveralls| image:: https://img.shields.io/coveralls/epfl-lts2/pyunlocbox.svg
   :target: https://coveralls.io/github/epfl-lts2/pyunlocbox
.. |github| image:: https://img.shields.io/github/stars/epfl-lts2/pyunlocbox.svg?style=social
   :target: https://github.com/epfl-lts2/pyunlocbox
.. |binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/epfl-lts2/pyunlocbox/master?filepath=playground.ipynb

The PyUNLocBoX is a Python package which uses
`proximal splitting methods <https://en.wikipedia.org/wiki/Proximal_gradient_method>`_
to solve non-differentiable convex optimization problems.
It is a free software, distributed under the BSD license, and
available on `PyPI <https://pypi.python.org/pypi/pyunlocbox>`_.
The documentation is available on
`Read the Docs <https://pyunlocbox.readthedocs.io>`_
and development takes place on
`GitHub <https://github.com/epfl-lts2/pyunlocbox>`_.
(A `Matlab counterpart <https://lts2.epfl.ch/unlocbox>`_ exists.)

The package is designed to be easy to use while allowing any advanced tasks. It
is not meant to be a black-box optimization tool. You'll have to carefully
design your solver. In exchange you'll get full control of what the package
does for you, without the pain of rewriting the proximity operators and the
solvers and with the added benefit of tested algorithms. With this package, you
can focus on your problem and the best way to solve it rather that the details
of the algorithms. It comes with the following solvers:

* Gradient descent
* Forward-backward splitting algorithm (FISTA, ISTA)
* Douglas-Rachford splitting algorithm
* Generalized forward-backward
* Monotone+Lipschitz forward-backward-forward primal-dual algorithm
* Projection-based primal-dual algorithm

Moreover, the following acceleration schemes are included:

* FISTA acceleration scheme
* Backtracking based on a quadratic approximation of the objective
* Regularized nonlinear acceleration (RNA)

To compose your objective, you can either define your custom functions (which
should implement an evaluation method and a gradient or proximity method) or
use one of the followings:

* L1-norm
* L2-norm
* TV-norm
* Nuclear-norm
* Projection on the L2-ball

Following is a typical usage example who solves an optimization problem
composed by the sum of two convex functions. The functions and solver objects
are first instantiated with the desired parameters. The problem is then solved
by a call to the solving function.

>>> from pyunlocbox import functions, solvers
>>> f1 = functions.norm_l2(y=[4, 5, 6, 7])
>>> f2 = functions.dummy()
>>> solver = solvers.forward_backward()
>>> ret = solvers.solve([f1, f2], [0., 0, 0, 0], solver, atol=1e-5)
Solution found after 9 iterations:
    objective function f(sol) = 6.714385e-08
    stopping criterion: ATOL
>>> ret['sol']
array([ 3.99990766,  4.99988458,  5.99986149,  6.99983841])

You can
`try it online <https://mybinder.org/v2/gh/epfl-lts2/pyunlocbox/master?filepath=playground.ipynb>`_,
look at the
`tutorials <https://pyunlocbox.readthedocs.io/en/stable/tutorials/index.html>`_
to learn how to use it, or look at the
`reference guide <https://pyunlocbox.readthedocs.io/en/stable/reference/index.html>`_
for an exhaustive documentation of the API. Enjoy the package!

Installation
------------

The PyUNLocBoX is available on PyPI::

    $ pip install pyunlocbox

Contributing
------------

See the guidelines for contributing in ``CONTRIBUTING.rst``.

Acknowledgments
---------------

The PyUNLocBoX was started in 2014 as an academic open-source project for
research purpose at the `EPFL LTS2 laboratory <https://lts2.epfl.ch>`_.
