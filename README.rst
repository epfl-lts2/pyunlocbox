=========================================
PyUNLocBoX: convex optimization in Python
=========================================

.. image:: https://readthedocs.org/projects/pyunlocbox/badge/?version=latest
   :target: https://pyunlocbox.readthedocs.io/en/latest/

.. image:: https://img.shields.io/travis/epfl-lts2/pyunlocbox.svg
   :target: https://travis-ci.org/epfl-lts2/pyunlocbox

.. image:: https://img.shields.io/coveralls/epfl-lts2/pyunlocbox.svg
   :target: https://coveralls.io/github/epfl-lts2/pyunlocbox

.. image:: https://img.shields.io/pypi/v/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox

.. image:: https://img.shields.io/pypi/l/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox

.. image:: https://img.shields.io/pypi/pyversions/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox

.. image:: https://img.shields.io/github/stars/epfl-lts2/pyunlocbox.svg?style=social
   :target: https://github.com/epfl-lts2/pyunlocbox

The PyUNLocBoX is a convex optimization package based on `proximal splitting
methods <https://en.wikipedia.org/wiki/Proximal_gradient_method>`_ and
implemented in Python (a `Matlab counterpart <https://lts2.epfl.ch/unlocbox>`_
exists). It is a free software, distributed under the BSD license, and
available on `PyPI <https://pypi.python.org/pypi/pyunlocbox>`_. The
documentation is available `online <https://pyunlocbox.readthedocs.io>`_ and
development takes place on `GitHub <https://github.com/epfl-lts2/pyunlocbox>`_.

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

>>> import pyunlocbox
>>> f1 = pyunlocbox.functions.norm_l2(y=[4, 5, 6, 7])
>>> f2 = pyunlocbox.functions.dummy()
>>> solver = pyunlocbox.solvers.forward_backward()
>>> ret = pyunlocbox.solvers.solve([f1, f2], [0., 0, 0, 0], solver, atol=1e-5)
Solution found after 9 iterations:
    objective function f(sol) = 6.714385e-08
    stopping criterion: ATOL
>>> ret['sol']
array([ 3.99990766,  4.99988458,  5.99986149,  6.99983841])

Installation
------------

The PyUnLocBox is available on PyPI::

    $ pip install pyunlocbox

Contributing
------------

The development of this package takes place on `GitHub
<https://github.com/epfl-lts2/pyunlocbox>`_. Issues and pull requests are
welcome.

You can improve or add solvers, functions, and acceleration schemes in
``pyunlocbox/solvers.py``, ``pyunlocbox/functions.py``, and
``pyunlocbox/acceleration.py``, along with their corresponding unit tests in
``pyunlocbox/tests/test_*.py`` (with reasonable coverage) and documentation in
``doc/reference/*.rst``. If you have a nice example to demonstrate the use of
the introduced functionality, please consider adding a tutorial in
``doc/tutorials``.

Do not forget to update ``README.rst`` and ``doc/history.rst`` with e.g. new
features or contributors. The version number needs to be updated in
``setup.py`` and ``pyunlocbox/__init__.py``.

Please make sure that your changes pass the tests (enforced by CI) and check
the generated coverage report at ``htmlcov/index.html`` to make sure the tests
reasonably cover the changes you've introduced::

$ make lint
$ make test
$ make docall

Authors
-------

PyUNLocBoX was started in 2014 as an academic project for research purpose at
the `EPFL LTS2 laboratory <https://lts2.epfl.ch>`_.

Development lead :

* Rodrigo Pena from EPFL LTS2 <rodrigo.pena@epfl.ch>
* Michaël Defferrard from EPFL LTS2 <michael.defferrard@epfl.ch>

Contributors :

* Alexandre Lafaye from EPFL LTS2 <alexandre.lafaye@epfl.ch>
* Basile Châtillon from EPFL LTS2 <basile.chatillon@epfl.ch>
* Nicolas Rod from EPFL LTS2 <nicolas.rod@epfl.ch>
* Nathanaël Perraudin from EPFL LTS2 <nathanael.perraudin@epfl.ch>
