.. :changelog:

=======
History
=======

x.x.x (xxxx-xx-xx)
------------------

New Features:

* Monotone+Lipschitz Forward-Backward-Forward primal-dual algorithm
* Projection-based primal-dual algorithm
* L2-norm proximal operator supports non-tight frames

Bug fixes :

* prox_tv_2d has been fixed

Infrastructure :

* Continuous integration testing on Python 2.7, 3.3, 3.4 and 3.5
* Travis-ci: check style and build doc
* Removed tox config (too cumbersome to use on dev box)
* Monitor code coverage and report to coveralls.io


0.2.2 (2015-01-16)
------------------

New feature version. Still experimental.

New Features:

* norm_tv has been added with gradient, div, evaluation and prox.
* Module signals has been added.
* A demo for douglas rachford is also now present.


0.2.1 (2014-08-20)
------------------

Bug fix version. Still experimental.

Bug fixes :

* Avoid complex casting to real
* Do not stop iterating if the objective function stays at zero

0.2.0 (2014-08-04)
------------------

Second usable version, available on GitHub and released on PyPI.
Still experimental.

New features :

* Douglas-Rachford splitting algorithm
* Projection on the L2-ball for tight and non tight frames
* Compressed sensing tutorial using L2-ball, L2-norm and Douglas-Rachford
* Automatic solver selection

Infrastructure :

* Unit tests for all functions and solvers
* Continuous integration testing on Python 2.6, 2.7, 3.2, 3.3 and 3.4

0.1.0 (2014-06-08)
------------------

First usable version, available on GitHub and released on PyPI.
Still experimental.

Features :

* Forward-backward splitting algorithm
* L1-norm function (eval and prox)
* L2-norm function (eval, grad and prox)
* TV-norm function (eval, grad, div and prox)
* Least square problem tutorial using L2-norm and forward-backward
* Compressed sensing tutorial using L1-norm, L2-norm and forward-backward

Infrastructure :

* Sphinx generated documentation using Numpy style docstrings
* Documentation hosted on Read the Docs
* Code hosted on GitHub
* Package hosted on PyPI
* Code checked by flake8
* Docstring and tutorial examples checked by doctest (as a test suite)
* Unit tests for functions module (as a test suite)
* All test suites executed in Python 2.6, 2.7 and 3.2 virtualenvs by tox
* Distributed automatic testing on Travis CI continuous integration platform
