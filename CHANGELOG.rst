=========
Changelog
=========

Versions follow `SemVer <https://semver.org>`_ (we try our best).

Unreleased
----------

* New function: proj_positive.
* New function: structured_sparsity.
* Continuous integration with Python 3.6, 3.7, 3.8. Dropped 2.7, 3.4, 3.5.
* Merged all the extra requirements in a single dev requirement.

0.5.2 (2017-12-15)
------------------

Mostly a maintenance release. Much cleaning happened and a conda package is now
available in conda-forge. Moreover, the package can now be tried online thanks
to binder.

0.5.1 (2017-07-04)
------------------

Development status updated from Alpha to Beta.

New features:

* Acceleration module, decoupling acceleration strategies from the solvers

  * Backtracking scheme
  * FISTA acceleration
  * FISTA with backtracking
  * Regularized non-linear acceleration (RNA)

* Solvers: gradient descent algorithm

Bug fix:

* Decrease dimensionality of variables in Douglas Rachford tutorial to reduce
  test time and timeout on Travis CI.

Infrastructure:

* Continuous integration: dropped 3.3 (matplotlib dropped it), added 3.6
* We don't build PDF documentation anymore. Less burden, HTML can be downloaded
  from readthedocs.

0.4.0 (2016-08-01)
------------------

New feature:

* Monotone+Lipschitz forward-backward-forward primal-dual algorithm (MLFBF)

Bug fix:

* Plots generated when building documentation (not stored in the repository)

Infrastructure:

* Continuous integration: dropped 2.6 and 3.2, added 3.5
* Travis-ci: check style and build doc
* Removed tox config (too cumbersome to use on dev box)
* Monitor code coverage and report to coveralls.io

0.3.0 (2015-05-29)
------------------

New features:

* Generalized forward-backward splitting algorithm
* Projection-based primal-dual algorithm
* TV-norm function (eval, prox)
* Nuclear-norm function (eval, prox)
* L2-norm proximal operator supports non-tight frames
* Two new tutorials using the TV-norm with Forward-Backward and
  Douglas-Rachford for image reconstruction and denoising
* New stopping criterion XTOL allows to stop when the variable is stable

Bug fix:

* Much more memory efficient. Note that the array which contains the initial
  solution is now modified in place.

0.2.1 (2014-08-20)
------------------

Bug fix version. Still experimental.

Bug fixes:

* Avoid complex casting to real
* Do not stop iterating if the objective function stays at zero

0.2.0 (2014-08-04)
------------------

Second usable version, available on GitHub and released on PyPI.
Still experimental.

New features:

* Douglas-Rachford splitting algorithm
* Projection on the L2-ball for tight and non tight frames
* Compressed sensing tutorial using L2-ball, L2-norm and Douglas-Rachford
* Automatic solver selection

Infrastructure:

* Unit tests for all functions and solvers
* Continuous integration testing on Python 2.6, 2.7, 3.2, 3.3 and 3.4

0.1.0 (2014-06-08)
------------------

First usable version, available on GitHub and released on PyPI.
Still experimental.

Features:

* Forward-backward splitting algorithm
* L1-norm function (eval and prox)
* L2-norm function (eval, grad and prox)
* Least square problem tutorial using L2-norm and forward-backward
* Compressed sensing tutorial using L1-norm, L2-norm and forward-backward

Infrastructure:

* Sphinx generated documentation using Numpy style docstrings
* Documentation hosted on Read the Docs
* Code hosted on GitHub
* Package hosted on PyPI
* Code checked by flake8
* Docstring and tutorial examples checked by doctest (as a test suite)
* Unit tests for functions module (as a test suite)
* All test suites executed in Python 2.6, 2.7 and 3.2 virtualenvs by tox
* Distributed automatic testing on Travis CI continuous integration platform
