=====
About
=====

PyUNLocBoX is a convex optimization toolbox using proximal splitting methods
implemented in Python. It is a free software distributed under the BSD license
and is a port of the Matlab UNLocBoX toolbox.

.. image:: https://img.shields.io/travis/epfl-lts2/pyunlocbox.svg
   :target: https://travis-ci.org/epfl-lts2/pyunlocbox

.. image:: https://img.shields.io/coveralls/epfl-lts2/pyunlocbox.svg
   :target: https://coveralls.io/github/epfl-lts2/pyunlocbox

.. image:: https://img.shields.io/pypi/v/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox

.. image:: https://img.shields.io/pypi/l/pyunlocbox.svg

.. image:: https://img.shields.io/pypi/pyversions/pyunlocbox.svg

* Development : https://github.com/epfl-lts2/pyunlocbox
* Documentation : https://pyunlocbox.readthedocs.io
* PyPI package : https://pypi.python.org/pypi/pyunlocbox
* Travis continuous integration : https://travis-ci.org/epfl-lts2/pyunlocbox
* UNLocBoX matlab toolbox : https://lts2.epfl.ch/unlocbox

Features
--------

* Solvers

  * Forward-backward splitting algorithm
  * Douglas-Rachford splitting algorithm
  * Monotone+Lipschitz Forward-Backward-Forward primal-dual algorithm
  * Projection-based primal-dual algorithm

* Proximal operators

  * L1-norm
  * L2-norm
  * TV-norm
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
Solution found after 10 iterations:
    objective function f(sol) = 7.460428e-09
    stopping criterion: ATOL
>>> ret['sol']
array([ 3.99996922,  4.99996153,  5.99995383,  6.99994614])

Installation
------------

PyUnLocBox is continuously tested on Python 2.7, 3.3, 3.4 and 3.5.

System-wide installation::

    $ pip install pyunlocbox

Installation in an isolated virtual environment::

    $ mkvirtualenv --system-site-packages pyunlocbox
    $ pip install pyunlocbox

You need virtualenvwrapper to run this command. The ``--system-site-packages``
option could be useful if you want to use a shared system installation of numpy
and matplotlib. Their building and installation require quite some
dependencies.

Another way is to manually download from PyPI, unpack the package and install
with::

    $ python setup.py install

Execute the project test suite once to make sure you have a working install::

    $ python setup.py test

Authors
-------

PyUNLocBoX was started in 2014 as an academic project for research purpose at
the LTS2 laboratory from EPFL (https://lts2.epfl.ch).

Development lead :

* Michaël Defferrard from EPFL LTS2 <michael.defferrard@epfl.ch>
* Nathanaël Perraudin from EPFL LTS2 <nathanael.perraudin@epfl.ch>

Contributors :

* Alexandre Lafaye from EPFL LTS2 <alexandre.lafaye@epfl.ch>
* Basile Châtillon from EPFL LTS2 <basile.chatillon@epfl.ch>
* Nicolas Rod from EPFL LTS2 <nicolas.rod@epfl.ch>
* Rodrigo Pena from EPFL LTS2 <rodrigo.pena@epfl.ch>
