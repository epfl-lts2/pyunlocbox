=====
About
=====

PyUNLocBoX is a convex optimization toolbox using proximal splitting methods
implemented in Python. It is a free software distributed under the BSD license
and is a port of the Matlab UNLocBoX toolbox.

.. image:: https://badge.fury.io/py/pyunlocbox.png
    :target: https://badge.fury.io/py/pyunlocbox

.. image:: https://travis-ci.org/epfl-lts2/pyunlocbox.png?branch=master
    :target: https://travis-ci.org/epfl-lts2/pyunlocbox

.. image:: https://pypip.in/d/pyunlocbox/badge.png
    :target: https://crate.io/packages/pyunlocbox?version=latest

* Code : https://github.com/epfl-lts2/pyunlocbox
* Documentation : http://pyunlocbox.readthedocs.org
* PyPI package : https://pypi.python.org/pypi/pyunlocbox
* Travis continuous integration : https://travis-ci.org/epfl-lts2/pyunlocbox
* UNLocBoX matlab toolbox : http://unlocbox.sourceforge.net

Features
--------

* Solvers

  * Forward-backward splitting algorithm

* Proximal operators

  * L1-norm
  * L2-norm

Installation
------------

System-wide installation::

    # pip install pyunlocbox

Installation in an isolated virtual environment::

    $ mkvirtualenv --system-site-packages pyunlocbox
    $ pip install pyunlocbox

You need virtualenvwrapper to run this command. The ``--system-site-packages``
option could be useful if you want to use a shared system installation of numpy
and matplotlib. Their building and installation requires quite some
dependencies.

Another way is to manually download from PyPI and unpack the package then
install with::

    $ python setup.py install

Execute the project test suite once to make sure you have a working install::

    $ python setup.py test

Authors
-------

PyUNLocBoX was started in 2014 as an academic project for research purpose of
the LTS2 laboratory from EPFL. See our website at http://lts2www.epfl.ch.

Development lead :

* Michaël Defferrard from EPFL LTS2 <michael.defferrard@epfl.ch>
* Nathanaël Perraudin from EPFL LTS2 <nathanael.perraudin@epfl.ch>

Contributors :

* None yet. Why not be the first ?
