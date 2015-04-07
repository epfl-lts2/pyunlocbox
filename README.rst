=====
About
=====

PyUNLocBoX is a convex optimization toolbox using proximal splitting methods
implemented in Python. It is a free software distributed under the BSD license
and is a port of the Matlab UNLocBoX toolbox.

.. image:: https://img.shields.io/pypi/v/pyunlocbox.svg
   :target: https://pypi.python.org/pypi/pyunlocbox

.. image:: https://img.shields.io/travis/epfl-lts2/pyunlocbox.svg
   :target: https://travis-ci.org/epfl-lts2/pyunlocbox

.. image:: https://img.shields.io/pypi/l/pyunlocbox.svg

* Development : https://github.com/epfl-lts2/pyunlocbox
* Documentation : http://pyunlocbox.readthedocs.org
* PyPI package : https://pypi.python.org/pypi/pyunlocbox
* Travis continuous integration : https://travis-ci.org/epfl-lts2/pyunlocbox
* UNLocBoX matlab toolbox : http://unlocbox.sourceforge.net

Features
--------

* Solvers

  * Forward-backward splitting algorithm
  * Douglas-Rachford splitting algorithm

* Proximal operators

  * L1-norm
  * L2-norm
  * TV-norm
  * Projection on the L2-ball

Installation
------------

PyUnLocBox is continuously tested with Python 2.6, 2.7, 3.2, 3.3 and 3.4.

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
the LTS2 laboratory from EPFL. See our website at http://lts2www.epfl.ch.

Development lead :

* Michaël Defferrard from EPFL LTS2 <michael.defferrard@epfl.ch>
* Nathanaël Perraudin from EPFL LTS2 <nathanael.perraudin@epfl.ch>

Contributors :

* Alexandre Lafaye from EPFL LTS2 <alexandre.lafaye@epfl.ch>
* Basile Châtillon from EPFL LTS2 <basile.chatillon@epfl.ch>
* Nicolas Rod from EPFL LTS2 <nicolas.rod@epfl.ch>
