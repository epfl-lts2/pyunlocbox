#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.

"""

import os
import unittest
import doctest

from . import test_functions, test_operators, test_solvers, test_acceleration


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, module_relative=False)


def setup(doctest):
    import numpy
    import pyunlocbox
    doctest.globs = {
        'functions': pyunlocbox.functions,
        'solvers': pyunlocbox.solvers,
        'acceleration': pyunlocbox.acceleration,
        'operators': pyunlocbox.operators,
        'np': numpy,
    }


suites = []
suites.append(test_functions.suite)
suites.append(test_operators.suite)
suites.append(test_solvers.suite)
suites.append(test_acceleration.suite)
suites.append(test_docstrings('pyunlocbox', '.py', setup))
suites.append(test_docstrings('.', '.rst'))  # No setup to not forget imports.
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
