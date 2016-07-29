#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.
"""

from . import test_functions, test_operators, test_solvers
import unittest
import doctest
import glob
import os


def test_tutorials():
    path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    path = os.path.join(path, os.path.pardir, 'doc', 'tutorials')
    path = os.path.abspath(path)
    files = glob.glob(os.path.join(path, '*.rst'))
    return doctest.DocFileSuite(*files, module_relative=False)


def test_docstrings():
    path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    path = os.path.abspath(path)
    files = glob.glob(os.path.join(path, '*.py'))
    return doctest.DocFileSuite(*files, module_relative=False)


suites = []
suites.append(test_functions.suite)
suites.append(test_operators.suite)
suites.append(test_solvers.suite)
suites.append(test_docstrings())
suites.append(test_tutorials())
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
