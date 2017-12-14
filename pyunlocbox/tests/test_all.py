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
    for root, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, module_relative=False)


suites = []
suites.append(test_functions.suite)
suites.append(test_operators.suite)
suites.append(test_solvers.suite)
suites.append(test_acceleration.suite)
suites.append(test_docstrings('pyunlocbox', '.py'))
suites.append(test_docstrings('.', '.rst'))
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
