#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.
"""

import unittest
from pyunlocbox.tests import test_functions, test_solvers
from pyunlocbox.tests import test_docstrings, test_tutorials


suites = []

suites.append(test_functions.suite)
suites.append(test_solvers.suite)
suites.append(test_docstrings.suite)
suites.append(test_tutorials.suite)

suite = unittest.TestSuite(suites)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
