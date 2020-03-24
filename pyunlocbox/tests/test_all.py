#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.

"""

import unittest

from pyunlocbox.tests import test_functions
from pyunlocbox.tests import test_operators
from pyunlocbox.tests import test_solvers
from pyunlocbox.tests import test_acceleration
from pyunlocbox.tests import test_docstrings


suites = []
suites.append(test_functions.suite)
suites.append(test_operators.suite)
suites.append(test_solvers.suite)
suites.append(test_acceleration.suite)
suites.append(test_docstrings.suite)
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
