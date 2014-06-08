#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.
"""

import unittest
from pyunlocbox.tests import test_functions, test_doc


suites = []

suites.append(test_functions.suite)
suites.append(test_doc.suite)

suite = unittest.TestSuite(suites)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
