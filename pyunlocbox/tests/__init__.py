#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package, broken by modules.

"""

import unittest

from . import test_functions
from . import test_operators
from . import test_solvers
from . import test_acceleration
from . import test_docstrings


suite = unittest.TestSuite([
    test_functions.suite,
    test_operators.suite,
    test_solvers.suite,
    test_acceleration.suite,
    test_docstrings.suite,
])
