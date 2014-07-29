#!/usr/bin/env python
# -*- coding: utf-8 -*-

# When importing the tests, you surely want these modules.
from pyunlocbox.tests import test_functions, test_solvers
from pyunlocbox.tests import test_docstrings, test_tutorials
from pyunlocbox.tests import test_all

# Silence the code checker warning about unused symbols.
assert test_functions
assert test_solvers
assert test_docstrings
assert test_tutorials
assert test_all
