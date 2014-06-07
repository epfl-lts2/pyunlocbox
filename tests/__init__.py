#!/usr/bin/env python
# -*- coding: utf-8 -*-

# When importing the tests, you surely want these modules.
from tests import test_functions
from tests import test_doc
from tests import test_all

# Silence the code checker warning about unused symbols.
assert test_functions
assert test_doc
assert test_all
