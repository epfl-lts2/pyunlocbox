#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the documentation of the pyunlocbox package.
"""

import doctest
import glob
import os
import unittest


# Test examples in documentation.
path = os.path.join(os.path.dirname(__file__), os.path.pardir)
path = os.path.join(path, os.path.pardir, 'doc', 'tutorials')
path = os.path.abspath(path)
files = glob.glob(os.path.join(path, '*.rst'))

assert files

suite = doctest.DocFileSuite(*files, module_relative=False, encoding='utf-8')


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
