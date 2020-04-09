# -*- coding: utf-8 -*-

"""
Test suite for the docstrings of the pyunlocbox package.

"""

import os
import unittest
import doctest


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, module_relative=False)


# Docstrings from API reference.
suite_reference = test_docstrings('pyunlocbox', '.py')

# Docstrings from tutorials.
suite_tutorials = test_docstrings('.', '.rst')

suite = unittest.TestSuite([suite_reference, suite_tutorials])
