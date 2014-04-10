#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the functions module of the pyunlocbox package.
"""

import unittest
import numpy.testing as nptest
from pyunlocbox import functions


class TestPyunlocbox(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def eval(self, x):
        return x**2 - 5

    def grad(self, x):
        return 2*x

    def prox(self, x, T):
        return x+T

    def test_func(self):
        """
        Test the func base class.
        First test that all the methods raise a NotImplemented exception.
        Then assign valid methods and test return values.
        """
        x = 10
        T = 1
        f = functions.func()
        self.assertRaises(NotImplementedError, f.eval, x)
        self.assertRaises(NotImplementedError, f.prox, x, T)
        self.assertRaises(NotImplementedError, f.grad, x)
        f.eval = self.eval
        f.grad = self.grad
        f.prox = self.prox
        self.assertEqual(f.eval(x), self.eval(x))
        self.assertEqual(f.grad(x), self.grad(x))
        self.assertEqual(f.prox(x, T), self.prox(x, T))

    def test_norm_l2(self):
        """
        Test the norm_l2 derived class.
        We test the three methods : eval, grad and prox.
        First with default class properties, then custom ones.
        """
        f = functions.norm_l2(lamb=2)
        self.assertEqual(f.eval(10), 20)
        self.assertEqual(f.eval(-10), 20)
        self.assertEqual(f.grad(10), 40)
        self.assertEqual(f.grad(-10), -40)
        self.assertEqual(f.eval([3,4]), 2*5)
        self.assertEqual(f.eval([-3,4]), 2*5)
        nptest.assert_allclose(f.grad([3,4]), [12,16])
        nptest.assert_allclose(f.grad([3,-4]), [12,-16])

if __name__ == '__main__':
    unittest.main()
