#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the functions module of the pyunlocbox package.
"""

import sys
import numpy as np
import numpy.testing as nptest
from pyunlocbox import functions

# Use the unittest2 backport on Python 2.6 to profit from the new features.
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class FunctionsTestCase(unittest.TestCase):

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

    def test_dummy(self):
        """
        Test the dummy derived class.
        All the methods should return 0.
        """
        f = functions.dummy()
        self.assertEqual(f.eval(34), 0)
        nptest.assert_array_equal(f.grad(34), [0])
        nptest.assert_array_equal(f.prox(34, 1), [34])
        x = [34, 2, 1.0, -10.2]
        self.assertEqual(f.eval(x), 0)
        nptest.assert_array_equal(f.grad(x), np.zeros(len(x)))
        nptest.assert_array_equal(f.prox(x, 1), x)

    def test_norm_l2(self):
        """
        Test the norm_l2 derived class.
        We test the three methods : eval, grad and prox.
        First with default class properties, then custom ones.
        """
        f = functions.norm_l2(lambda_=3)
        self.assertEqual(f.eval([10, 0]), 300)
        self.assertEqual(f.eval(np.array([-10, 0])), 300)
        nptest.assert_allclose(f.grad([10, 0]), [60, 0])
        nptest.assert_allclose(f.grad([-10, 0]), [-60, 0])
        self.assertEqual(f.eval([3, 4]), 3*5**2)
        self.assertEqual(f.eval(np.array([-3, 4])), 3*5**2)
        nptest.assert_allclose(f.grad([3, 4]), [18, 24])
        nptest.assert_allclose(f.grad([3, -4]), [18, -24])
        self.assertEqual(f.prox(0, 1), 0)
        self.assertEqual(f.prox(7, 1./6), 3.5)
        f = functions.norm_l2(lambda_=4)
        nptest.assert_allclose(f.prox([7, -22], .125), [3.5, -11])

        f = functions.norm_l2(lambda_=1, A=lambda x: 2*x, At=lambda x: x/2,
                              y=[8, 12])
        self.assertEqual(f.eval([4, 6]), 0)
        self.assertEqual(f.eval([5, -2]), 256+4)
        nptest.assert_allclose(f.grad([4, 6]), 0)
#        nptest.assert_allclose(f.grad([5, -2]), [8, -64])
        nptest.assert_allclose(f.prox([4, 6], 1), [4, 6])

        f = functions.norm_l2(lambda_=2, y=np.fft.fft([2, 4])/np.sqrt(2),
                              A=lambda x: np.fft.fft(x)/np.sqrt(x.size),
                              At=lambda x: np.fft.ifft(x)*np.sqrt(x.size))
#        self.assertEqual(f.eval(np.fft.ifft([2, 4])*np.sqrt(2)), 0)
#        self.assertEqual(f.eval([3, 5]), 2*np.sqrt(25+81))
        nptest.assert_allclose(f.grad([2, 4]), 0)
#        nptest.assert_allclose(f.grad([3, 5]), [4*np.sqrt(5), 4*3])
        nptest.assert_allclose(f.prox([2, 4], 1), [2, 4])
        nptest.assert_allclose(f.prox([3, 5], 1), [2.2, 4.2])
        nptest.assert_allclose(f.prox([2.2, 4.2], 1), [2.04, 4.04])
        nptest.assert_allclose(f.prox([2.04, 4.04], 1), [2.008, 4.008])

    def test_soft_thresholding(self):
        """
        Test the soft thresholding helper function.
        """
        x = np.arange(-4, 5, 1)
        # Test integer division for complex method.
        Ts = [2]
        y_gold = [[-2, -1, 0, 0, 0, 0, 0, 1, 2]]
        # Test division by 0 for complex method.
        Ts.append([.4, .3, .2, .1, 0, .1, .2, .3, .4])
        y_gold.append([-3.6, -2.7, -1.8, -.9, 0, .9, 1.8, 2.7, 3.6])
        for k, T in enumerate(Ts):
            for cmplx in [False, True]:
                y_test = functions._soft_threshold(x, T, cmplx)
                nptest.assert_array_equal(y_test, y_gold[k])

    def test_norm_l1(self):
        """
        Test the norm_l1 derived class.
        We test the two methods : eval and prox.
        First with default class properties, then custom ones.
        """
        f = functions.norm_l1(lambda_=3)
        self.assertEqual(f.eval([10, 0]), 30)
        self.assertEqual(f.eval(np.array([-10, 0])), 30)
        self.assertEqual(f.eval([3, 4]), 21)
        self.assertEqual(f.eval(np.array([-3, 4])), 21)

    def test_norm_tv(self):
        """
        Test the norm_tv derived class.
        We test the grad, div, eval and prox.
        """

        # Test Matrices initialization
        # test for a 1dim matrice (testing with a 5)
        mat1d = np.array([1., 2., 3., 4., 5.])
        # test for a 2dim matrice (testing with a 2x4)
        mat2d = np.array([[2., 3., 0., 1.], [22., 1., 4., 5.]])
        # test for a 3 dim matrice (testing with a 2x3x2)
        mat3d = np.array([[[1., 7.], [2., 8.], [3., 9.]],
                          [[4., 10.], [5., 11.], [6., 12.]]])
        # test for a 4dim matrice (2x3x2x2)
        mat4d = np.array([[[[1., 13.], [7., 19.]], [[2., 14.], [8., 20.]],
                          [[3., 15.], [9., 21.]]], [[[4., 16.], [10., 22.]],
                          [[5., 17.], [11., 23.]], [[6., 18.], [12., 24.]]]])
        # test for a 5dim matrice (2x2x3x2x2)
        mat5d = np.array([[[[[1, 25], [13, 37]],
                            [[5, 29], [17, 41]],
                            [[9, 33], [21, 45]]],
                           [[[2, 26], [14, 38]],
                            [[6, 30], [18, 42]],
                            [[10, 34], [22, 46]]]],
                          [[[[3, 27], [15, 39]],
                            [[7, 31], [19, 43]],
                            [[11, 35], [23, 47]]],
                           [[[4, 28], [16, 40]],
                            [[8, 32], [20, 44]],
                            [[12, 36], [24, 48]]]]])

        # test without weight
        f = functions.norm_tv(dim=1, wx=1, wy=1, wz=1, wt=1)
        dx = f.grad(mat1d)
        nptest.assert_array_equal(np.array([1, 1, 1, 1, 0]), dx)

        # test without weight
        f = functions.norm_tv(dim=1)
        dx = f.grad(mat2d)
        mat_dx = np.array([[20, -2, 4, 4], [0, 0, 0, 0]])
        mat_dy = np.array([[1, -3, 1, 0], [-21, 3, 1, 0]])
        nptest.assert_array_equal(mat_dx, dx)

        f = functions.norm_tv(dim=2)
        dx, dy = f.grad(mat2d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)

        # test with weights
        f = functions.norm_tv(dim=2, wx=2, wy=0.5, wz=3, wt=2)
        dx, dy = f.grad(mat2d)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)

        # test without weight
        f = functions.norm_tv(dim=1)
        dx = f.grad(mat3d)
        mat_dx = np.array([[[3, 3], [3, 3], [3, 3]],
                           [[0, 0], [0, 0], [0, 0]]])
        mat_dy = np.array([[[1, 1], [1, 1], [0, 0]],
                           [[1, 1], [1, 1], [0, 0]]])
        mat_dz = np.array([[[6, 0], [6, 0], [6, 0]],
                           [[6, 0], [6, 0], [6, 0]]])
        nptest.assert_array_equal(mat_dx, dx)
        f = functions.norm_tv(dim=2)
        dx, dy = f.grad(mat3d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        f = functions.norm_tv(dim=3)
        dx, dy, dz = f.grad(mat3d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)

        # test with weights
        f = functions.norm_tv(dim=3, wx=2, wy=0.5, wz=3, wt=2)
        dx, dy, dz = f.grad(mat3d)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)

        # test without weight
        f = functions.norm_tv(dim=1)
        dx = f.grad(mat4d)
        mat_dx = np.array([[[[3, 3], [3, 3]],
                            [[3, 3], [3, 3]],
                            [[3, 3], [3, 3]]],
                           [[[0, 0], [0, 0]],
                            [[0, 0], [0, 0]],
                            [[0, 0], [0, 0]]]])
        mat_dy = np.array([[[[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[0, 0], [0, 0]]],
                           [[[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[0, 0], [0, 0]]]])
        mat_dz = np.array([[[[6, 6], [0, 0]],
                            [[6, 6], [0, 0]],
                            [[6, 6], [0, 0]]],
                           [[[6, 6], [0, 0]],
                            [[6, 6], [0, 0]],
                            [[6, 6], [0, 0]]]])
        mat_dt = np.array([[[[12, 0], [12, 0]],
                            [[12, 0], [12, 0]],
                            [[12, 0], [12, 0]]],
                           [[[12, 0], [12, 0]],
                            [[12, 0], [12, 0]],
                            [[12, 0], [12, 0]]]])
        nptest.assert_array_equal(mat_dx, dx)
        f = functions.norm_tv(dim=2)
        dx, dy = f.grad(mat4d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        f = functions.norm_tv(dim=3)
        dx, dy, dz = f.grad(mat4d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        f = functions.norm_tv(dim=4)
        dx, dy, dz, dt = f.grad(mat4d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        nptest.assert_array_equal(mat_dt, dt)

        # test with weights
        f = functions.norm_tv(dim=4, wx=2, wy=0.5, wz=3, wt=2)
        dx, dy, dz, dt = f.grad(mat4d)
        # Because of numpy, can't *=
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        mat_dt_w = mat_dt * 2.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)
        nptest.assert_array_equal(mat_dt_w, dt)

        # test without weight
        f = functions.norm_tv(dim=1)
        dx = f.grad(mat5d)
        mat_dx = np.array([[[[[2, 2], [2, 2]],
                             [[2, 2], [2, 2]],
                             [[2, 2], [2, 2]]],
                            [[[2, 2], [2, 2]],
                             [[2, 2], [2, 2]],
                             [[2, 2], [2, 2]]]],
                           [[[[0, 0], [0, 0]],
                             [[0, 0], [0, 0]],
                             [[0, 0], [0, 0]]],
                            [[[0, 0], [0, 0]],
                             [[0, 0], [0, 0]],
                             [[0, 0], [0, 0]]]]])
        mat_dy = np.array([[[[[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]]],
                            [[[0, 0], [0, 0]],
                             [[0, 0], [0, 0]],
                             [[0, 0], [0, 0]]]],
                           [[[[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]]],
                            [[[0, 0], [0, 0]],
                             [[0, 0], [0, 0]],
                             [[0, 0], [0, 0]]]]])
        mat_dz = np.array([[[[[4, 4], [4, 4]],
                             [[4, 4], [4, 4]],
                             [[0, 0], [0, 0]]],
                            [[[4, 4], [4, 4]],
                             [[4, 4], [4, 4]],
                             [[0, 0], [0, 0]]]],
                           [[[[4, 4], [4, 4]],
                             [[4, 4], [4, 4]],
                             [[0, 0], [0, 0]]],
                            [[[4, 4], [4, 4]],
                             [[4, 4], [4, 4]],
                             [[0, 0], [0, 0]]]]])
        mat_dt = np.array([[[[[12, 12], [0, 0]],
                             [[12, 12], [0, 0]],
                             [[12, 12], [0, 0]]],
                            [[[12, 12], [0, 0]],
                             [[12, 12], [0, 0]],
                             [[12, 12], [0, 0]]]],
                           [[[[12, 12], [0, 0]],
                             [[12, 12], [0, 0]],
                             [[12, 12], [0, 0]]],
                            [[[12, 12], [0, 0]],
                             [[12, 12], [0, 0]],
                             [[12, 12], [0, 0]]]]])
        nptest.assert_array_equal(mat_dx, dx)
        f = functions.norm_tv(dim=2)
        dx, dy = f.grad(mat5d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        f = functions.norm_tv(dim=3)
        dx, dy, dz = f.grad(mat5d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        f = functions.norm_tv(dim=4)
        dx, dy, dz, dt = f.grad(mat5d)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        nptest.assert_array_equal(mat_dt, dt)

        # test with weights
        f = functions.norm_tv(dim=4, wx=2, wy=0.5, wz=3, wt=2)
        dx, dy, dz, dt = f.grad(mat5d)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        mat_dt_w = mat_dt * 2.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)
        nptest.assert_array_equal(mat_dt_w, dt)

        # Divergence tests
        # test with 1dim matrices
        dx = np.array([1, 2, 3, 4, 5])

        # test without weights
        f = functions.norm_tv(dim=1)
        nptest.assert_array_equal(np.array([1, 1, 1, 1, -4]), f._div(dx))

        # test with weights
        f = functions.norm_tv(dim=1, wx=2, wy=3, wz=4, wt=2)
        nptest.assert_array_equal(np.array([2, 2, 2, 2, -8]), f._div(dx))

        # test with 2dim matrices
        dx = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        dy = np.array([[13, 14, 15, 16],
                       [17, 18, 19, 20],
                       [21, 22, 23, 24]])

        # test without weights
        f = functions.norm_tv(dim=2)
        x_mat = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [-5, -6, -7, -8]])
        xy_mat = np.array([[14, 3, 4, -11],
                           [21, 5, 5, -15],
                           [16, -5, -6, -31]])
        nptest.assert_array_equal(x_mat, f._div(dx))
        nptest.assert_array_equal(xy_mat, f._div(dx, dy))

        # test with weights
        f = functions.norm_tv(dim=2, wx=2, wy=3, wz=4, wt=2)
        xy_mat_w = np.array([[41, 7, 9, -37],
                             [59, 11, 11, -49],
                             [53, -9, -11, -85]])
        nptest.assert_array_equal(xy_mat_w, f._div(dx, dy))

        # test with 3dim matrices (3x3x3)
        f = functions.norm_tv(dim=3, wx=2, wy=3, wz=4, wt=2)
        dx = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
                       [[4, 13, 22], [5, 14, 23], [6, 15, 24]],
                       [[7, 16, 25], [8, 17, 26], [9, 18, 27]]])
        dy = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
                       [[4, 13, 22], [5, 14, 23], [6, 15, 24]],
                       [[7, 16, 25], [8, 17, 26], [9, 18, 27]]])
        dz = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
                       [[4, 13, 22], [5, 14, 23], [6, 15, 24]],
                       [[7, 16, 25], [8, 17, 26], [9, 18, 27]]])
        # test without weights
        f = functions.norm_tv(dim=3)
        x_mat = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
                          [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                          [[-4, -13, -22], [-5, -14, -23], [-6, -15, -24]]])
        xy_mat = np.array([[[2, 20, 38], [3, 12, 21], [1, 1, 1]],
                           [[7, 16, 25], [4, 4, 4], [-2, -11, -20]],
                           [[3, 3, 3], [-4, -13, -22], [-14, -32, -50]]])
        xyz_mat = np.array([[[3, 29, 28], [5, 21, 10], [4, 10, -11]],
                            [[11, 25, 12], [9, 13, -10], [4, -2, -35]],
                            [[10, 12, -13], [4, -4, -39], [-5, -23, -68]]])
        xyzt_mat = np.array([[[9, 86, 55], [15, 61, -1], [12, 27, -66]],
                             [[34, 81, 20], [29, 45, -47], [15, 0, -123]],
                             [[41, 58, -33], [25, 11, -111], [0, -45, -198]]])
        nptest.assert_array_equal(x_mat, f._div(dx))
        nptest.assert_array_equal(xy_mat,  f._div(dx, dy))
        nptest.assert_array_equal(xyz_mat, f._div(dx, dy, dz))
        # test with weights
        f = functions.norm_tv(dim=3, wx=2, wy=3, wz=4, wt=2)
        xyz_mat_w = np.array([[[9, 86, 55], [15, 61, -1], [12, 27, -66]],
                              [[34, 81, 20], [29, 45, -47], [15, 0, -123]],
                              [[41, 58, -33], [25, 11, -111], [0, -45, -198]]])
        nptest.assert_array_equal(xyz_mat_w, f._div(dx, dy, dz))

        # test with 4d matrices (3x3x3x3)
        dx = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                        [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                        [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                       [[[4, 31, 58], [13, 40, 67], [22, 49, 76]],
                        [[5, 32, 59], [14, 41, 68], [23, 50, 77]],
                        [[6, 33, 60], [15, 42, 69], [24, 51, 78]]],
                       [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                        [[8, 35, 62], [17, 44, 71], [26, 53, 80]],
                        [[9, 36, 63], [18, 45, 72], [27, 54, 81]]]])
        dy = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                        [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                        [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                       [[[4, 31, 58], [13, 40, 67], [22, 49, 76]],
                        [[5, 32, 59], [14, 41, 68], [23, 50, 77]],
                        [[6, 33, 60], [15, 42, 69], [24, 51, 78]]],
                       [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                        [[8, 35, 62], [17, 44, 71], [26, 53, 80]],
                        [[9, 36, 63], [18, 45, 72], [27, 54, 81]]]])
        dz = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                        [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                        [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                       [[[4, 31, 58], [13, 40, 67], [22, 49, 76]],
                        [[5, 32, 59], [14, 41, 68], [23, 50, 77]],
                        [[6, 33, 60], [15, 42, 69], [24, 51, 78]]],
                       [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                        [[8, 35, 62], [17, 44, 71], [26, 53, 80]],
                        [[9, 36, 63], [18, 45, 72], [27, 54, 81]]]])
        dt = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                        [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                        [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                       [[[4, 31, 58], [13, 40, 67], [22, 49, 76]],
                        [[5, 32, 59], [14, 41, 68], [23, 50, 77]],
                        [[6, 33, 60], [15, 42, 69], [24, 51, 78]]],
                       [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                        [[8, 35, 62], [17, 44, 71], [26, 53, 80]],
                        [[9, 36, 63], [18, 45, 72], [27, 54, 81]]]])
        # test without weights
        f = functions.norm_tv(dim=4)
        x_mat = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                           [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                           [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                          [[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                           [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                           [[3, 3, 3], [3, 3, 3], [3, 3, 3]]],
                          [[[-4, -31, -58], [-13, -40, -67], [-22, -49, -76]],
                           [[-5, -32, -59], [-14, -41, -68], [-23, -50, -77]],
                           [[-6, -33, -60], [-15, -42, -69], [-24, -51, -78]]]])
        xy_mat = np.array([[[[2, 56, 110], [20, 74, 128], [38, 92, 146]],
                            [[3, 30, 57], [12, 39, 66], [21, 48, 75]],
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                           [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                            [[4, 4, 4], [4, 4, 4], [4, 4, 4]],
                            [[-2, -29, -56], [-11, -38, -65], [-20, -47, -74]]],
                           [[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                            [[-4, -31, -58], [-13, -40, -67], [-22, -49, -76]],
                            [[-14, -68, -122], [-32, -86, -140], [-50, -104, -158]]]])
        xyz_mat = np.array([[[[3, 84, 165], [29, 83, 137], [28, 55, 82]],
                             [[5, 59, 113], [21, 48, 75], [10, 10, 10]],
                             [[4, 31, 58], [10, 10, 10], [-11, -38, -65]]],
                            [[[11, 65, 119], [25, 52, 79], [12, 12, 12]],
                             [[9, 36, 63], [13, 13, 13], [-10, -37, -64]],
                             [[4, 4, 4], [-2, -29, -56], [-35, -89, -143]]],
                            [[[10, 37, 64], [12, 12, 12], [-13, -40, -67]],
                             [[4, 4, 4], [-4, -31, -58], [-39, -93, -147]],
                             [[-5, -32, -59], [-23, -77, -131], [-68, -149, -230]]]])
        xyzt_mat = np.array([[[[4, 111, 137], [39, 110, 100], [47, 82, 36]],
                              [[7, 86, 84], [32, 75, 37], [30, 37, -37]],
                              [[7, 58, 28], [22, 37, -29], [10, -11, -113]]],
                             [[[15, 92, 88], [38, 79, 39], [34, 39, -37]],
                              [[14, 63, 31], [27, 40, -28], [13, -10, -114]],
                              [[10, 31, -29], [13, -2, -98], [-11, -62, -194]]],
                             [[[17, 64, 30], [28, 39, -31], [12, -13, -119]],
                              [[12, 31, -31], [13, -4, -102], [-13, -66, -200]],
                              [[4, -5, -95], [-5, -50, -176], [-41, -122, -284]]]])
        nptest.assert_array_equal(x_mat, f._div(dx))
        nptest.assert_array_equal(xy_mat, f._div(dx, dy))
        nptest.assert_array_equal(xyz_mat, f._div(dx, dy, dz))
        nptest.assert_array_equal(xyzt_mat, f._div(dx, dy, dz, dt))
        # test with weights
        f = functions.norm_tv(dim=4, wx=2, wy=3, wz=4, wt=2)
        xyzt_mat_w = np.array([[[[11, 306, 439], [106, 275, 282], [93, 136, 17]],
                                [[19, 231, 281], [83, 169, 93], [39, -1, -203]],
                                [[18, 147, 114], [51, 54, -105], [-24, -147, -432]]],
                               [[[42, 277, 350], [107, 216, 163], [64, 47, -132]],
                                [[39, 191, 181], [73, 99, -37], [-1, -101, -363]],
                                [[27, 96, 3], [30, -27, -246], [-75, -258, -603]]],
                               [[[55, 230, 243], [90, 139, 26], [17, -60, -299]],
                                [[41, 133, 63], [45, 11, -185], [-59, -219, -541]],
                                [[18, 27, -126], [-9, -126, -405], [-144, -387, -792]]]])
        nptest.assert_array_equal(xyzt_mat_w, f._div(dx, dy, dz, dt))

        # Test for eval
        # test with 2d matrices
        f = functions.norm_tv(dim=1)
        nptest.assert_array_equal(np.array([20, 2, 4, 4]), f._eval(mat2d))
        f = functions.norm_tv(dim=2)
        nptest.assert_array_equal(56.753641295582440, f._eval(mat2d))
        # test with 3d matrices (2x3x2)
        f = functions.norm_tv(dim=2)
        sol = np.array([11.324555320336760, 11.324555320336760])
        nptest.assert_array_equal(sol, f._eval(mat3d))

        f = functions.norm_tv(dim=3)
        xeval = np.array([49.762944279683097])
        nptest.assert_array_equal(xeval, f._eval(mat3d))
        # test with 4d matrices (2x2x3x2)

        # Test for prox with 3d array
        f = functions.norm_tv(tol=10e-4, dim=1)
        gamma = 30
        sol = np.array([[12.003459453582762, 1.999654054641723,
                         2.000691890716554, 3.000691890716554],
                        [11.996540546417238, 2.000345945358277,
                         1.999308109283446, 2.999308109283446]])
        nptest.assert_array_equal(np.around(sol, decimals=5),
                                  np.around((f._prox(mat2d, gamma)),
                                            decimals=5))

        f = functions.norm_tv(tol=10e-4, dim=2)
        gamma = 30
        sol = np.array([[4.83374, 4.78352, 4.71408, 4.66606],
                        [4.83463, 4.78465, 4.71555, 4.66776]])
        nptest.assert_array_equal(np.around(sol, decimals=5),
                                  np.around((f._prox(mat2d, gamma)),
                                            decimals=5))

        f = functions.norm_tv(tol=10e-4, dim=2)
        gamma = 42.
        sol = np.array([[[3.50087, 9.50087],
                         [3.50000, 9.50000],
                         [3.49913, 9.49913]],
                        [[3.50087, 9.50087],
                         [3.50000, 9.50000],
                         [3.49913, 9.49913]]])
        nptest.assert_array_equal(sol, np.around(f._prox(mat3d, gamma),
                                                 decimals=5))

        f = functions.norm_tv(tol=10e-4, dim=3)
        gamma = 18.
        sol = np.array([[[6.5, 6.5], [6.5, 6.5], [6.5, 6.5]],
                        [[6.5, 6.5], [6.5, 6.5], [6.5, 6.5]]])
        nptest.assert_array_equal(sol, np.around(f._prox(mat3d, gamma),
                                                 decimals=1))

        f = functions.norm_tv(tol=10e-4, dim=3)
        gamma = 15.
        sol = np.around(np.array([[[[6.4925, 18.4925], [6.4937, 18.4937]],
                                   [[6.4927, 18.4927], [6.4939, 18.4939]],
                                  [[6.4993, 18.4993], [6.5005, 18.5005]]],
                                  [[[6.4995, 18.4995], [6.5007, 18.5007]],
                                  [[6.5061, 18.5061], [6.5073, 18.5073]],
                                   [[6.5063, 18.5063], [6.5075, 18.5075]]]]),
                        decimals=1)
        nptest.assert_array_equal(sol, np.around(f._prox(mat4d, gamma),
                                                 decimals=1))

        f = functions.norm_tv(tol=10e-4, dim=4)
        gamma = 5.
        sol = np.array([[[[9, 14], [12, 14]], [[10, 14], [12, 14]],
                        [[10, 14], [12, 14]]], [[[10, 14], [12, 14]],
                        [[11, 14], [12, 14]], [[11, 14], [13, 14]]]])
        nptest.assert_array_equal(sol, np.around(f._prox(mat4d, gamma)))

    def test_proj_b2(self):
        """
        Test the projection on the L2-ball.
        ISTA and FISTA algorithms for tight and non-tight frames.
        """
        tol = 1e-7

        # Tight frame.
        y = [0, 2]
        x = [5, 4]
        f = functions.proj_b2(y=y, epsilon=tol)
        nptest.assert_allclose(f.prox(x, 0), y, atol=tol)

        # Always evaluate to zero.
        self.assertEqual(f.eval(x), 0)

        # Non-tight frame : compare FISTA and ISTA results.
        nx = 30
        ny = 15
        np.random.seed(1)
        x = np.random.standard_normal(nx)
        y = np.random.standard_normal(ny)
        A = np.random.standard_normal((ny, nx))
        nu = np.linalg.norm(A, ord=2)**2
        f = functions.proj_b2(y=y, A=A, nu=nu, tight=False, method='FISTA',
                              epsilon=5, tol=tol/10.)
        sol_fista = f.prox(x, 0)
        f.method = 'ISTA'
        sol_ista = f.prox(x, 0)
        err = np.linalg.norm(sol_fista - sol_ista) / np.linalg.norm(sol_fista)
        self.assertLess(err, tol)


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
