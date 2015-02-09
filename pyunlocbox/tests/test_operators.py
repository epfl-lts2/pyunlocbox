#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the operators module of the pyunlocbox package
"""

import sys
import numpy as np
import numpy.testing as nptest
from pyunlocbox import operators

# Use the unittest2 backport on Python 2.6 to profit from the new features.
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

class OperatorsTestCase(unittest.TestCase): 

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_grad(self):

        # Test Matrices initialization
        # test for a 1dim matrice (testing with a 5)
        mat1d = np.arange(5.) + 1
        # test for a 2dim matrice (testing with a 2x4)
        mat2d = np.array([[2., 3., 0., 1.], [22., 1., 4., 5.]])
        # test for a 3 dim matrice (testing with a 2x3x2)
        mat3d = np.arange(1., stop=13.).reshape(2, 2, 3).transpose((1, 2, 0))
        # test for a 4dim matrice (2x3x2x2)
        mat4d = np.arange(1., stop=25.).reshape(2, 2, 2, 3).transpose((2, 3, 1, 0))
        # test for a 5dim matrice (2x2x3x2x2)
        mat5d = np.arange(1, stop=49).reshape(2, 2, 3, 2, 2).transpose((3, 4, 2, 1, 0))

        # test without weight
        dx = op.grad(mat1d, dim=1)
        nptest.assert_array_equal(np.array([1, 1, 1, 1, 0]), dx)

        # test without weight
        dx = op.grad(mat2d, dim=2)
        mat_dx = np.array([[20, -2, 4, 4], [0, 0, 0, 0]])
        mat_dy = np.array([[1, -3, 1, 0], [-21, 3, 1, 0]])
        nptest.assert_array_equal(mat_dx, dx)

        dx, dy = op.grad(mat2d, dim=2)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)

        # test with weights
        dx, dy = op.grad(mat2d, dim=2, wx=2, wy=0.5, wz=3, wt=2)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)

        # test without weight
        dx = op.grad(mat3d, dim=1)
        mat_dx = np.array([[[3, 3], [3, 3], [3, 3]],
                           [[0, 0], [0, 0], [0, 0]]])
        mat_dy = np.array([[[1, 1], [1, 1], [0, 0]],
                           [[1, 1], [1, 1], [0, 0]]])
        mat_dz = np.array([[[6, 0], [6, 0], [6, 0]],
                           [[6, 0], [6, 0], [6, 0]]])
        nptest.assert_array_equal(mat_dx, dx)
        dx, dy = op.grad(mat3d, dim=2)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        dx, dy, dz = op.grad(mat3d, dim=3)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)

        # test with weights
        dx, dy, dz = op.grad(mat3d, dim=3, wx=2, wy=0.5, wz=3, wt=2)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)

        # test without weight
        dx = op.grad(mat4d, dim=1)
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
        dx, dy = op.grad(mat4d, dim=2)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        dx, dy, dz = op.grad(mat4d, dim=3)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        dx, dy, dz, dt = op.grad(mat4d, dim4)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        nptest.assert_array_equal(mat_dt, dt)

        # test with weights
        dx, dy, dz, dt = op.grad(mat4d, dim=4, wx=2, wy=0.5, wz=3, wt=2)
        # Because oop numpy, can't *=
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        mat_dt_w = mat_dt * 2.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)
        nptest.assert_array_equal(mat_dt_w, dt)

        # test without weight
        dx = op.grad(mat5d, dim=1)
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
        dx, dy = op.grad(mat5d, dim=2)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        dx, dy, dz = op.grad(mat5d, dim=3)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        dx, dy, dz, dt = op.grad(mat5d, dim=4)
        nptest.assert_array_equal(mat_dx, dx)
        nptest.assert_array_equal(mat_dy, dy)
        nptest.assert_array_equal(mat_dz, dz)
        nptest.assert_array_equal(mat_dt, dt)

        # test with weights
        dx, dy, dz, dt = op.grad(mat5d, dim=4, wx=2, wy=0.5, wz=3, wt=2)
        mat_dx_w = mat_dx * 2.
        mat_dy_w = mat_dy * 0.5
        mat_dz_w = mat_dz * 3.
        mat_dt_w = mat_dt * 2.
        nptest.assert_array_equal(mat_dx_w, dx)
        nptest.assert_array_equal(mat_dy_w, dy)
        nptest.assert_array_equal(mat_dz_w, dz)
        nptest.assert_array_equal(mat_dt_w, dt)

    def test_div(self):
        # Divergence tests
        # test with 1dim matrices
        dx = np.array([1, 2, 3, 4, 5])

        # test without weights
        nptest.assert_array_equal(np.array([1, 1, 1, 1, -4]), op.div(dx))

        # test with weights
        weights = {'wx': 2, 'wy': 3, 'wz': 4, 'wt': 2}
        nptest.assert_array_equal(np.array([2, 2, 2, 2, -8]), op.div(dx, **weights))

        # test with 2dim matrices
        dx = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        dy = np.array([[13, 14, 15, 16],
                       [17, 18, 19, 20],
                       [21, 22, 23, 24]])

        # test without weights
        x_mat = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [-5, -6, -7, -8]])
        xy_mat = np.array([[14, 3, 4, -11],
                           [21, 5, 5, -15],
                           [16, -5, -6, -31]])
        nptest.assert_array_equal(x_mat, op.div(dx))
        nptest.assert_array_equal(xy_mat, op.div(dx, dy))

        # test with weights
        xy_mat_w = np.array([[41, 7, 9, -37],
                             [59, 11, 11, -49],
                             [53, -9, -11, -85]])
        nptest.assert_array_equal(xy_mat_w, op.div(dx, dy, **weights))

        # test with 3dim matrices (3x3x3)
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
        nptest.assert_array_equal(x_mat, op.div(dx))
        nptest.assert_array_equal(xy_mat,  op.div(dx, dy))
        nptest.assert_array_equal(xyz_mat, op.div(dx, dy, dz))
        # test with weights
        xyz_mat_w = np.array([[[9, 86, 55], [15, 61, -1], [12, 27, -66]],
                              [[34, 81, 20], [29, 45, -47], [15, 0, -123]],
                              [[41, 58, -33], [25, 11, -111], [0, -45, -198]]])
        nptest.assert_array_equal(xyz_mat_w, op.div(dx, dy, dz, **weights))

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
        nptest.assert_array_equal(x_mat, op.div(dx))
        nptest.assert_array_equal(xy_mat, op.div(dx, dy))
        nptest.assert_array_equal(xyz_mat, op.div(dx, dy, dz))
        nptest.assert_array_equal(xyzt_mat, op.div(dx, dy, dz, dt))
        # test with weights
        xyzt_mat_w = np.array([[[[11, 306, 439], [106, 275, 282], [93, 136, 17]],
                                [[19, 231, 281], [83, 169, 93], [39, -1, -203]],
                                [[18, 147, 114], [51, 54, -105], [-24, -147, -432]]],
                               [[[42, 277, 350], [107, 216, 163], [64, 47, -132]],
                                [[39, 191, 181], [73, 99, -37], [-1, -101, -363]],
                                [[27, 96, 3], [30, -27, -246], [-75, -258, -603]]],
                               [[[55, 230, 243], [90, 139, 26], [17, -60, -299]],
                                [[41, 133, 63], [45, 11, -185], [-59, -219, -541]],
                                [[18, 27, -126], [-9, -126, -405], [-144, -387, -792]]]])
        nptest.assert_array_equal(xyzt_mat_w, op.div(dx, dy, dz, dt, **weights))
