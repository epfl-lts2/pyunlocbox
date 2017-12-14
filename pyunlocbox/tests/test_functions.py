#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the functions module of the pyunlocbox package.

"""

import unittest
import inspect

import numpy as np
import numpy.testing as nptest

from pyunlocbox import functions


class TestCase(unittest.TestCase):

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
        self.assertEqual(len(f.cap(x)), 0)
        # Set up but never used:
        # f.eval = lambda x: x**2 - 5
        # f.grad = lambda x: 2 * x
        # f.prox = lambda x, T: x + T

        def assert_equivalent(param1, param2):
            x = [[7, 8, 9], [10, 324, -45], [-7, -.2, 5]]
            funcs = inspect.getmembers(functions, inspect.isclass)
            for f in funcs:
                if f[0] not in ['func', 'norm', 'proj']:
                    f1 = f[1](**param1)
                    f2 = f[1](**param2)
                    self.assertEqual(f1.eval(x), f2.eval(x))
                    nptest.assert_array_equal(f1.prox(x, 3), f2.prox(x, 3))
                    if 'GRAD' in f1.cap(x):
                        nptest.assert_array_equal(f1.grad(x), f2.grad(x))

        # Default parameters. Callable or matrices.
        assert_equivalent({'y': 3.2}, {'y': lambda: 3.2})
        assert_equivalent({'A': None}, {'A': np.identity(3)})
        A = np.array([[-4, 2, 5], [1, 3, -7], [2, -1, 0]])
        assert_equivalent({'A': A}, {'A': A, 'At': A.T})
        assert_equivalent({'A': lambda x: A.dot(x)}, {'A': A, 'At': A})

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
        self.assertEqual(f.eval([3, 4]), 3 * 5**2)
        self.assertEqual(f.eval(np.array([-3, 4])), 3 * 5**2)
        nptest.assert_allclose(f.grad([3, 4]), [18, 24])
        nptest.assert_allclose(f.grad([3, -4]), [18, -24])
        self.assertEqual(f.prox(0, 1), 0)
        self.assertEqual(f.prox(7, 1. / 6), 3.5)
        f = functions.norm_l2(lambda_=4)
        nptest.assert_allclose(f.prox([7, -22], .125), [3.5, -11])

        f = functions.norm_l2(lambda_=1, A=lambda x: 2 * x, At=lambda x: x / 2,
                              y=[8, 12])
        self.assertEqual(f.eval([4, 6]), 0)
        self.assertEqual(f.eval([5, -2]), 256 + 4)
        nptest.assert_allclose(f.grad([4, 6]), 0)
#        nptest.assert_allclose(f.grad([5, -2]), [8, -64])
        nptest.assert_allclose(f.prox([4, 6], 1), [4, 6])

        f = functions.norm_l2(lambda_=2, y=np.fft.fft([2, 4]) / np.sqrt(2),
                              A=lambda x: np.fft.fft(x) / np.sqrt(x.size),
                              At=lambda x: np.fft.ifft(x) * np.sqrt(x.size))
#        self.assertEqual(f.eval(np.fft.ifft([2, 4])*np.sqrt(2)), 0)
#        self.assertEqual(f.eval([3, 5]), 2*np.sqrt(25+81))
        nptest.assert_allclose(f.grad([2, 4]), 0)
#        nptest.assert_allclose(f.grad([3, 5]), [4*np.sqrt(5), 4*3])
        nptest.assert_allclose(f.prox([2, 4], 1), [2, 4])
        nptest.assert_allclose(f.prox([3, 5], 1), [2.2, 4.2])
        nptest.assert_allclose(f.prox([2.2, 4.2], 1), [2.04, 4.04])
        nptest.assert_allclose(f.prox([2.04, 4.04], 1), [2.008, 4.008])

        # Test prox for non-tight matrices A
        L = np.array([[8, 1, 10], [1, 9, 1], [3, 7, 5], [1, 4, 4]])
        f = functions.norm_l2(A=L, tight=False, y=np.array([1, 2, 3, 4]),
                              w=np.array([1, 1, 0.5, 0.75]))
        nptest.assert_allclose(f.eval([1, 1, 1]), 455.0625)
        nptest.assert_allclose(f.grad([1, 1, 1]),
                               [329.625, 262.500, 430.500], rtol=1e-3)
        nptest.assert_allclose(f.prox([1, 1, 1], 1),
                               [-0.887,  0.252,  0.798], rtol=1e-3)
        nptest.assert_allclose(f.prox([6, 7, 3], 1),
                               [-0.345,  0.298,  0.388], rtol=1e-3)
        nptest.assert_allclose(f.prox([10, 0, -5], 1),
                               [1.103,  0.319,  -0.732], rtol=1e-3)

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
        self.assertEqual(f.eval([-3, 4]), 21)
        nptest.assert_array_equal(f.prox(np.array([[1, -4], [5, -2]]), 1),
                                  [[0, -1], [2, 0]])

        f = functions.norm_l1(tight=False)
        x = np.ones((4,))
        T = 0.5
        self.assertRaises(NotImplementedError, f.prox, x, T)

    def test_norm_nuclear(self):
        """
        Test the norm_nuclear derived class.
        We test the two methods : eval and prox.
        First with default class properties, then custom ones.

        """
        f = functions.norm_nuclear(lambda_=3)
        self.assertEqual(f.eval(np.diag([10, 0])), 30)
        self.assertEqual(f.eval(np.diag(np.array([-10, 0]))), 30)
        self.assertEqual(f.eval([[-3]]), 9)
        nptest.assert_allclose(f.prox(np.array([[1, 1], [1, 1]]), 1. / 3),
                               [[.5, .5], [.5, .5]])

    def test_norm_tv(self):
        """
        Test the norm_tv derived class.
        We test the grad, div, eval and prox.

        """
        # Test Matrices initialization
        # test for a 1dim matrice (testing with a 5)
        # mat1d = np.arange(5) + 1
        # test for a 2dim matrice (testing with a 2x4)
        mat2d = np.array([[2, 3, 0, 1], [22, 1, 4, 5]])
        # test for a 3 dim matrice (testing with a 2x3x2)
        mat3d = np.arange(1, stop=13).reshape(2, 2, 3).transpose((1, 2, 0))
        # test for a 4dim matrice (2x3x2x2)
        # mat4d = np.arange(1, stop=25).reshape(2, 2, 2, 3)
        # mat4d = mat4d.transpose((2, 3, 1, 0))
        # test for a 5dim matrice (2x2x3x2x2)
        # mat5d = np.arange(1, stop=49).reshape(2, 2, 3, 2, 2)
        # mat5d = mat5d.transpose((3, 4, 2, 1, 0))

        # Test for evals
        def test_eval():

            # test with 2d matrices
            # test without weight
            f = functions.norm_tv(dim=1)
            xeval = 30
            nptest.assert_array_equal(xeval, f.eval(mat2d))
            f = functions.norm_tv(dim=2)
            xeval = np.array([56.753641295582440])
            nptest.assert_array_equal(xeval, f.eval(mat2d))

            # test with weights
            f = functions.norm_tv(dim=1, wx=3)
            xeval = np.sum(np.array([60, 6, 12, 12]))
            nptest.assert_array_equal(xeval, f.eval(mat2d))
            f = functions.norm_tv(dim=2, wx=0.5, wy=2)
            xeval = np.array([71.1092])
            nptest.assert_array_equal(
                xeval, np.around(f.eval(mat2d), decimals=4))

            # test with 3d matrices (2x3x2)
            # test without weight
            f = functions.norm_tv(dim=2)
            sol = np.sum(np.array([11.324555320336760, 11.324555320336760]))
            nptest.assert_array_equal(sol, f.eval(mat3d))
            f = functions.norm_tv(dim=3)
            xeval = np.array(49.762944279683104)
            nptest.assert_array_equal(xeval, f.eval(mat3d))

            # test with weights
            f = functions.norm_tv(dim=2, wx=2, wy=3)
            sol = np.sum(np.array([25.4164, 25.4164]))
            nptest.assert_array_equal(sol, np.around(f.eval(mat3d),
                                                     decimals=4))

            f = functions.norm_tv(dim=3, wx=2, wy=3, wz=.5)
            xeval = np.array([58.3068])
            nptest.assert_array_equal(
                xeval, np.around(f.eval(mat3d), decimals=4))

        # Test for prox
        def test_prox():

            # Test with 2d matrices
            # Test without weights
            f = functions.norm_tv(tol=10e-4, dim=1)
            gamma = 30
            sol = np.array([[12.003459453582762, 1.999654054641723,
                             2.000691890716554, 3.000691890716554],
                            [11.996540546417238, 2.000345945358277,
                             1.999308109283446, 2.999308109283446]])
            nptest.assert_array_equal(np.around(sol, decimals=5),
                                      np.around((f.prox(mat2d, gamma)),
                                                decimals=5))

            f = functions.norm_tv(tol=10e-4, dim=2)
            gamma = 1.5
            x2d = np.array([[2, 3, 0, 1], [22, 1, 4, 5], [2, 10, 7, 8]])
            sol = np.array([[3.44427, 2.87332, 2.51662, 2.45336],
                            [18.38207, 3.10251, 4.0028, 4.64074],
                            [4.50809, 6.44118, 6.38421, 6.25082]])
            nptest.assert_array_equal(np.around(sol, decimals=5),
                                      np.around((f.prox(x2d, gamma)),
                                                decimals=5))

            # Test with weights

            # Test with 3d matrices
            # Test without weights
            f = functions.norm_tv(tol=10e-4, dim=2)
            gamma = 42
            sol = np.array([[[3.50087, 9.50087],
                             [3.50000, 9.50000],
                             [3.49913, 9.49913]],
                            [[3.50087, 9.50087],
                             [3.50000, 9.50000],
                             [3.49913, 9.49913]]])
            nptest.assert_array_equal(sol, np.around(f.prox(mat3d, gamma),
                                                     decimals=5))

            f = functions.norm_tv(tol=10e-4, dim=3)
            gamma = 18
            sol = np.array([[[6.5, 6.5], [6.5, 6.5], [6.5, 6.5]],
                            [[6.5, 6.5], [6.5, 6.5], [6.5, 6.5]]])
            nptest.assert_array_equal(sol, np.around(f.prox(mat3d, gamma),
                                                     decimals=1))
            # Test with weights
            f = functions.norm_tv(tol=10e-10, dim=2, wx=5, wy=10)
            gamma = 3
            x3d = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
                            [[4, 13, 22], [5, 14, 23], [6, 15, 24]],
                            [[7, 16, 25], [8, 17, 26], [9, 18, 27]]])
            sol = np.array([[[5, 14, 23], [5, 14, 23], [5, 14, 23]],
                            [[5, 14, 23], [5, 14, 23], [5, 14, 23]],
                            [[5, 14, 23], [5, 14, 23], [5, 14, 23]]])
            nptest.assert_array_equal(sol, np.around(f.prox(x3d, gamma)))

            # Test with 4d matrices
            # Test without weights
            f = functions.norm_tv(tol=10e-4, dim=3)
            gamma = 10
            x4d = np.array([[[[1, 28, 55], [10, 37, 64], [19, 46, 73]],
                             [[2, 29, 56], [11, 38, 65], [20, 47, 74]],
                             [[3, 30, 57], [12, 39, 66], [21, 48, 75]]],
                            [[[4, 31, 58], [13, 40, 67], [22, 49, 76]],
                             [[5, 32, 59], [14, 41, 68], [23, 50, 77]],
                             [[6, 33, 60], [15, 42, 69], [24, 51, 78]]],
                            [[[7, 34, 61], [16, 43, 70], [25, 52, 79]],
                             [[8, 35, 62], [17, 44, 71], [26, 53, 80]],
                             [[9, 36, 63], [18, 45, 72], [27, 54, 81]]]])
            sol = np.array([[[[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]]],
                            [[[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]]],
                            [[[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]],
                             [[14, 41, 68], [14, 41, 68], [14, 41, 68]]]])
            sol = np.around(sol)
            nptest.assert_array_equal(sol, np.around(f.prox(x4d, gamma)))

            f = functions.norm_tv(tol=10e-4, dim=4)
            gamma = 15
            sol = np.array([[[[22, 34, 54], [26, 40, 54], [31, 44, 53]],
                             [[23, 35, 54], [27, 40, 54], [32, 44, 53]],
                             [[23, 35, 54], [27, 41, 54], [32, 45, 53]]],
                            [[[24, 36, 54], [28, 41, 54], [32, 45, 53]],
                             [[24, 36, 54], [28, 42, 53], [33, 45, 53]],
                             [[24, 37, 54], [29, 42, 53], [33, 46, 53]]],
                            [[[25, 38, 54], [29, 43, 53], [34, 46, 53]],
                             [[25, 38, 54], [30, 43, 53], [34, 46, 53]],
                             [[26, 39, 54], [30, 43, 53], [35, 47, 53]]]])
            sol = np.around(sol)
            nptest.assert_array_equal(sol, np.around(f.prox(x4d, gamma)))

        # Test with weights
        test_prox()
        test_eval()

    def test_proj_b2(self):
        """
        Test the projection on the L2-ball.
        ISTA and FISTA algorithms for tight and non-tight frames.

        """
        tol = 1e-7

        # Tight frame, radius 0 --> x == y.
        x = np.random.uniform(size=7) + 10
        y = np.random.uniform(size=7) - 10
        f = functions.proj_b2(y=y, epsilon=tol)
        nptest.assert_allclose(f.prox(x, 0), y, atol=tol)

        # Tight frame, random radius --> ||x-y||_2 = radius.
        radius = np.random.uniform()
        f = functions.proj_b2(y=y, epsilon=radius)
        nptest.assert_almost_equal(np.linalg.norm(f.prox(x, 0) - y), radius)

        # Always evaluate to zero.
        self.assertEqual(f.eval(x), 0)

        # Non-tight frame : compare FISTA and ISTA results.
        nx = 30
        ny = 15
        x = np.random.standard_normal(nx)
        y = np.random.standard_normal(ny)
        A = np.random.standard_normal((ny, nx))
        nu = np.linalg.norm(A, ord=2)**2
        f = functions.proj_b2(y=y, A=A, nu=nu, tight=False, method='FISTA',
                              epsilon=5, tol=tol / 10)
        sol_fista = f.prox(x, 0)
        f.method = 'ISTA'
        sol_ista = f.prox(x, 0)
        nptest.assert_allclose(sol_fista, sol_ista, rtol=1e-3)

        f.method = 'NOT_A_VALID_METHOD'
        self.assertRaises(ValueError, f.prox, x, 0)

    def test_independent_problems(self):

        # Parameters.
        N = 3   # independent problems.
        n = 25  # dimensions.

        # Generate some data.
        X = 7 - 10 * np.random.uniform(size=(n, N))
        step = 10 * np.random.uniform()

        # Test all available functions.
        funcs = inspect.getmembers(functions, inspect.isclass)
        for func in funcs:

            # Instanciate the class.
            if func[0] in ['norm_tv']:
                f = func[1](dim=1)  # Each column is one-dimensional.
            else:
                f = func[1]()

            # The combined objective function of the N problems is the sum of
            # each objective.
            if func[0] not in ['func', 'norm', 'norm_nuclear', 'proj']:
                res = 0
                for iN in range(N):
                    res += f.eval(X[:, iN])
                nptest.assert_array_almost_equal(res, f.eval(X))

            # Each column is the prox of one of the N problems.
            # TODO: norm_tv shoud pass this test. Why is there a difference ?
            if func[0] not in ['func', 'norm', 'norm_nuclear', 'norm_tv',
                               'proj']:
                res = np.zeros((n, N))
                for iN in range(N):
                    res[:, iN] = f.prox(X[:, iN], step)
                nptest.assert_array_almost_equal(res, f.prox(X, step))

            # Each column is the gradient of one of the N problems.
            if func[0] not in ['func', 'norm', 'norm_l1', 'norm_nuclear',
                               'norm_tv', 'proj', 'proj_b2']:
                res = np.zeros((n, N))
                for iN in range(N):
                    res[:, iN] = f.grad(X[:, iN])
                nptest.assert_array_almost_equal(res, f.grad(X))


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
