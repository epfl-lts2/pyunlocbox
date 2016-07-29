#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the solvers module of the pyunlocbox package.
"""

import unittest
import numpy as np
import numpy.testing as nptest
from pyunlocbox import functions, solvers


class FunctionsTestCase(unittest.TestCase):

    def test_solve(self):
        """
        Test some features of the solving function.
        """
        y = 5 - 10 * np.random.uniform(size=(15, 4))

        def x0(): return np.zeros(y.shape)
        nverb = {'verbosity': 'NONE'}

        # Function verbosity.
        f = functions.dummy()
        self.assertEqual(f.verbosity, 'NONE')
        f.verbosity = 'LOW'
        solvers.solve([f], x0(), **nverb)
        self.assertEqual(f.verbosity, 'LOW')

        # Input parameters.
        self.assertRaises(ValueError, solvers.solve, [f], x0(), verbosity='??')

        # Addition of dummy function.
        self.assertRaises(ValueError, solvers.solve, [], x0(), **nverb)
        solver = solvers.forward_backward()
        solvers.solve([f], x0(), solver, **nverb)
        # self.assertIsInstance(solver.f1, functions.dummy)
        # self.assertIsInstance(solver.f2, functions.dummy)

        # Automatic solver selection.
        f0 = functions.func()
        f0._eval = lambda x: 0
        f0._grad = lambda x: x
        f1 = functions.func()
        f1._eval = lambda x: 0
        f1._grad = lambda x: x
        f1._prox = lambda x, T: x
        f2 = functions.func()
        f2._eval = lambda x: 0
        f2._prox = lambda x, T: x
        self.assertRaises(ValueError, solvers.solve, [f0, f0], x0(), **nverb)
        ret = solvers.solve([f0, f1], x0(), **nverb)
        self.assertEqual(ret['solver'], 'forward_backward')
        ret = solvers.solve([f1, f0], x0(), **nverb)
        self.assertEqual(ret['solver'], 'forward_backward')
        ret = solvers.solve([f1, f2], x0(), **nverb)
        self.assertEqual(ret['solver'], 'forward_backward')
        ret = solvers.solve([f2, f2], x0(), **nverb)
        self.assertEqual(ret['solver'], 'douglas_rachford')
        ret = solvers.solve([f1, f2, f0], x0(), **nverb)
        self.assertEqual(ret['solver'], 'generalized_forward_backward')

        # Stopping criteria.
        f = functions.norm_l2(y=y)
        tol = 1e-6
        r = solvers.solve([f], x0(), None, tol, None, None, None, None, 'NONE')
        self.assertEqual(r['crit'], 'ATOL')
        self.assertLess(np.sum(r['objective'][-1]), tol)
        self.assertEqual(r['niter'], 10)
        tol = 1e-8
        r = solvers.solve([f], x0(), None, None, tol, None, None, None, 'NONE')
        self.assertEqual(r['crit'], 'DTOL')
        err = np.abs(np.sum(r['objective'][-1]) - np.sum(r['objective'][-2]))
        self.assertLess(err, tol)
        self.assertEqual(r['niter'], 14)
        tol = .1
        r = solvers.solve([f], x0(), None, None, None, tol, None, None, 'NONE')
        self.assertEqual(r['crit'], 'RTOL')
        err = np.abs(np.sum(r['objective'][-1]) - np.sum(r['objective'][-2]))
        err /= np.sum(r['objective'][-1])
        self.assertLess(err, tol)
        self.assertEqual(r['niter'], 14)
        tol = 1e-4
        r = solvers.solve([f], x0(), None, None, None, None, tol, None, 'NONE')
        self.assertEqual(r['crit'], 'XTOL')
        r2 = solvers.solve([f], x0(), maxit=r['niter'] - 1, **nverb)
        err = np.linalg.norm(r['sol'] - r2['sol']) / np.sqrt(x0().size)
        self.assertLess(err, tol)
        self.assertEqual(r['niter'], 12)
        nit = 15
        r = solvers.solve([f], x0(), None, None, None, None, None, nit, 'NONE')
        self.assertEqual(r['crit'], 'MAXIT')
        self.assertEqual(r['niter'], nit)

        # Return values.
        f = functions.norm_l2(y=y)
        ret = solvers.solve([f], x0(), **nverb)
        self.assertEqual(len(ret), 6)
        self.assertIsInstance(ret['sol'], np.ndarray)
        self.assertIsInstance(ret['solver'], str)
        self.assertIsInstance(ret['crit'], str)
        self.assertIsInstance(ret['niter'], int)
        self.assertIsInstance(ret['time'], float)
        self.assertIsInstance(ret['objective'], list)

    def test_forward_backward_fista(self):
        """
        Test FISTA algorithm of forward-backward solver with L1-norm, L2-norm
        and dummy functions.
        """
        y = [4, 5, 6, 7]
        solver = solvers.forward_backward(method='FISTA')
        param = {'solver': solver, 'atol': 1e-5, 'verbosity': 'NONE'}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        sol = [3.99996922, 4.99996153, 5.99995383, 6.99994614]
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 10)

        # Dummy prox and L2-norm gradient.
        f1 = functions.dummy()
        f2 = functions.norm_l2(y=y, lambda_=0.6)
        sol = [3.99867319, 4.99834148, 5.99800978, 6.99767808]
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 10)

        # L2-norm prox and L2-norm gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l2(y=y)
        sol = [3.99904855, 4.99881069, 5.99857282, 6.99833496]
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'MAXIT')

        # L1-norm prox and dummy gradient.
        f1 = functions.norm_l1(y=y)
        f2 = functions.dummy()
        sol = y
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 6)

        # Dummy prox and L1-norm gradient. As L1-norm possesses no gradient,
        # the algorithm exchanges the functions : exact same solution.
        f1 = functions.dummy()
        f2 = functions.norm_l1(y=y)
        sol = y
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 6)

        # L1-norm prox and L1-norm gradient. L1-norm possesses no gradient.
        f1 = functions.norm_l1(y=y)
        f2 = functions.norm_l1(y=y)
        self.assertRaises(ValueError, solvers.solve,
                          [f1, f2], np.zeros(len(y)), **param)

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        sol = y
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 4)

    def test_forward_backward_ista(self):
        """
        Test ISTA algorithm of forward-backward solver with L1-norm, L2-norm
        and dummy functions. Test the effect of step and lambda parameters.
        """
        y = [4, 5, 6, 7]
        # Smaller step size and update rate --> slower convergence.
        solver = solvers.forward_backward(method='ISTA', step=.8, lambda_=.5)
        param = {'solver': solver, 'atol': 1e-5, 'verbosity': 'NONE'}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        sol = [3.99915094, 4.99893867, 5.9987264, 6.99851414]
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 23)

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        sol = [3.99999825, 4.9999979, 5.99999756, 6.99999723]
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ATOL')
        self.assertEqual(ret['niter'], 21)

    def test_douglas_rachford(self):
        """
        Test douglas-rachford solver with L1-norm, L2-norm and dummy functions.
        """
        y = [4, 5, 6, 7]
        solver = solvers.douglas_rachford()
        param = {'solver': solver, 'verbosity': 'NONE'}

        # L2-norm prox and dummy prox.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['crit'], 'RTOL')
        self.assertEqual(ret['niter'], 35)

        # L2-norm prox and L1-norm prox.
        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l1(y=y)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['crit'], 'RTOL')
        self.assertEqual(ret['niter'], 4)

    def test_generalized_forward_backward(self):
        """
        Test the generalized forward-backward algorithm.
        """
        y = [4, 5, 6, 7]
        L = 4  # Gradient of the smooth function is Lipschitz continuous.
        solver = solvers.generalized_forward_backward(step=.9 / L, lambda_=.8)
        params = {'solver': solver, 'verbosity': 'NONE'}

        # Functions.
        f1 = functions.norm_l1(y=y, lambda_=.7)    # Non-smooth.
        f2 = functions.norm_l2(y=y, lambda_=L / 2.)  # Smooth.

        # Solve with 1 smooth and 1 non-smooth.
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['niter'], 25)

        # Solve with 1 smooth.
        ret = solvers.solve([f1], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['niter'], 77)

        # Solve with 1 non-smooth.
        ret = solvers.solve([f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['niter'], 18)

        # Solve with 1 smooth and 2 non-smooth.
        ret = solvers.solve([f1, f2, f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['niter'], 26)

        # Solve with 2 smooth and 2 non-smooth.
        ret = solvers.solve([f2, f1, f2, f1], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret['sol'], y)
        self.assertEqual(ret['niter'], 25)

    def test_mlfbf(self):
        """
        Test the MLFBF solver with arbitrarily selected functions.
        """
        x = [1., 1., 1.]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        max_step = 1 / (1 + np.linalg.norm(L, 2))
        solver = solvers.mlfbf(L=L, step=max_step / 2.)
        params = {'solver': solver, 'verbosity': 'NONE'}

        # L2-norm prox and dummy prox.
        f = functions.dummy()
        f._prox = lambda x, T: np.maximum(np.zeros(len(x)), x)
        g = functions.norm_l2(lambda_=0.5)
        h = functions.norm_l2(y=np.array([294, 390, 361]), lambda_=0.5)
        ret = solvers.solve([f, g, h], np.zeros(len(x)),
                            maxit=1000, rtol=0, **params)
        nptest.assert_allclose(ret['sol'], x, rtol=1e-5)

    def test_projection_based(self):
        """
        Test the projection-based solver with arbitrarily selected functions.
        """
        x = [0, 0, 0]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        solver = solvers.projection_based(L=L, step=1.)
        params = {'solver': solver, 'verbosity': 'NONE'}

        # L1-norm prox and dummy prox.
        f = functions.norm_l1(y=np.array([294, 390, 361]))
        g = functions.norm_l1()
        ret = solvers.solve([f, g], np.array([500, 1000, -400]),
                            maxit=1000, rtol=None, xtol=0.1, **params)
        nptest.assert_allclose(ret['sol'], x, rtol=1e-5)

    def test_solver_comparison(self):
        """
        Test that all solvers return the same and correct solution.
        """

        # Convex functions.
        y = [1, 0, 0.1, 8, -6.5, 0.2, 0.004, 0.01]
        sol = [0.75, 0, 0, 7.75, -6.25, 0, 0, 0]
        w1, w2 = .8, .4
        f1 = functions.norm_l2(y=y, lambda_=w1/2.)  # Smooth.
        f2 = functions.norm_l1(lambda_=w2/2.)       # Non-smooth.
        # f3 = functions.proj_b2(epsilon=0.6)         # Non-smooth.

        # Solvers.
        L = w1  # Lipschitz continuous gradient.
        params = {'step': 1./L, 'lambda_': 0.5}
        slvs = []
        slvs.append(solvers.forward_backward(method='ISTA', **params))
        slvs.append(solvers.forward_backward(method='FISTA', **params))
        slvs.append(solvers.douglas_rachford(**params))
        slvs.append(solvers.generalized_forward_backward(**params))
        # slvs.append(solvers.mlfbf(step=.7/L))  # TODO
        slvs.append(solvers.projection_based(step=1e-3))

        # Compare solutions.
        params = {'rtol': 1e-14, 'verbosity': 'NONE', 'maxit': 1e4}
        niters = [26, 2, 61, 26, 1e4]
        for solver, niter in zip(slvs, niters):
            x0 = np.zeros(len(y))
            ret = solvers.solve([f1, f2], x0, solver, **params)
            # These primal-dual solvers are much slower.
            if type(solver) is solvers.projection_based:
                nptest.assert_allclose(ret['sol'], sol, atol=0.2)
            else:
                nptest.assert_allclose(ret['sol'], sol)
            self.assertEqual(ret['niter'], niter)
            self.assertIs(ret['sol'], x0)  # The initial value was modified.


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)
