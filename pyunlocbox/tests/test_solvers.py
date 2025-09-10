#!/usr/bin/env python

"""
Test suite for the solvers module of the pyunlocbox package.

"""

import numpy as np
import numpy.testing as nptest
import pytest

from pyunlocbox import acceleration, functions, solvers


class TestSolvers:

    def test_solve(self):
        """
        Test some features of the solving function.

        """

        # We have to set a seed here for the random draw if we are required
        # below to assert that the number of iterations of the solvers are
        # equal to some specific values. Otherwise, we get trivial errors when
        # x0 is a little farther away from y in a given draw.
        rs = np.random.RandomState(42)

        y = 5 - 10 * rs.uniform(size=(15, 4))

        def x0():
            return np.zeros_like(y)

        nverb = {"verbosity": "NONE"}

        # Function verbosity.
        f = functions.dummy()
        assert f.verbosity == "NONE"
        f.verbosity = "LOW"
        solvers.solve([f], x0(), **nverb)
        assert f.verbosity == "LOW"

        # Input parameters.
        with pytest.raises(ValueError):
            solvers.solve([f], x0(), verbosity="??")

        # Addition of dummy function.
        with pytest.raises(ValueError):
            solvers.solve([], x0(), **nverb)
        solver = solvers.forward_backward()
        solvers.solve([f], x0(), solver, **nverb)
        # assert isinstance(solver.f1, functions.dummy)
        # assert isinstance(solver.f2, functions.dummy)

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
        with pytest.raises(ValueError):
            solvers.solve([f0, f0], x0(), **nverb)
        ret = solvers.solve([f0, f1], x0(), **nverb)
        assert ret["solver"] == "forward_backward"
        ret = solvers.solve([f1, f0], x0(), **nverb)
        assert ret["solver"] == "forward_backward"
        ret = solvers.solve([f1, f2], x0(), **nverb)
        assert ret["solver"] == "forward_backward"
        ret = solvers.solve([f2, f2], x0(), **nverb)
        assert ret["solver"] == "douglas_rachford"
        ret = solvers.solve([f1, f2, f0], x0(), **nverb)
        assert ret["solver"] == "generalized_forward_backward"

        # Stopping criteria.
        f = functions.norm_l2(y=y)
        tol = 1e-6
        r = solvers.solve([f], x0(), None, tol, None, None, None, None, "NONE")
        assert r["crit"] == "ATOL"
        assert np.sum(r["objective"][-1]) < tol
        assert r["niter"] == 9
        tol = 1e-8
        r = solvers.solve([f], x0(), None, None, tol, None, None, None, "NONE")
        assert r["crit"] == "DTOL"
        err = np.abs(np.sum(r["objective"][-1]) - np.sum(r["objective"][-2]))
        assert err < tol
        assert r["niter"] == 17
        tol = 0.1
        r = solvers.solve([f], x0(), None, None, None, tol, None, None, "NONE")
        assert r["crit"] == "RTOL"
        err = np.abs(np.sum(r["objective"][-1]) - np.sum(r["objective"][-2]))
        err /= np.sum(r["objective"][-1])
        assert err < tol
        assert r["niter"] == 13
        tol = 1e-4
        r = solvers.solve([f], x0(), None, None, None, None, tol, None, "NONE")
        assert r["crit"] == "XTOL"
        r2 = solvers.solve([f], x0(), maxit=r["niter"] - 1, **nverb)
        err = np.linalg.norm(r["sol"] - r2["sol"]) / np.sqrt(x0().size)
        assert err < tol
        assert r["niter"] == 14
        nit = 15
        r = solvers.solve([f], x0(), None, None, None, None, None, nit, "NONE")
        assert r["crit"] == "MAXIT"
        assert r["niter"] == nit

        # Return values.
        f = functions.norm_l2(y=y)
        ret = solvers.solve([f], x0(), **nverb)
        assert len(ret) == 6
        assert isinstance(ret["sol"], np.ndarray)
        assert isinstance(ret["solver"], str)
        assert isinstance(ret["crit"], str)
        assert isinstance(ret["niter"], int)
        assert isinstance(ret["time"], float)
        assert isinstance(ret["objective"], list)

    def test_solver(self):
        """
        Base solver class.

        """
        funs = [functions.dummy(), functions.dummy()]
        x0 = np.zeros((4,))
        s = solvers.solver()
        s.sol = x0
        with pytest.raises(ValueError):
            s.__init__(-1.0)
        with pytest.raises(NotImplementedError):
            s.pre(funs, x0)
        with pytest.raises(NotImplementedError):
            s._algo()
        with pytest.raises(NotImplementedError):
            s.post()

    def test_gradient_descent(self):
        """
        Test gradient descent solver with l2-norms in the objective.

        """
        y = [4.0, 5.0, 6.0, 7.0]
        A = np.array(
            [
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
        sol = np.array([0.28846154, 0.11538462, 1.23076923, 1.78846154])
        step = 0.5 / (np.linalg.norm(A) + 1.0)
        solver = solvers.gradient_descent(step=step)
        param = {"solver": solver, "rtol": 0, "verbosity": "NONE"}

        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l2(A=A)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], sol)
        assert ret["crit"] == "MAXIT"
        assert ret["niter"] == 200

    def test_forward_backward(self):
        """
        Test forward-backward splitting algorithm without acceleration, and
        with L1-norm, L2-norm, and dummy functions.

        """
        y = [4.0, 5.0, 6.0, 7.0]
        solver = solvers.forward_backward(accel=acceleration.dummy())
        param = {"solver": solver, "rtol": 1e-6, "verbosity": "NONE"}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 35

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 4

        # Sanity check
        f3 = functions.dummy()
        x0 = np.zeros((4,))
        with pytest.raises(ValueError):
            solver.pre([f1, f2, f3], x0)

    def test_douglas_rachford(self):
        """
        Test douglas-rachford solver with L1-norm, L2-norm and dummy functions.
        """
        y = [4, 5, 6, 7]
        solver = solvers.douglas_rachford()
        param = {"solver": solver, "verbosity": "NONE"}

        # L2-norm prox and dummy prox.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 35

        # L2-norm prox and L1-norm prox.
        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l1(y=y)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 4

        # Sanity checks
        x0 = np.zeros((4,))
        solver.lambda_ = 2.0
        with pytest.raises(ValueError):
            solver.pre([f1, f2], x0)
        solver.lambda_ = -2.0
        with pytest.raises(ValueError):
            solver.pre([f1, f2], x0)
        with pytest.raises(ValueError):
            solver.pre([f1, f2, f1], x0)

    def test_linearized_douglas_rachford(self):
        "Test linearized douglas-rachford solver with two L1-norms."
        x = [-4, 3, -1]
        y = [4, -9, -13, -4]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        max_step = 0.5 / (1 + np.linalg.norm(L, 2))
        solver = solvers.douglas_rachford(step=max_step, A=L)

        # Two L1-norm prox.
        x0 = np.zeros(3)
        f1 = functions.norm_l1()
        f2 = functions.norm_l1(y=y)
        solver = solvers.douglas_rachford(step=max_step * 50, A=L)
        ret = solvers.solve(
            [f1, f2], x0, solver, atol=1e-1, maxit=1000, rtol=1e-5, verbosity="NONE"
        )
        nptest.assert_allclose(ret["sol"], x, rtol=1e-2)

        # Sanity checks
        with pytest.raises(ValueError):
            solver.pre([f1], x0)
        with pytest.raises(ValueError):
            solver.pre([f2, f1], x0)

    def test_generalized_forward_backward(self):
        """
        Test the generalized forward-backward algorithm.

        """
        y = [4, 5, 6, 7]
        L = 4  # Gradient of the smooth function is Lipschitz continuous.
        solver = solvers.generalized_forward_backward(step=0.9 / L, lambda_=0.8)
        params = {"solver": solver, "verbosity": "NONE"}

        # Functions.
        f1 = functions.norm_l1(y=y, lambda_=0.7)  # Non-smooth.
        f2 = functions.norm_l2(y=y, lambda_=L / 2.0)  # Smooth.

        # Solve with 1 smooth and 1 non-smooth.
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["niter"] == 25

        # Solve with 1 smooth.
        ret = solvers.solve([f1], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["niter"] == 77

        # Solve with 1 non-smooth.
        ret = solvers.solve([f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["niter"] == 18

        # Solve with 1 smooth and 2 non-smooth.
        ret = solvers.solve([f1, f2, f2], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["niter"] == 26

        # Solve with 2 smooth and 2 non-smooth.
        ret = solvers.solve([f2, f1, f2, f1], np.zeros(len(y)), **params)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["niter"] == 25

        # Sanity checks
        x0 = np.zeros((4,))
        solver.lambda_ = 2.0
        with pytest.raises(ValueError):
            solver.pre([f1, f2], x0)
        solver.lambda_ = -2.0
        with pytest.raises(ValueError):
            solver.pre([f1, f2], x0)
        f1 = functions.func()
        f2 = functions.func()
        f3 = functions.func()
        solver.lambda_ = 1.0
        with pytest.raises(ValueError):
            solver.pre([f1, f2, f3], x0)

    def test_mlfbf(self):
        """
        Test the MLFBF solver with arbitrarily selected functions.

        """
        x = [1.0, 1.0, 1.0]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        max_step = 1 / (1 + np.linalg.norm(L, 2))
        solver = solvers.mlfbf(L=L, step=max_step / 2.0)
        params = {"solver": solver, "verbosity": "NONE"}

        def x0():
            return np.zeros(len(x))

        # L2-norm prox and dummy prox.
        f = functions.dummy()
        f._prox = lambda x, T: np.maximum(np.zeros(len(x)), x)
        g = functions.norm_l2(lambda_=0.5)
        h = functions.norm_l2(y=np.array([294, 390, 361]), lambda_=0.5)
        ret = solvers.solve([f, g, h], x0(), maxit=1000, rtol=0, **params)
        nptest.assert_allclose(ret["sol"], x, rtol=1e-5)

        # Same test, but with callable L
        solver = solvers.mlfbf(
            L=lambda x: np.dot(L, x),
            Lt=lambda y: np.dot(L.T, y),
            d0=np.dot(L, x0()),
            step=max_step / 2.0,
        )
        ret = solvers.solve([f, g, h], x0(), maxit=1000, rtol=0, **params)
        nptest.assert_allclose(ret["sol"], x, rtol=1e-5)

        # Sanity check
        with pytest.raises(ValueError):
            solver.pre([f, g], x0())

        # Make a second test where the solution is calculated by hand
        n = 10
        y = np.random.rand(n) * 2
        z = np.random.rand(n)
        c = 1

        delta = (y - z - c) ** 2 + 4 * (1 + y * z - z * c)
        sol = 0.5 * ((y - z - c) + np.sqrt(delta))

        class mlog(functions.func):
            def __init__(self, z):
                super().__init__()
                self.z = z

            def _eval(self, x):
                return -np.sum(np.log(x + self.z))

            def _prox(self, x, T):
                delta = (x - self.z) ** 2 + 4 * (T + x * self.z)
                sol = 0.5 * (x - self.z + np.sqrt(delta))
                return sol

        f = functions.norm_l1(lambda_=c)
        g = mlog(z=z)
        h = functions.norm_l2(lambda_=0.5, y=y)

        mu = 1 + 1
        step = 1 / mu / 2

        solver = solvers.mlfbf(step=step)
        ret = solvers.solve(
            [f, g, h], y.copy(), solver, maxit=200, rtol=0, verbosity="NONE"
        )

        nptest.assert_allclose(ret["sol"], sol, atol=1e-10)

        # Make a final test where the function g can not be evaluate
        # on the primal variables
        y = np.random.rand(3)
        y_2 = L.dot(y)
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        x0 = np.zeros(len(y))
        f = functions.norm_l1(y=y)
        g = functions.norm_l2(lambda_=0.5, y=y_2)
        h = functions.norm_l2(y=y, lambda_=0.5)
        max_step = 1 / (1 + np.linalg.norm(L, 2))
        solver = solvers.mlfbf(L=L, step=max_step / 2.0)
        ret = solvers.solve([f, g, h], x0, solver, maxit=1000, rtol=0, verbosity="NONE")
        np.testing.assert_allclose(ret["sol"], y)

    def test_projection_based(self):
        """
        Test the projection-based solver with arbitrarily selected functions.

        """
        x = [0, 0, 0]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        solver = solvers.projection_based(L=L, step=1.0)
        params = {"solver": solver, "verbosity": "NONE"}

        # L1-norm prox and dummy prox.
        f = functions.norm_l1(y=np.array([294, 390, 361]))
        g = functions.norm_l1()
        ret = solvers.solve(
            [f, g],
            np.array([500, 1000, -400]),
            maxit=1000,
            rtol=None,
            xtol=0.1,
            **params,
        )
        nptest.assert_allclose(ret["sol"], x, rtol=1e-5)

        # Sanity checks
        def x0():
            return np.zeros(len(x))

        with pytest.raises(ValueError):
            solver.pre([f], x0())
        solver.lambda_ = 3.0
        with pytest.raises(ValueError):
            solver.pre([f, g], x0())
        solver.lambda_ = -3.0
        with pytest.raises(ValueError):
            solver.pre([f, g], x0())

    def test_chambolle_pock(self):
        """
        Test the Chambolle-Pock algorithm.

        """
        x = [-4, 3, -1]
        L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
        max_step = 0.5 / (1 + np.linalg.norm(L, 2))
        solver = solvers.chambolle_pock(
            L=L, sigma=max_step, theta=max_step, tau=max_step
        )

        # Two L1-norm prox.
        y = np.array([4, -9, -13, -4])
        F = functions.norm_l1(y=y)
        G = functions.norm_l1()
        x0 = np.array([0, 0, 0])
        ret = solvers.solve(
            [G, F], x0, solver, maxit=1000, rtol=None, xtol=None, verbosity="NONE"
        )
        nptest.assert_allclose(ret["sol"], x, rtol=1e-5)

        # Sanity checks
        with pytest.raises(ValueError):
            solver.pre([F], x0)
        solver.sigma = -1.0
        with pytest.raises(ValueError):
            solver.pre([G, F], x0)

    def test_solver_comparison(self):
        """
        Test that all solvers return the same and correct solution.

        """

        # Convex functions.
        y = [1, 0, 0.1, 8, -6.5, 0.2, 0.004, 0.01]
        sol = [0.75, 0, 0, 7.75, -6.25, 0, 0, 0]
        w1, w2 = 0.8, 0.4
        f1 = functions.norm_l2(y=y, lambda_=w1 / 2.0)  # Smooth.
        f2 = functions.norm_l1(lambda_=w2 / 2.0)  # Non-smooth.

        # Solvers.
        L = w1  # Lipschitz continuous gradient.
        step = 1.0 / L
        lambda_ = 0.5
        params = {"step": step, "lambda_": lambda_}
        slvs = []
        slvs.append(solvers.forward_backward(accel=acceleration.dummy(), step=step))
        slvs.append(solvers.douglas_rachford(**params))
        slvs.append(solvers.generalized_forward_backward(**params))

        # Compare solutions.
        params = {"rtol": 1e-14, "verbosity": "NONE", "maxit": 1e4}
        niters = [2, 61, 26]
        for solver, niter in zip(slvs, niters):
            x0 = np.zeros(len(y))
            ret = solvers.solve([f1, f2], x0, solver, **params)
            nptest.assert_allclose(ret["sol"], sol, atol=1e-12)
            assert ret["niter"] == niter
            # The initial value not was modified.
            np.testing.assert_array_equal(np.zeros(len(y)), x0)
            ret = solvers.solve([f1, f2], x0, solver, inplace=True, **params)
            # The initial value was modified.
            assert ret["sol"] is x0

    def test_primal_dual_solver_comparison(self):
        """
        Test that all primal-dual solvers return the same and correct solution.

        I had to create this separate function because the primal-dual solvers
        were too slow for the problem above.

        """

        # Convex functions.
        y = np.random.randn(3)
        L = np.random.randn(4, 3)

        sol = y
        y2 = L.dot(y)
        f1 = functions.norm_l1(y=y)
        f2 = functions.norm_l2(y=y2)
        f3 = functions.dummy()

        # Solvers.
        step = 0.5 / (1 + np.linalg.norm(L, 2))
        slvs = []
        slvs.append(solvers.mlfbf(step=step, L=L))
        slvs.append(solvers.projection_based(step=step, L=L))
        slvs.append(
            solvers.chambolle_pock(step=step, theta=step, sigma=step, tau=step, L=L)
        )

        # Compare solutions.
        niter = 1000
        params = {"rtol": 0, "verbosity": "NONE", "maxit": niter}
        for solver in slvs:
            x0 = np.zeros(len(y))

            if type(solver) is solvers.mlfbf:
                ret = solvers.solve([f1, f2, f3], x0, solver, **params)
            else:
                ret = solvers.solve([f1, f2], x0, solver, **params)
            nptest.assert_allclose(ret["sol"], sol)
            assert ret["niter"] == niter
            # The initial value was not modified.
            nptest.assert_array_equal(x0, np.zeros(len(y)))

            if type(solver) is solvers.mlfbf:
                ret = solvers.solve([f1, f2, f3], x0, solver, inplace=True, **params)
            else:
                ret = solvers.solve([f1, f2], x0, solver, inplace=True, **params)
            # The initial value was modified.
            assert ret["sol"] is x0
            nptest.assert_allclose(ret["sol"], sol)
