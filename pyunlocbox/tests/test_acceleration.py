#!/usr/bin/env python

"""
Test suite for the acceleration module of the pyunlocbox package.

"""

import numpy as np
import numpy.testing as nptest
import pytest

from pyunlocbox import acceleration, functions, solvers


class TestAcceleration:

    def test_accel(self):
        """
        Test base acceleration scheme class

        """
        funs = [functions.dummy(), functions.dummy()]
        x0 = np.zeros((4,))
        a = acceleration.accel()
        s = solvers.forward_backward()
        o = [[1.0, 2.0], [0.0, 1.0]]
        n = 2

        with pytest.raises(NotImplementedError):
            a.pre(funs, x0)
        with pytest.raises(NotImplementedError):
            a.update_step(s, o, n)
        with pytest.raises(NotImplementedError):
            a.update_sol(s, o, n)
        with pytest.raises(NotImplementedError):
            a.post()

    def test_backtracking(self):
        """
        Test forward-backward splitting solver with backtracking, solving
        problems with L1-norm, L2-norm, and dummy functions.

        """
        # Test constructor sanity
        a = acceleration.backtracking()
        with pytest.raises(ValueError):
            a.__init__(2.0)
        with pytest.raises(ValueError):
            a.__init__(-2.0)

        y = [4.0, 5.0, 6.0, 7.0]
        accel = acceleration.backtracking()
        step = 10  # Make sure backtracking is called
        solver = solvers.forward_backward(accel=accel, step=step)
        param = {"solver": solver, "atol": 1e-32, "verbosity": "NONE"}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "ATOL"
        assert ret["niter"] == 13

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "ATOL"
        assert ret["niter"] <= 4  # win64 takes one iteration

    def test_forward_backward_fista(self):
        """
        Test forward-backward splitting solver with fista acceleration,
        solving problems with L1-norm, L2-norm, and dummy functions.

        """
        y = [4.0, 5.0, 6.0, 7.0]
        solver = solvers.forward_backward(accel=acceleration.fista())
        param = {"solver": solver, "rtol": 1e-6, "verbosity": "NONE"}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 60

        # Dummy prox and L2-norm gradient.
        f1 = functions.dummy()
        f2 = functions.norm_l2(y=y, lambda_=0.6)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 84

        # L2-norm prox and L2-norm gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l2(y=y)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y, rtol=1e-2)
        assert ret["crit"] == "MAXIT"
        assert ret["niter"] == 200

        # L1-norm prox and dummy gradient.
        f1 = functions.norm_l1(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 6

        # Dummy prox and L1-norm gradient. As L1-norm possesses no gradient,
        # the algorithm exchanges the functions : exact same solution.
        f1 = functions.dummy()
        f2 = functions.norm_l1(y=y)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 6

        # L1-norm prox and L1-norm gradient. L1-norm possesses no gradient.
        f1 = functions.norm_l1(y=y)
        f2 = functions.norm_l1(y=y)
        with pytest.raises(ValueError):
            solvers.solve([f1, f2], np.zeros(len(y)), **param)

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 10

    def test_forward_backward_fista_backtracking(self):
        """
        Test forward-backward splitting solver with fista acceleration and
        backtracking, solving problems with L1-norm, L2-norm, and dummy
        functions.

        """
        y = [4.0, 5.0, 6.0, 7.0]
        accel = acceleration.fista_backtracking()
        solver = solvers.forward_backward(accel=accel)
        param = {"solver": solver, "rtol": 1e-6, "verbosity": "NONE"}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 60

        # L1-norm prox and L2-norm gradient.
        f1 = functions.norm_l1(y=y, lambda_=1.0)
        f2 = functions.norm_l2(y=y, lambda_=0.8)
        ret = solvers.solve([f1, f2], np.zeros(len(y)), **param)
        nptest.assert_allclose(ret["sol"], y)
        assert ret["crit"] == "RTOL"
        assert ret["niter"] == 3

    def test_regularized_nonlinear(self):
        """
        Test gradient descent solver with regularized non-linear acceleration,
        solving problems with L2-norm functions.

        """
        dim = 25
        np.random.seed(0)
        x0 = np.random.rand(dim)
        xstar = np.random.rand(dim)
        x0 = xstar + 5.0 * (x0 - xstar) / np.linalg.norm(x0 - xstar)

        A = np.random.rand(dim, dim)
        step = 1 / np.linalg.norm(np.dot(A.T, A))

        accel = acceleration.regularized_nonlinear(k=5)
        solver = solvers.gradient_descent(step=step, accel=accel)
        param = {"solver": solver, "rtol": 0, "maxit": 200, "verbosity": "NONE"}

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(lambda_=0.5, A=A, y=np.dot(A, xstar))
        f2 = functions.dummy()
        ret = solvers.solve([f1, f2], x0, **param)
        pctdiff = 100 * np.sum((xstar - ret["sol"]) ** 2) / np.sum(xstar**2)
        nptest.assert_array_less(pctdiff, 1.91)

        # Sanity checks
        accel = acceleration.regularized_nonlinear()
        with pytest.raises(ValueError):
            accel.__init__(10, ["not", "good"])
        with pytest.raises(ValueError):
            accel.__init__(10, "nope")

    def test_acceleration_comparison(self):
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
        slvs = []
        slvs.append(solvers.forward_backward(accel=acceleration.dummy(), step=step))
        slvs.append(solvers.forward_backward(accel=acceleration.fista(), step=step))
        slvs.append(
            solvers.forward_backward(
                accel=acceleration.fista_backtracking(eta=0.999), step=step
            )
        )

        # Compare solutions.
        params = {"rtol": 1e-14, "verbosity": "NONE", "maxit": 1e4}
        niters = [2, 2, 6]
        for solver, niter in zip(slvs, niters):
            x0 = np.zeros(len(y))
            ret = solvers.solve([f1, f2], x0, solver, **params)
            nptest.assert_allclose(ret["sol"], sol)
            assert ret["niter"] == niter
