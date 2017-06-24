# -*- coding: utf-8 -*-

r"""
This module implements acceleration schemes for use with the :class:`solver`
classes. Pass a given acceleration object as an argument to your chosen solver
during its initialization so that the solver can use it. The base class
:class:`acceleration` defines the interface of all acceleration objects. The
specialized acceleration objects inherit from it and implement the class
methods. The following acceleration schemes are included :

* :class:`dummy`: Dummy acceleration scheme. Does nothing.
* :class:`backtracking`: Backtracking line search.
* :class:`fista`: FISTA acceleration scheme.
* :class:`fista_backtracking`: FISTA with backtracking.

"""

import logging
import numpy as np


class accel(object):
    r"""
    Defines the acceleration scheme object interface.

    This class defines the interface of an acceleration scheme object intended
    to be passed to a solver inheriting from
    :class:`pyunlocbox.solvers.solver`. It is intended to be a base class for
    standard acceleration schemes which will implement the required methods.
    It can also be instantiated by user code and dynamically modified for
    rapid testing. This class also defines the generic attributes of all
    acceleration scheme objects.

    """

    def __init__(self):
        pass

    def pre(self, functions, x0):
        """
        Pre-processing specific to the acceleration scheme.

        Gets called when :func:`pyunlocbox.solvers.solve` starts running.
        """
        self.sol = np.array(x0, copy=True)
        self._pre(functions, self.sol)

    def _pre(self, functions, x0):
        raise NotImplementedError("Class user should define this method.")

    def update_step(self, solver, objective, niter):
        """
        Update the step size for the next iteration.

        Parameters
        ----------
        solver : pyunlocbox.solvers.solver
            Solver on which to act.
        objective : list of floats
            List of evaluations of the objective function since the beginning
            of the iterative process.
        niter : int
            Current iteration number.

        Returns
        -------
        float
            Updated step size.
        """
        return self._update_step(solver, objective, niter)

    def _update_step(self, solver, objective, niter):
        raise NotImplementedError("Class user should define this method.")

    def update_sol(self, solver, objective, niter):
        """
        Update the solution point for the next iteration.

        Parameters
        ----------
        solver : pyunlocbox.solvers.solver
            Solver on which to act.
        objective : list of floats
            List of evaluations of the objective function since the beginning
            of the iterative process.
        niter : int
            Current iteration number.

        Returns
        -------
        array_like
            Updated solution point.
        """
        return self._update_sol(solver, objective, niter)

    def _update_sol(self, solver, objective, niter):
        raise NotImplementedError("Class user should define this method.")

    def post(self):
        """
        Post-processing specific to the acceleration scheme.

        Mainly used to delete references added during initialization so that
        the garbage collector can free the memory. Gets called when
        :func:`pyunlocbox.solvers.solve` finishes running.
        """
        self._post()
        del self.sol

    def _post(self):
        raise NotImplementedError("Class user should define this method.")


class dummy(accel):
    r"""
    Dummy acceleration scheme.

    Used by default in most of the solvers. It simply returns unaltered the
    step size and solution point it receives.
    """

    def _pre(self, functions, x0):
        pass

    def _update_step(self, solver, objective, niter):
        return solver.step

    def _update_sol(self, solver, objective, niter):
        # Track the solution, but otherwise do nothing
        self.sol[:] = solver.sol
        return solver.sol

    def _post(self):
        pass


class backtracking(dummy):
    r"""
    Backtracking based on a local quadratic approximation of the objective.

    Parameters
    ----------
    eta : float
        A number between 0 and 1 representing the ratio of the geometric
        sequence formed by successive step sizes. In other words, it
        establishes the relation `step_new = eta * step_old`.
        Default is 0.5.

    Notes
    -----
    This is the backtracking strategy proposed in the original FISTA paper,
    :cite:`beck2009FISTA`.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> accel = acceleration.backtracking()
    >>> solver = solvers.forward_backward(accel=accel, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 12 iterations:
        objective function f(sol) = 7.510185e-06
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 3.99902344,  4.9987793 ,  5.99853516,  6.99829102])

    """

    def __init__(self, eta=0.5, **kwargs):
        if (eta > 1) or (eta <= 0):
            raise ValueError("eta must be between 0 and 1.")
        self.eta = eta
        super(backtracking, self).__init__(**kwargs)

    def _pre(self, functions, x0):
        self.smooth_funs = []  # Smooth functions.
        for f in functions:
            if 'GRAD' in f.cap(x0):
                self.smooth_funs.append(f)

    def _update_step(self, solver, objective, niter):
        """
        Notes
        -----
        TODO: For now we're recomputing gradients in order to evaluate the
        backtracking criterion. In the future, it might be interesting to
        think of some design changes so that this function has access to the
        gradients directly.
        """
        valn = np.sum(objective[-1])
        valp = 0
        grad = np.zeros(solver.sol.shape)
        for f in self.smooth_funs:
            valp += f.eval(solver.sol)
            grad += f.grad(self.sol)

        while (2 * solver.step *
               (valp - valn - np.dot(solver.sol - self.sol, grad)) >
                np.sum((solver.sol - self.sol)**2)):
            solver.step *= self.eta
            solver._algo()
            valp = np.sum([f.eval(solver.sol) for f in self.smooth_funs])

        return solver.step

    def _post(self):
        del self.smooth_funs


class fista(dummy):
    r"""
    Acceleration scheme for forward-backward solvers.

    Notes
    -----
    This is the acceleration scheme proposed in the original FISTA paper,
    :cite:`beck2009FISTA`.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> accel=acceleration.fista()
    >>> solver = solvers.forward_backward(accel=accel, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 15 iterations:
        objective function f(sol) = 4.957288e-07
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 4.0002509 ,  5.00031362,  6.00037635,  7.00043907])

    """

    def __init__(self, **kwargs):
        self.t = 1.
        super(fista, self).__init__(**kwargs)

    def _update_sol(self, solver, objective, niter):
        self.t = 1. if (niter == 1) else self.t  # Restart variable t if needed
        t = (1. + np.sqrt(1. + 4. * self.t**2.)) / 2.
        y = solver.sol + ((self.t - 1) / t) * (solver.sol - self.sol)
        self.t = t
        self.sol[:] = solver.sol
        return y


class fista_backtracking(backtracking, fista):
    r"""
    Acceleration scheme with backtracking for forward-backward solvers.

    Notes
    -----
    This is the acceleration scheme and backtracking strategy proposed in the
    original FISTA paper, :cite:`beck2009FISTA`.

    Examples
    --------
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> import numpy as np
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> accel=acceleration.fista_backtracking()
    >>> solver = solvers.forward_backward(accel=accel, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 13 iterations:
        objective function f(sol) = 9.518528e-08
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 3.99989006,  4.99986257,  5.99983509,  6.9998076 ])

    """

    def __init__(self, **kwargs):
        """
        I can do multiple inheritance here and avoid the deadly diamond of
        death because the classes backtracking and fista modify different
        methods of their parent class dummy. If that would not be the case, I
        guess the best solution would be to inherit from accel and rewrite the
        _update_step() and _update_sol() methods.
        """
        backtracking.__init__(self, **kwargs)
        fista.__init__(self, **kwargs)


class regularized_nonlinear(dummy):
    r"""
    Regularized nonlinear acceleration for gradient descent.

    Parameters
    ----------
    k : int, optional
        Number of points to keep in the buffer for computing the extrapolation.
        Default is 10

    Notes
    -----
    This is the acceleration scheme proposed in :cite:`scieur2016`

    Examples
    --------
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> import numpy as np
    >>> y = [4., 5., 6., 7.]
    >>> A = np.ones((4,4))
    >>> A[(1,2), 0] = 0
    >>> A[3, 1] = 0
    >>> A[(2,3), 2] = 0
    >>> A[(0,2), 3] = 0
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.norm_l2(A=A)
    >>> accel = acceleration.regularized_nonlinear()
    >>> solver = solvers.gradient_descent(step=0.5/(np.linalg.norm(A) + 1.))
    >>> ret = solvers.solve([f1, f2], x0, solver, rtol=1e-32)
    Solution found after 58 iterations:
        objective function f(sol) = 1.043654e+02
        stopping criterion: RTOL
    >>> ret['sol']
    array([ 0.28846153, 0.1153846, 1.23076922, 1.78846153])

    """

    def __init__(self, k=10, _lambda=1e-6, adaptive=True, line_search=True,
                 **kwargs):
        self.k = k
        self.adaptive = adaptive
        if self.adaptive:
            self._lambda = 1.
            self._lambda_min = 2.**(-self.k)
        else:
            self._lambda = _lambda
        self.line_search = True
        self.buffer = []

    def _pre(self, functions, x0):
        self.buffer.append(x0)
        self.functions = functions

    def _update_sol(self, solver, objective, niter):

        if (niter % self.k) == 0:  # Extrapolate at each k iterations

            logging.debug('Entering extrapolation snippet')

            self.buffer.append(solver.sol)
            logging.debug('buffer = {}'.format(self.buffer))

            # (Normalized) matrix of differences
            U = np.diff(self.buffer, axis=0)
            UU = np.dot(U, U.T)
            normUU = np.linalg.norm(UU)
            UU /= normUU

            logging.debug('U = {}'.format(U))
            logging.debug('UU = {}'.format(UU))

            # Coefficients of the extrapolation
            c = np.linalg.solve(UU + self._lambda * np.eye(self.k),
                                np.ones(self.k))
            c /= np.linalg.norm(c)

            logging.debug('c = {}'.format(c))

            extrap = np.dot(np.asarray(self.buffer[:-1]).T, c)

            logging.debug('extrap = {}'.format(extrap))

            if self.adaptive:  # Find the best regularization parameter

                logging.debug('Entering adaptative snippet')

                f_pre = -np.inf
                f_pos = np.sum([f.eval(extrap) for f in self.functions])

                logging.debug('f_pre = {}'.format(f_pre))
                logging.debug('f_pos = {}'.format(f_pos))

                while (f_pos > f_pre) and self._lambda >= self._lambda_min:
                    f_pre = f_pos

                    self._lambda /= 2.

                    logging.debug('_lambda = {}'.format(self._lambda))

                    c = np.linalg.solve(UU + self._lambda * np.eye(self.k),
                                        np.ones(self.k))
                    c /= np.linalg.norm(c)

                    logging.debug('c = {}'.format(c))

                    extrap = np.dot(np.asarray(self.buffer[:-1]).T, c)

                    logging.debug('extrap = {}'.format(extrap))

                    f_pos = np.sum([f.eval(extrap) for f in self.functions])

                    logging.debug('f_pre = {}'.format(f_pre))
                    logging.debug('f_pos = {}'.format(f_pos))

            if self.line_search:  # Improve proposal with line search

                logging.debug('Entering line search snippet')

                direct = extrap - solver.sol  # Direction of the line
                stepsize = 1.

                f_pre = np.sum([f.eval(solver.sol + stepsize * direct)
                                for f in self.functions])

                while True:
                    stepsize *= 2.

                    logging.debug('stepsize = {}'.format(stepsize))

                    f_pos = np.sum([f.eval(solver.sol + stepsize * direct)
                                    for f in self.functions])

                    logging.debug('f_pre = {}'.format(f_pre))
                    logging.debug('f_pos = {}'.format(f_pos))

                    if (f_pre > f_pos):
                        f_pre = f_pos
                    else:
                        stepsize /= 2.
                        break

                # New point proposal
                logging.debug('stepsize = {}'.format(stepsize))

                extrap = solver.sol + stepsize * direct

                logging.debug('extrap = {}'.format(extrap))

            # Clear buffer for the next extrapolation process
            self.buffer = []
            self.buffer.append(extrap)

            return extrap

        else:  # Gather points for future extrapolation
            logging.debug('Filling up buffer')
            self.buffer.append(solver.sol)
            return solver.sol
