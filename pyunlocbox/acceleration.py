# -*- coding: utf-8 -*-

r"""
The :mod:`pyunlocbox.acceleration` module implements acceleration schemes for
use with the :mod:`pyunlocbox.solvers`. Pass a given acceleration object as an
argument to your chosen solver during its initialization so that the solver can
use it.

Interface
---------

The :class:`accel` base class defines a common interface to all acceleration
schemes:

.. autosummary::

    accel.pre
    accel.update_step
    accel.update_sol
    accel.post

Acceleration schemes
--------------------

Then, derived classes implement various common acceleration schemes.

.. autosummary::

    dummy
    backtracking
    fista
    fista_backtracking
    regularized_nonlinear

"""

import copy
import logging
import warnings

import numpy as np
from scipy.optimize.linesearch import line_search_armijo


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
        self._pre(functions, x0)

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

    def _post(self):
        raise NotImplementedError("Class user should define this method.")


class dummy(accel):
    r"""
    Dummy acceleration scheme which does nothing.

    Used by default in most of the solvers. It simply returns unaltered the
    step size and solution point it receives.

    """

    def _pre(self, functions, x0):
        pass

    def _update_step(self, solver, objective, niter):
        return solver.step

    def _update_sol(self, solver, objective, niter):
        return solver.sol

    def _post(self):
        pass


# -----------------------------------------------------------------------------
# Stepsize optimizers
# -----------------------------------------------------------------------------


class backtracking(dummy):
    r"""
    Backtracking based on a local quadratic approximation of the smooth
    part of the objective.

    Parameters
    ----------
    eta : float
        A number between 0 and 1 representing the ratio of the geometric
        sequence formed by successive step sizes. In other words, it
        establishes the relation `step_new = eta * step_old`.
        Default is 0.5.

    Notes
    -----
    This is the backtracking strategy used in the original FISTA paper,
    :cite:`beck2009FISTA`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l1(y=y, lambda_=1.0)
    >>> f2 = functions.norm_l2(y=y, lambda_=0.8)
    >>> accel = acceleration.backtracking()
    >>> solver = solvers.forward_backward(accel=accel, step=10)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-32, rtol=None)
    Solution found after 4 iterations:
        objective function f(sol) = 0.000000e+00
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 4.,  5.,  6.,  7.])

    """

    def __init__(self, eta=0.5, **kwargs):
        if (eta > 1) or (eta <= 0):
            raise ValueError("eta must be between 0 and 1.")
        self.eta = eta
        super(backtracking, self).__init__(**kwargs)

    def _update_step(self, solver, objective, niter):
        """
        Notes
        -----
        TODO: For now we're recomputing gradients in order to evaluate the
        backtracking criterion. In the future, it might be interesting to
        think of some design changes so that this function has access to the
        gradients directly.

        Since the call to `solver._algo()` modifies the internal state of the
        solver itself, we need to store the solver's property values before
        doing backtracking, and then restore the solver's state after
        backtracking is done. This takes more memory, but it's the only way to
        guarantee that backtracking is performing on a fixed solver state.
        """
        # Save current state of the solver
        properties = copy.deepcopy(vars(solver))
        logging.debug('(Begin) solver properties: {}'.format(properties))

        # Initialize some useful variables
        fn = 0
        grad = np.zeros(properties['sol'].shape)
        for f in solver.smooth_funs:
            fn += f.eval(properties['sol'])
            grad += f.grad(properties['sol'])
        step = properties['step']

        logging.debug('fn = {}'.format(fn))

        while True:
            # Run the solver with the current stepsize
            solver.step = step
            logging.debug('Current step: {}'.format(step))
            solver._algo()
            logging.debug(
                '(During) solver properties: {}'.format(vars(solver)))

            # Record results
            fp = np.sum([f.eval(solver.sol) for f in solver.smooth_funs])
            logging.debug('fp = {}'.format(fp))

            dot_prod = np.dot(solver.sol - properties['sol'], grad)
            logging.debug('dot_prod = {}'.format(dot_prod))

            norm_diff = np.sum((solver.sol - properties['sol'])**2)
            logging.debug('norm_diff = {}'.format(norm_diff))

            # Restore the previous state of the solver
            for key, val in properties.items():
                setattr(solver, key, copy.copy(val))
            logging.debug('(Reset) solver properties: {}'.format(vars(solver)))

            if (2. * step * (fp - fn - dot_prod) <= norm_diff):
                logging.debug('Break condition reached')
                break
            else:
                logging.debug('Decreasing step')
                step *= self.eta

        return step


# -----------------------------------------------------------------------------
# Solution point optimizers
# -----------------------------------------------------------------------------


class fista(dummy):
    r"""
    Acceleration scheme for forward-backward solvers.

    Notes
    -----
    This is the acceleration scheme proposed in the original FISTA paper,
    :cite:`beck2009FISTA`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers, acceleration
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

    def _pre(self, functions, x0):
        self.sol = np.array(x0, copy=True)

    def _update_sol(self, solver, objective, niter):
        self.t = 1. if (niter == 1) else self.t  # Restart variable t if needed
        t = (1. + np.sqrt(1. + 4. * self.t**2.)) / 2.
        y = solver.sol + ((self.t - 1) / t) * (solver.sol - self.sol)
        self.t = t
        self.sol[:] = solver.sol
        return y

    def _post(self):
        del self.sol


class regularized_nonlinear(dummy):
    r"""
    Regularized nonlinear acceleration (RNA) for gradient descent.

    Parameters
    ----------
    k : int, optional
        Number of points to keep in the buffer for computing the extrapolation.
        (Default is 10.)
    lambda_ : float or list of floats, optional
        Regularization parameter in the acceleration scheme. The user can pass
        a list of candidates, and the acceleration algorithm will pick the one
        that provides the best extrapolation.
        (Default is 1e-6.)
    adaptive : boolean, optional
        If adaptive = True and the user has not provided a list of
        regularization parameters, the acceleration algorithm will assemble a
        grid of possible regularization parameters based on the SVD of the
        Gram matrix of vectors of differences in the extrapolation buffer.
        If adaptive = False, the algorithm will simply try to use the value(s)
        given in `lambda_`.
        (Default is True.)
    dolinesearch : boolean, optional
        If dolinesearch = True, the acceleration scheme will try to return a
        point in the line segment between the current extrapolation and the
        previous one that provides a decrease in the value of the objective
        function.
        If dolinesearch = False, the algorithm simply returns the current
        extrapolation.
        (Default is True.)
    forcedecrease : boolean, optional
        If forcedecrese = True and we obtain a bad extrapolation, the
        algorithm returns the unchanged solution produced by the solver.
        If forcedecrease = False, the algorithm returns the new extrapolation
        no matter what.
        (Default is True.)

    Notes
    -----
    This is the acceleration scheme proposed in :cite:`scieur2016`.

    See also Damien Scieur's `repository <https://github.com/windows7lover
    /RegularizedNonlinearAcceleration>`_ for the Matlab version that inspired
    this implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> dim = 25;
    >>> np.random.seed(0)
    >>> xstar = np.random.rand(dim) # True solution
    >>> x0 = np.random.rand(dim)
    >>> x0 = xstar + 5.*(x0 - xstar) / np.linalg.norm(x0 - xstar)
    >>> A = np.random.rand(dim, dim)
    >>> step = 1/np.linalg.norm(np.dot(A.T, A))
    >>> f = functions.norm_l2(lambda_=0.5, A=A, y=np.dot(A, xstar))
    >>> fd = functions.dummy()
    >>> accel = acceleration.regularized_nonlinear(k=5)
    >>> solver = solvers.gradient_descent(step=step, accel=accel)
    >>> params = {'rtol':0, 'maxit':200, 'verbosity':'NONE'}
    >>> ret = solvers.solve([f, fd], x0, solver, **params)
    >>> pctdiff = 100*np.sum((xstar - ret['sol'])**2)/np.sum(xstar**2)
    >>> print('Difference: {0:.1f}%'.format(pctdiff))
    Difference: 1.3%

    """

    def __init__(self, k=10, lambda_=1e-6, adaptive=True, dolinesearch=True,
                 forcedecrease=True, **kwargs):
        self.k = k
        self.lambda_ = lambda_
        self.adaptive = adaptive
        self.dolinesearch = dolinesearch
        self.forcedecrease = forcedecrease

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, lambda_):
        try:
            self._lambda_ = [float(elem) for elem in lambda_]
        except TypeError:
            try:
                self._lambda_ = [float(lambda_)]
            except ValueError as err:
                print('lambda_ is not a number')
                raise err
        except ValueError as err:
            print('lambda_ is not a list of numbers')
            raise err

    def _pre(self, functions, x0):
        self.buffer = []
        self.functions = functions

    def _update_sol(self, solver, objective, niter):

        if (niter % (self.k + 1)) == 0:  # Extrapolate at each k iterations

            self.buffer.append(solver.sol)

            # (Normalized) matrix of differences
            U = np.diff(self.buffer, axis=0)
            UU = np.dot(U, U.T)
            UU /= np.linalg.norm(UU)

            # If no parameter grid was provided, assemble one.
            if self.adaptive and (len(self.lambda_) <= 1):
                svals = np.sort(np.abs(np.linalg.eigvals(UU)))
                svals = np.log(svals)
                svals = 0.5 * (svals[:-1] + svals[1:])
                self.lambda_ = np.concatenate(([0.], np.exp(svals)))

            # Grid search for the best parameter for the extrapolation
            fvals = []
            c = np.zeros((self.k,))
            extrap = np.zeros(np.shape(solver.sol))

            for lambda_ in self.lambda_:
                # Coefficients of the extrapolation
                c[:] = np.linalg.solve(UU + lambda_ * np.eye(self.k),
                                       np.ones(self.k))
                c[:] /= np.sum(c)

                extrap[:] = np.dot(np.asarray(self.buffer[:-1]).T, c)

                fvals.append(np.sum([f.eval(extrap) for f in self.functions]))

            if self.forcedecrease and (min(fvals) > np.sum(objective[-1])):
                # If we have bad extrapolations, keep solution as is
                extrap[:] = solver.sol
            else:
                # Return the best extrapolation from the grid search
                lambda_ = self.lambda_[fvals.index(min(fvals))]

                # We can afford to solve the linear system here again because
                # self.k is normally very small. Alternatively, we could have
                # kept track of the best extrapolations during the grid search,
                # but that would require at least double the memory, as we'd
                # have to store both the current extrapolation and the best
                # extrapolation.
                c[:] = np.linalg.solve(UU + lambda_ * np.eye(self.k),
                                       np.ones(self.k))
                c[:] /= np.sum(c)
                extrap[:] = np.dot(np.asarray(self.buffer[:-1]).T, c)

            # Improve proposal with line search
            if self.dolinesearch:
                # Objective evaluation functional
                def f(x):
                    return np.sum([f.eval(x) for f in self.functions])
                # Solution at previous extrapolation
                xk = self.buffer[0]
                # Search direction
                pk = extrap - xk
                # Objective value during the previous extrapolation
                old_fval = np.sum(objective[-self.k])

                a, fc, fa = line_search_armijo(f=f,
                                               xk=xk,
                                               pk=pk,
                                               gfk=-pk,
                                               old_fval=old_fval,
                                               c1=1e-4,
                                               alpha0=1.)

                # New point proposal
                if a is None:
                    warnings.warn('Line search failed to find good step size')
                else:
                    extrap[:] = xk + a * pk

            # Clear buffer and parameter grid for next extrapolation process
            self.buffer = []
            self.lambda_ = [] if self.adaptive else self.lambda_

            return extrap

        else:  # Gather points for future extrapolation
            self.buffer.append(copy.copy(solver.sol))
            return solver.sol

    def _post(self):
        del self.buffer, self.functions

# -----------------------------------------------------------------------------
# Mixed optimizers
# -----------------------------------------------------------------------------


class fista_backtracking(backtracking, fista):
    r"""
    Acceleration scheme with backtracking for forward-backward solvers.

    Notes
    -----
    This is the acceleration scheme and backtracking strategy proposed in the
    original FISTA paper, :cite:`beck2009FISTA`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers, acceleration
    >>> y = [4, 5, 6, 7]
    >>> x0 = np.zeros(len(y))
    >>> f1 = functions.norm_l2(y=y)
    >>> f2 = functions.dummy()
    >>> accel=acceleration.fista_backtracking()
    >>> solver = solvers.forward_backward(accel=accel, step=0.5)
    >>> ret = solvers.solve([f1, f2], x0, solver, atol=1e-5)
    Solution found after 15 iterations:
        objective function f(sol) = 4.957288e-07
        stopping criterion: ATOL
    >>> ret['sol']
    array([ 4.0002509 ,  5.00031362,  6.00037635,  7.00043907])

    """

    def __init__(self, eta=0.5, **kwargs):
        # I can do multiple inheritance here and avoid the deadly diamond of
        # death because the classes backtracking and fista modify different
        # methods of their parent class dummy. If that would not be the case, I
        # guess the best solution would be to inherit from accel and rewrite
        # the `_update_step()` and `_update_sol()` methods.

        backtracking.__init__(self, eta=eta, **kwargs)
        fista.__init__(self, **kwargs)
