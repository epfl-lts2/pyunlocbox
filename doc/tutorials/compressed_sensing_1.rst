=========================================
Compressed sensing using forward-backward
=========================================


>>> # Problem creation.
>>> import numpy as np
>>> tau = 1      # Regularization parameter.
>>> N = 5000     # Signal size.
>>> K = 100      # Sparsity level.
>>> R = max(4, np.ceil(np.log(N)))
>>> M = K * R    # Number of measurements.

>>> print('Number of measurements : %d' % (M,))
Number of measurements : 900
>>> print('Compression ratio : %3.2f' % (N/M,))
Compression ratio : 5.56

>>> # Measurements matrix.
>>> # Reproducible results.
>>> np.random.seed(1)
>>> A = np.random.standard_normal((M, N))

>>> # Create a K sparse signal.
>>> x = np.zeros(N)
>>> I = np.random.permutation(N)
>>> x[I[0:K]] = np.random.standard_normal(K)
>>> x = x / np.linalg.norm(x)

>>> # Measurements.
>>> y = np.dot(A, x)

You can also pass a verbosity of ``'low'`` or ``'high'`` if you want the solver
to output some informations.

>>> # Set the two convex function objects.
>>> from pyunlocbox import functions
>>> f1 = functions.norm_l1(lambda_=tau, verbosity='none')
>>> f2 = functions.norm_l2(y=y, A=A, verbosity='none')

>>> # Alternative 1 (same results) : pass operators instead of matrices.
>>> A_ = lambda x: np.dot(A, x)
>>> At_ = lambda x: np.dot(np.transpose(A), x)
>>> f3 = functions.norm_l2(y=y, A=A_, At=At_, verbosity='none')
>>> assert f3

>>> # Alternative 2 (same results) : manual definition of the L2 norm.
>>> f4 = functions.func()
>>> f4.grad = lambda x: 2.0 * np.dot(np.transpose(A), np.dot(A, x) - y)
>>> f4.eval = lambda x: np.linalg.norm(np.dot(A, x) - y)**2

>>> # Set the solver object. Step size : beta = 2*norm(A)^2.
>>> gamma = 0.5 / np.linalg.norm(A, ord=2)**2
>>> from pyunlocbox import solvers
>>> solver = solvers.forward_backward(method='FISTA', gamma=gamma)

>>> # Solve the problem.
>>> x0 = np.zeros(N)
>>> ret = solvers.solve([f1, f2], x0, solver, relTol=1e-4, maxIter=300,
...                     verbosity='low')
Solution found in 176 iterations :
    objective function f(sol) = 8.221302e+00
    last relative objective improvement : 8.363264e-05
    stopping criterion : REL_TOL

>>> import matplotlib.pyplot as plt

>>> # Display the results.
>>> fig = plt.figure()
>>> _ = plt.plot(x, 'o', label='Original')
>>> _ = plt.plot(ret['sol'], 'xr', label='Reconstructed')
>>> _ = plt.grid(True)
>>> _ = plt.title('Achieved reconstruction')
>>> _ = plt.legend(numpoints=1)
>>> _ = plt.xlabel('Signal dimension number')
>>> _ = plt.ylabel('Signal value')
>>> fig.savefig('doc/tutorials/compressed_sensing_1_results.pdf')
>>> fig.savefig('doc/tutorials/compressed_sensing_1_results.png')

.. image:: compressed_sensing_results.*

>>> # Display the convergence
>>> fig = plt.figure()
>>> objective = np.array(ret['objective'])
>>> _ = plt.semilogy(objective[:, 0], label='L1-norm')
>>> _ = plt.semilogy(objective[:, 1], label='L2-norm')
>>> _ = plt.semilogy(np.sum(objective, axis=1), label='Global objective')
>>> _ = plt.grid(True)
>>> _ = plt.title(r'Convergence with $\tau$ = %d' % tau)
>>> _ = plt.legend()
>>> _ = plt.xlabel('Iteration number')
>>> _ = plt.ylabel('Objective function value')
>>> fig.savefig('doc/tutorials/compressed_sensing_1_convergence.pdf')
>>> fig.savefig('doc/tutorials/compressed_sensing_1_convergence.png')

.. image:: compressed_sensing_convergence.*
