# -*- coding: utf-8 -*-

r"""
This module implements some demonstrations on how to use the toolbox to solve
real problems. The following demonstrations are included :

* :func:`compressed_sensing_1`: Compressed sensing problem solved by
  forward-backward proximal splitting algorithm.
"""

from pyunlocbox import functions, solvers
import numpy as np
import matplotlib.pyplot as plt


def compressed_sensing_1():
    r"""
    TODO
    """

    # Problem creation.
    tau = 1      # Regularization parameter.
    N = 5000     # Signal size.
    K = 100      # Sparsity level.
    R = max(4, np.ceil(np.log(N)))
    M = K * R    # Number of measurements.

    print(('Signal size of %d, %d measurements : compression ratio of %3.2f'
          % (N, M, N/M)))

    # Measurements matrix.
    A = np.random.standard_normal((M, N))

    # Create a K sparse signal.
    x = np.zeros(N)
    I = np.random.permutation(N)
    x[I[0:K]] = np.random.standard_normal(K)
    x = x / np.linalg.norm(x)

    # Measurements.
    y = np.dot(A, x)

    # Set the two convex function objects.
    f1 = functions.norm_l1(lambda_=tau, verbosity='low')
    f2 = functions.norm_l2(y=y, A=A, verbosity='low')

    # Alternative 1 (same results) : pass operators instead of matrices.
    A_ = lambda x: np.dot(A, x)
    At_ = lambda x: np.dot(np.transpose(A), x)
    f3 = functions.norm_l2(y=y, A=A_, At=At_, verbosity='low')
    assert f3

    # Alternative 2 (same results) : manual definition of the L2 norm.
    f4 = functions.func()
    f4.grad = lambda x: 2.0 * np.dot(np.transpose(A), np.dot(A, x) - y)
    f4.eval = lambda x: np.linalg.norm(np.dot(A, x) - y)**2

    # Set the solver object. Step size : beta = 2*norm(A)^2.
    gamma = 0.5 / np.linalg.norm(A, ord=2)**2
    solver = solvers.forward_backward(method='FISTA', gamma=gamma)

    # Solve the problem.
    x0 = np.zeros(N)
    ret = solvers.solve([f1, f2], x0, solver, relTol=1e-4, maxIter=300,
                        verbosity='high')

    # Display the results.
    fig = plt.figure()
    plt.plot(x, 'o', label='Original')
    plt.plot(ret['sol'], 'xr', label='Reconstructed')
    plt.grid(True)
    plt.title('Achieved reconstruction')
    plt.legend(numpoints=1)
    plt.xlabel('Signal dimension number')
    plt.ylabel('Signal value')
    fig.savefig('compressed_sensing_results.png')

    # Display the convergence
    fig = plt.figure()
    objective = np.array(ret['objective'])
    plt.semilogy(objective[:, 0], label='L1-norm')
    plt.semilogy(objective[:, 1], label='L2-norm')
    plt.semilogy(np.sum(objective, axis=1), label='Global objective')
    plt.grid(True)
    plt.title(r'Convergence with $\tau$ = %d' % tau)
    plt.legend()
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function value')
    fig.savefig('compressed_sensing_convergence.png')

#    plt.show()
