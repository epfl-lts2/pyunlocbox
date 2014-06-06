# -*- coding: utf-8 -*-

r"""
This module implements some demonstrations on how to use the toolbox to solve
real problems. The following demonstrations are included :

* :func:`compressed_sensing_1`: Compressed sensing problem solved by
  forward-backward proximal splitting algorithm.
"""

from pyunlocbox.functions import norm_l1, norm_l2
from pyunlocbox.solvers import solve, forward_backward
import numpy as np


def compressed_sensing_1():
    r"""
    TODO
    """

    # Problem creation.
    tau = 1      # Regularization parameter.
    N = 5000     # Signal size.
    K = 100      # Sparsity level.
    R = max(4, np.ceil(np.log(N)))
    M = K * R * 2    # Number of measurements.

    print('The compression ratio is %f' % (N/M,))

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
    f1 = norm_l1(lambda_=tau)
    A_ = lambda x: np.dot(A, x)
    At_ = lambda x: np.dot(np.transpose(A), x)
    f2 = norm_l2(y=y, A=A_, At=At_)

    # Set the solver object.
    gamma = 0.5 / np.linalg.norm(A)**2  # Step size (beta = 2*norm(A)^2)
    solver = forward_backward(method='ISTA', gamma=gamma)

    # Solve the problem.
    ret = solve([f1, f2], np.zeros(N), solver, relTol=10**-4, maxIter=300,
                verbosity='high')
