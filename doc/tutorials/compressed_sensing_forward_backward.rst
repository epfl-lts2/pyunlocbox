=========================================
Compressed sensing using forward-backward
=========================================

This tutorial presents a `compressed sensing
<https://en.wikipedia.org/wiki/Compressed_sensing>`_ problem solved by the
forward-backward splitting algorithm. The problem can be expressed as follow :

.. math:: \operatorname{arg\,min}\limits_x \|Ax-y\|^2 + \tau \|x\|_1

where `y` are the measurements, `A` is the measurement matrix and `tau` is the
regularization parameter.

The number of measurements `M` is computed with respect to the signal size `N`
and the sparsity level `K` :

>>> N = 5000
>>> K = 100
>>> import numpy as np
>>> M = int(K * max(4, np.ceil(np.log(N))))
>>> print('Number of measurements : %d' % (M,))
Number of measurements : 900
>>> print('Compression ratio : %3.2f' % (float(N)/M,))
Compression ratio : 5.56

.. note:: With the above defined number of measurements, the algorithm is
    supposed to very often perform a perfect reconstruction.

We generate a random measurement matrix `A` :

>>> np.random.seed(1)  # Reproducible results.
>>> A = np.random.standard_normal((M, N))

Create the `K` sparse signal `x` :

>>> x = np.zeros(N)
>>> I = np.random.permutation(N)
>>> x[I[0:K]] = np.random.standard_normal(K)
>>> x = x / np.linalg.norm(x)

Generate the measured signal `y` :

>>> y = np.dot(A, x)

The first objective function to minimize is defined by

.. math:: f_1(x) = \tau \cdot \|x\|_1

which can be expressed by the toolbox L1-norm function object. It can be
instantiated as follow, while setting the regularization parameter `tau` :

>>> from pyunlocbox import functions
>>> tau = 1.0
>>> f1 = functions.norm_l1(lambda_=tau)

The second objective function to minimize is defined by

.. math:: f_2(x) = \|Ax-b\|_2^2

which can be expressed by the toolbox L2-norm function object. It can be
instantiated as follow :

>>> f2 = functions.norm_l2(y=y, A=A)

or alternatively as follow :

>>> A_ = lambda x: np.dot(A, x)
>>> At_ = lambda x: np.dot(np.transpose(A), x)
>>> f3 = functions.norm_l2(y=y, A=A_, At=At_)

.. note:: In this case the forward and adjoint operators were passed as
    functions not as matrices.

A third alternative would be to define the function object by hand :

>>> f4 = functions.func()
>>> f4._grad = lambda x: 2.0 * np.dot(np.transpose(A), np.dot(A, x) - y)
>>> f4._eval = lambda x: np.linalg.norm(np.dot(A, x) - y)**2

.. note:: The three alternatives to instantiate the function objects (`f2`,
    `f3` and `f4`) are strictly equivalent and give the exact same results.

Now that the two function objects to minimize (the L1-norm and the L2-norm) are
instantiated, we can instantiate the solver object. The step size for optimal
convergence is :math:`\frac{1}{\beta}` where :math:`\beta` is the Lipschitz
constant of the gradient of `f2`, `f3`, `f4` given by:

.. math:: \beta = 2 \cdot \|A\|_{\text{op}}^2 = 2 \cdot \lambda_{max} (A^*A).

To solve this problem, we use the forward-backward splitting algorithm which is
instantiated as follow :

>>> step = 0.5 / np.linalg.norm(A, ord=2)**2
>>> from pyunlocbox import solvers
>>> solver = solvers.forward_backward(method='FISTA', step=step)

.. note:: A complete description of the constructor parameters and default
    values is given by the solver object
    :class:`pyunlocbox.solvers.forward_backward` reference documentation.

After the instantiations of the functions and solver objects, the setting of a
starting point `x0`, the problem is solved by the toolbox solving function as
follow :

>>> x0 = np.zeros(N)
>>> ret = solvers.solve([f1, f2], x0, solver, rtol=1e-4, maxit=300)
Solution found after 176 iterations :
    objective function f(sol) = 8.221302e+00
    stopping criterion : RTOL

.. note:: A complete description of the parameters, their default values and
    the returned values is given by the solving function
    :func:`pyunlocbox.solvers.solve` reference documentation.

Let's display the results :

>>> try:
...     import matplotlib.pyplot as plt
...     _ = plt.figure()
...     _ = plt.plot(x, 'o', label='Original')
...     _ = plt.plot(ret['sol'], 'xr', label='Reconstructed')
...     _ = plt.grid(True)
...     _ = plt.title('Achieved reconstruction')
...     _ = plt.legend(numpoints=1)
...     _ = plt.xlabel('Signal dimension number')
...     _ = plt.ylabel('Signal value')
...     #plt.savefig('doc/tutorials/img/cs_fb_results.pdf')
...     #plt.savefig('doc/tutorials/img/cs_fb_results.png')
... except:
...     pass

.. image:: img/cs_fb_results.*

The above figure shows a good reconstruction which is both sparse (thanks to
the L1-norm objective) and close to the measurements (thanks to the L2-norm
objective).

Let's display the convergence of the two objective functions :

>>> try:
...     objective = np.array(ret['objective'])
...     _ = plt.figure()
...     _ = plt.semilogy(objective[:, 0], label='L1-norm objective')
...     _ = plt.semilogy(objective[:, 1], label='L2-norm objective')
...     _ = plt.semilogy(np.sum(objective, axis=1), label='Global objective')
...     _ = plt.grid(True)
...     _ = plt.title('Convergence')
...     _ = plt.legend()
...     _ = plt.xlabel('Iteration number')
...     _ = plt.ylabel('Objective function value')
...     #plt.savefig('doc/tutorials/img/cs_fb_convergence.pdf')
...     #plt.savefig('doc/tutorials/img/cs_fb_convergence.png')
... except:
...     pass

.. image:: img/cs_fb_convergence.*
