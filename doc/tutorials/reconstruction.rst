=================================================================
Image reconstruction (Forward-Backward, Total Variation, L2-norm)
=================================================================

This tutorial presents an image reconstruction problem solved by the
Forward-Backward splitting algorithm. The convex optimization problem is the
sum of a data fidelity term and a regularization term which expresses a prior
on the smoothness of the solution, given by

.. math:: \min\limits_x \tau \|g(x)-y\|_2^2 + \|x\|_\text{TV}

where :math:`\|\cdot\|_\text{TV}` denotes the total variation, `y` are the
measurements, `g` is a masking operator and :math:`\tau` expresses the
trade-off between the two terms.

We load an example image

>>> from pyunlocbox import signals
>>> im_original = signals.lena().gray_scale

and generate a random masking matrix

>>> import numpy as np
>>> np.random.seed(14)  # Reproducible results.
>>> mask = np.random.uniform(size=im_original.shape)
>>> mask = mask > 0.85

which masks 85% of the pixels. The masked image is given by

>>> g = lambda x: mask * x
>>> im_masked = g(im_original)

The prior objective to minimize is defined by

.. math:: f_1(x) = \|x\|_\text{TV}

which can be expressed by the toolbox TV-norm function object, instantiated
with

>>> from pyunlocbox import functions
>>> f1 = functions.norm_tv(maxit=50, dim=2)

The fidelity objective to minimize is defined by

.. math:: f_2(x) = \tau \|g(x)-y\|_2^2

which can be expressed by the toolbox L2-norm function object, instantiated
with

>>> tau = 100
>>> f2 = functions.norm_l2(y=im_masked, A=g, lambda_=tau)

.. note:: We set :math:`\tau` to a large value as we trust our measurements and
   want the solution to be close to them. For noisy measurements a lower value
   should be considered.

The step size for optimal convergence is :math:`\frac{1}{\beta}` where
:math:`\beta=2\tau` is the Lipschitz constant of the gradient of :math:`f_2`
:cite:`beck2009FISTA`. The Forward-Backward splitting algorithm is instantiated
with

>>> from pyunlocbox import solvers
>>> solver = solvers.forward_backward(method='FISTA', step=0.5/tau)

and the problem solved with

>>> ret = solvers.solve([f1, f2], im_masked, solver, maxit=100)
Solution found after 93 iterations :
    objective function f(sol) = 4.163503e+03
    last relative objective improvement : 8.291663e-04
    stopping criterion : RTOL

Let's display the results:

>>> try:
...     import matplotlib.pyplot as plt
...     fig = plt.figure()
...     ax1 = fig.add_subplot(1, 3, 1)
...     _ = ax1.imshow(im_original, cmap='gray')
...     _ = ax1.axis('off')
...     _ = ax1.set_title('Original image')
...     ax2 = fig.add_subplot(1, 3, 2)
...     _ = ax2.imshow(im_masked, cmap='gray')
...     _ = ax2.axis('off')
...     _ = ax2.set_title('Masked image')
...     ax3 = fig.add_subplot(1, 3, 3)
...     _ = ax3.imshow(ret['sol'], cmap='gray')
...     _ = ax3.axis('off')
...     _ = ax3.set_title('Reconstructed image')
...     #fig.show()
...     #fig.savefig('doc/tutorials/img/reconstruct.pdf', bbox_inches='tight')
...     #fig.savefig('doc/tutorials/img/reconstruct.png', bbox_inches='tight')
... except:
...     pass

.. image:: img/reconstruct.*

The above figure shows a good reconstruction which is both smooth (the TV
prior) and close to the measurements (the L2 fidelity).
