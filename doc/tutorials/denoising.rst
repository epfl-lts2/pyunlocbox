============================================================
Image denoising (Douglas-Rachford, Total Variation, L2-norm)
============================================================

This tutorial presents an image denoising problem solved by the
Douglas-Rachford splitting algorithm. The convex optimization problem, a term
which expresses a prior on the smoothness of the solution constrained by some
data fidelity, is given by

.. math:: \min\limits_x \|x\|_\text{TV} \text{ s.t. } \|x-y\|_2 \leq \epsilon

where :math:`\|\cdot\|_\text{TV}` denotes the total variation, `y` are the
measurements and :math:`\epsilon` expresses the noise level.

We load an example image

>>> from pyunlocbox import signals
>>> im_original = signals.whitecircle().gray_scale

and add some random Gaussian noise

>>> sigma = 100
>>> import numpy as np
>>> np.random.seed(7)  # Reproducible results.
>>> im_noisy = im_original + sigma * np.random.normal(size=im_original.shape)

The prior objective function to minimize is defined by

.. math:: f_1(x) = \|x\|_\text{TV}

which can be expressed by the toolbox TV-norm function object, instantiated
with

>>> from pyunlocbox import functions
>>> f1 = functions.norm_tv(maxit=50, dim=2)

The fidelity constraint expressed as an objective function to minimize is
defined by

.. math:: f_2(x) = \iota_S(x)

where :math:`\iota_S()` is the indicator function of the set :math:`S =
\left\{z \in \mathbb{R}^n \mid \|z-y\|_2 \leq \epsilon \right\}` which is zero
if :math:`z` is in the set and infinite otherwise. This function can be
expressed by the toolbox L2-ball function, instantiated with

>>> epsilon = sigma * np.sqrt(im_original.size)
>>> f2 = functions.proj_b2(y=im_noisy, epsilon=epsilon)

The Douglas-Rachford splitting algorithm is instantiated with

>>> from pyunlocbox import solvers
>>> solver = solvers.douglas_rachford(step=10)

and the problem solved with

>>> ret = solvers.solve([f1, f2], im_noisy, solver, maxit=200)
Solution found after 20 iterations :
    objective function f(sol) = 3.972049e+05
    last relative objective improvement : 3.633195e-04
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
...     _ = ax2.imshow(im_noisy, cmap='gray')
...     _ = ax2.axis('off')
...     _ = ax2.set_title('Noisy image')
...     ax3 = fig.add_subplot(1, 3, 3)
...     _ = ax3.imshow(ret['sol'], cmap='gray')
...     _ = ax3.axis('off')
...     _ = ax3.set_title('Denoised image')
...     #fig.show()
...     #fig.savefig('doc/tutorials/img/denoising.pdf', bbox_inches='tight')
...     #fig.savefig('doc/tutorials/img/denoising.png', bbox_inches='tight')
... except:
...     pass

.. image:: img/denoising.*

The above figure shows a good reconstruction which is both smooth (the TV
prior) and close to the measurements (the L2 fidelity constraint).
