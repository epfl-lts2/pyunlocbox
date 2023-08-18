r"""
SV_DECONV
=====
This example implements the saptially varying PSF deconvolution algorithm
based on: Flicker & Rigaut, 2005 https://doi.org/10.1364/JOSAA.22.000504
"""

import numpy as np
import pyunlocbox as plx
from scipy.signal import fftconvolve

###############################################################################
# Sample data
im = np.random.randn(2748, 3840)
W = np.random.randn(20, np.prod(im.shape))
U = np.random.randn(np.prod(im.shape), 20)

# Setup Forward and Adjoint Models and Create Solver
def forward(im):
    im_sim = np.zeros_like(im)
    for i in range(15):
        weight = W[i,:].reshape(*im.shape)
        psf_mode = U[:,i].reshape(*im.shape)
        im_sim += fftconvolve(im* weight, psf_mode, mode='same')
    return im_sim

def forward_adj(im):
    im_sim = np.zeros_like(im)
    for i in range(15):
        weight = W[i,:].reshape(*im.shape)
        psf_mode = U[:,i].reshape(*im.shape)
        im_sim += fftconvolve(im, np.flipud(np.fliplr(psf_mode)), mode='same')* weight
    return im_sim

tau=10
f1 = plx.functions.norm_l2(y=im, A=forward, At=forward_adj, lambda_=tau)
f2 = plx.functions.norm_tv(maxit=50, dim=2)
solver = plx.solvers.forward_backward(step=0.5/tau, accel=plx.acceleration.fista())
ret = plx.solvers.solve([f1, f2], x0=im.copy(), solver=solver, maxit=10, verbosity='ALL')