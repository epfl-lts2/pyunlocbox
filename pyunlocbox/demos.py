#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pyunlocbox import signals, functions, solvers


def douglas_rachford():
    """
    Douglas Rachford demonstration

    Examples
    --------
    >>> import pyunlocbox
    >>> pyunlocbox.demos.douglas_rachford()
    Solution found after 54 iterations :
        objective function f(sol) = 4.791124e+03
        last relative objective improvement : 5.675881e-04
        stopping criterion : RTOL

    """

    # Original image
    s = signals.lena()
    im_original = s.gray_scale
    # Creating the problem
    np.random.seed(14)  # Reproducible results.
    A = np.random.rand(im_original.shape[0], im_original.shape[1])
    A = A > 0.85

    # Depleted image
    im_depleted = im_original * A

    # Defining proximal operators
    # Define the prox of f2 see the function proj_B2 for more help
    operatorA = lambda x: A * x
    f2 = functions.proj_b2(y=im_depleted, A=operatorA, At=operatorA, epsilon=0)
    f1 = functions.norm_tv(maxit=50, dim=2)

    # Solving the problem
    solver = solvers.douglas_rachford(lambda_=1, step=0.1)
    param = {'x0': im_depleted, 'solver': solver,
             'atol': 1e-5, 'maxit': 200, 'verbosity': 'LOW'}
    ret = solvers.solve([f1, f2], **param)
    sol = ret['sol']

    # Show the result
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(im_original, cmap='gray')
    a.set_title('Original image')
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(im_depleted, cmap='gray')
    a.set_title('Depleted image')
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(sol, cmap='gray')
    a.set_title('Reconstructed image')
    plt.show()


def denoising():
    """
    Denoising demonstration

    Examples
    --------
    >>> import pyunlocbox
    >>> pyunlocbox.demos.denoising()
    Solution found after 137 iterations :
        objective function f(sol) = 3.959749e+05
        last relative objective improvement : 9.802712e-04
        stopping criterion : RTOL

    """

    # Original image
    s = signals.whitecircle()
    im_original = s.gray_scale

    # Creating the problem
    sigma = 10

    # Depleted image
    np.random.seed(7)  # Reproducible results.
    im_depleted = im_original + sigma * np.random.randn(*im_original.shape)

    # Defining proximal operators
    # Define the prox of f2 see the function proj_B2 for more help
    f2 = functions.proj_b2(y=im_depleted, epsilon=sigma * np.sqrt(np.size(im_original)))
    f3 = functions.norm_l2(y=im_depleted, lambda_=1)
    f1 = functions.norm_tv(maxit=50, dim=2)

    # Solving the problem
    solver = solvers.douglas_rachford(lambda_=1, step=0.1)
    param = {'x0': im_depleted, 'solver': solver,
             'atol': 1e-5, 'maxit': 200, 'verbosity': 'LOW'}
    ret = solvers.solve([f1, f2], **param)
    sol = ret['sol']

    # Show the result
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(im_original, cmap='gray')
    a.set_title('Original image')
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(im_depleted, cmap='gray')
    a.set_title('Depleted image')
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(sol, cmap='gray')
    a.set_title('Reconstructed image')
    plt.show()


def test_prox():
    """
    Douglas Rachford demonstration

    Examples
    --------
    >>> import pyunlocbox
    >>> pyunlocbox.demos.test_prox()
    Proximal TV Operator

    """

    # Original image
    s = signals.whitecircle()
    im_original = s.gray_scale

    f1 = functions.norm_tv(maxit=200, dim=2, verbosity='LOW')

    im_depleted = f1.prox(im_original, 2000)
    # Show the result
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(im_original, cmap='gray')
    a.set_title('Original image')
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(im_depleted, cmap='gray')
    a.set_title('Depleted image')
    plt.show()
