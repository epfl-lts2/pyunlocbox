#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pyunlocbox
from setuptools import setup


setup(
    name = pyunlocbox.__name__,
    version = pyunlocbox.__version__,
    description = 'A convex optimization toolbox using proximal '
                  'splitting methods.',
    long_description = open('README.rst').read(),
    author = 'Michaël Defferrard (EPFL LTS2) and '
             'Nathanaël Perraudin (EPFL LTS2)',
    author_email = 'michael.defferrard@epfl.ch, nathanael.perraudin@epfl.ch',
    url = 'https://github.com/epfl-lts2/pyunlocbox',
    packages = [pyunlocbox.__name__],
    test_suite = 'pyunlocbox.tests.test_all.suite',
    requires = [
        'numpy',
    ],
    license = "BSD",
    keywords = 'convex optimization',
    classifiers = [
        'Topic :: Scientific/Engineering :: Mathematics',
        'Environment :: Console',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
)
