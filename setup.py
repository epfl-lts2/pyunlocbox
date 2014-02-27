#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='pyunlocbox',
    version='0.1.0',
    description='Python convex optimization toolbox using proximal splitting methods. Port of UNLocBox for Matlab.',
    long_description=readme + '\n\n' + history,
    author='EPFL LTS2',
    author_email='nathanael.perraudin@epfl.ch',
    url='https://github.com/epfl-lts2/pyunlocbox',
    packages=[
        'pyunlocbox',
    ],
    package_dir={'pyunlocbox': 'pyunlocbox'},
    include_package_data=True,
    install_requires=[
    ],
    license="BSD",
    zip_safe=False,
    keywords='pyunlocbox',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
)