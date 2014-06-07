#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pyunlocbox


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
    name = pyunlocbox.__name__,
    version = pyunlocbox.__version__,
    description = 'PyUNLocBoX is a convex optimization toolbox using proximal '
                  'splitting methods implemented in Python. It is a free '
                  'software distributed under the BSD license and is a port '
                  'of the Matlab UNLocBoX toolbox.',
    long_description = readme + '\n\n' + history,
    author = pyunlocbox.__author__,
    author_email = pyunlocbox.__email__,
    url = 'https://github.com/epfl-lts2/pyunlocbox',
    packages = ['pyunlocbox'],
    package_dir = {'pyunlocbox': 'pyunlocbox'},
    include_package_data = True,
    install_requires = [
        'numpy',
    ],
    license = "BSD",
    zip_safe = False,
    keywords = 'convex optimization',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    test_suite = 'tests.test_all.suite',
)
