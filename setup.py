#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name='pyunlocbox',
    version='0.5.2',
    description='Convex Optimization in Python using Proximal Splitting',
    long_description=open('README.rst').read(),
    author='EPFL LTS2',
    url='https://github.com/epfl-lts2/pyunlocbox',
    packages=[
        'pyunlocbox',
        'pyunlocbox.tests'
    ],
    test_suite='pyunlocbox.tests.test_all.suite',
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={
        # Testing dependencies.
        'test': [
            'flake8',
            'coverage',
            'coveralls',
        ],
        # Dependencies to build the documentation.
        'doc': [
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-rtd-theme',
            'matplotlib',
        ],
        # Dependencies to build and upload packages.
        'pkg': [
            'wheel',
            'twine',
        ],
    },
    license="BSD",
    keywords='convex optimization',
    platforms='any',
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
