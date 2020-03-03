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
    project_urls={
        'Documentation': 'https://pyunlocbox.readthedocs.io',
        'Source Code': 'https://github.com/epfl-lts2/pyunlocbox',
        'Bug Tracker': 'https://github.com/epfl-lts2/pyunlocbox/issues',
        'Try It Online': 'https://mybinder.org/v2/gh/epfl-lts2/pyunlocbox/master?filepath=playground.ipynb',  # noqa
    },
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
        'dev': [
            # Testing dependencies.
            'flake8',
            'coverage',
            'coveralls',
            # Dependencies to build the documentation.
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-rtd-theme',
            'matplotlib',
            # Dependencies to build and upload packages.
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
