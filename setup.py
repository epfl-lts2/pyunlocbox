#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name='pyunlocbox',
    version='0.5.2',
    description='Convex Optimization in Python using Proximal Splitting',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='EPFL LTS2',
    url='https://github.com/epfl-lts2/pyunlocbox',
    project_urls={
        'Documentation': 'https://pyunlocbox.readthedocs.io',
        'Download': 'https://pypi.org/project/pyunlocbox',
        'Source Code': 'https://github.com/epfl-lts2/pyunlocbox',
        'Bug Tracker': 'https://github.com/epfl-lts2/pyunlocbox/issues',
        'Try It Online': 'https://mybinder.org/v2/gh/epfl-lts2/pyunlocbox/master?filepath=examples/playground.ipynb',  # noqa
    },
    packages=[
        'pyunlocbox',
        'pyunlocbox.tests'
    ],
    test_suite='pyunlocbox.tests.suite',
    install_requires=[
        'numpy',
        'scipy'
    ],
    extras_require={
        'dev': [
            # Run the tests.
            'flake8',
            'coverage',
            'coveralls',
            # Build the documentation.
            'sphinx',
            'numpydoc',
            'sphinxcontrib-bibtex',
            'sphinx-rtd-theme',
            'sphinx-copybutton',
            'matplotlib',
            # Build and upload packages.
            'wheel',
            'twine',
        ],
    },
    license="BSD",
    keywords='convex optimization',
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
