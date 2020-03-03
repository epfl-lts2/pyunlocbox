============
Contributing
============

Contributions are welcome, and they are greatly appreciated! The development of
this package takes place on `GitHub <https://github.com/epfl-lts2/pyunlocbox>`_.
Issues, bugs, and feature requests should be reported `there
<https://github.com/epfl-lts2/pyunlocbox/issues>`_.
Code and documentation can be improved by submitting a `pull request
<https://github.com/epfl-lts2/pyunlocbox/pulls>`_. Please add documentation and
tests for any new code.

The package can be set up (ideally in a virtual environment) for local
development with the following::

    $ git clone https://github.com/epfl-lts2/pyunlocbox.git
    $ pip install -U -e pyunlocbox[dev]

The ``dev`` "extras requirement" ensures that dependencies required for
development (to run the test suite and build the documentation) are installed.

You can improve or add solvers, functions, and acceleration schemes in
``pyunlocbox/solvers.py``, ``pyunlocbox/functions.py``, and
``pyunlocbox/acceleration.py``, along with their corresponding unit tests in
``pyunlocbox/tests/test_*.py`` (with reasonable coverage) and documentation in
``doc/reference/*.rst``. If you have a nice example to demonstrate the use of
the introduced functionality, please consider adding a tutorial in
``doc/tutorials``.

Do not forget to update ``README.rst`` and ``doc/history.rst`` with e.g. new
features.

After making any change, please check the style, run the tests, and build the
documentation with the following (enforced by Travis CI)::

    $ make lint
    $ make test
    $ make doc

Check the generated coverage report at ``htmlcov/index.html`` to make sure the
tests reasonably cover the changes you've introduced.

To iterate faster, you can partially run the test suite, at various degrees of
granularity, as follows::

   $ python -m unittest pyunlocbox.tests.test_functions
   $ python -m unittest pyunlocbox.tests.test_functions.TestCase.test_norm_l1

Making a release
----------------

#. Update the version number and release date in ``setup.py``,
   ``pyunlocbox/__init__.py`` and ``doc/history.rst``.
#. Create a git tag with ``git tag -a v0.5.0 -m "PyUNLocBox v0.5.0"``.
#. Push the tag to GitHub with ``git push github v0.5.0``. The tag should now
   appear in the releases and tags tab.
#. `Create a release <https://github.com/epfl-lts2/pygsp/releases/new>`_ on
   GitHub and select the created tag. A DOI should then be issued by Zenodo.
#. Go on Zenodo and fix the metadata if necessary.
#. Build the distribution with ``make dist`` and check that the
   ``dist/pyunlocbox-0.5.0.tar.gz`` source archive contains all required files.
   The binary wheel should be found as
   ``dist/pyunlocbox-0.5.0-py2.py3-none-any.whl``.
#. Test the upload and installation process::

    $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    $ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyunlocbox

   Log in as the LTS2 user.
#. Build and upload the distribution to the real PyPI with ``make release``.
#. Update the conda feedstock (at least the version number and sha256 in
   ``recipe/meta.yaml``) by sending a PR to
   `conda-forge <https://github.com/conda-forge/pyunlocbox-feedstock>`_.
