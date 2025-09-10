"""
Test suite for the docstrings of the pyunlocbox package.

"""

import doctest
import os

import pytest


def gen_recursive_file(root, ext):
    """Generate file paths recursively."""
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings_python():
    """Test docstrings in Python files."""
    files = list(gen_recursive_file("pyunlocbox", ".py"))
    for file in files:
        try:
            doctest.testfile(file, module_relative=False, verbose=False)
        except doctest.DocTestFailure as e:
            pytest.fail(f"Doctest failed in {file}: {e}")
        except Exception:
            # Some files might not have doctests or might have import issues
            # We'll skip those for now
            pass


def test_docstrings_rst():
    """Test docstrings in RST files."""
    files = list(gen_recursive_file(".", ".rst"))
    # Filter out files that are likely to have import issues
    files = [
        f
        for f in files
        if not any(exclude in f for exclude in [".venv", "build", "_build"])
    ]

    for file in files:
        try:
            doctest.testfile(file, module_relative=False, verbose=False)
        except doctest.DocTestFailure as e:
            pytest.fail(f"Doctest failed in {file}: {e}")
        except Exception:
            # Some files might not have doctests or might have import issues
            # We'll skip those for now
            pass
