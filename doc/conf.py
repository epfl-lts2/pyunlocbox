#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Import package from source directory (without installation).
cwd = os.getcwd()
project_root = os.path.dirname(cwd)
sys.path.insert(0, project_root)
import pyunlocbox  # noqa: E402

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax', 'numpydoc',
              'sphinx.ext.inheritance_diagram', 'sphinxcontrib.bibtex']
exclude_patterns = ['_build']
source_suffix = '.rst'
master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']

project = pyunlocbox.__name__
version = pyunlocbox.__version__
release = pyunlocbox.__version__
copyright = 'EPFL LTS2'

numpydoc_show_class_members = False
pygments_style = 'sphinx'
html_theme = 'default'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
latex_documents = [
    ('index', 'pyunlocbox.tex', 'PPyUNLocBoX documentation',
     u'Michaël Defferrard, EPFL LTS2', 'manual'),
]
