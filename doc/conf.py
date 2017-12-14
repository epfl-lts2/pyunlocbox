# -*- coding: utf-8 -*-

import pyunlocbox

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax',
              'sphinx.ext.inheritance_diagram', 'sphinxcontrib.bibtex']

extensions.append('matplotlib.sphinxext.plot_directive')
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_working_directory = '.'

extensions.append('numpydoc')
numpydoc_show_class_members = False

exclude_patterns = ['_build']
source_suffix = '.rst'
master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']

project = 'PyUNLocBoX'
version = pyunlocbox.__version__
release = pyunlocbox.__version__
copyright = u'Michaël Defferrard, EPFL LTS2'

pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 2,
}
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
latex_documents = [
    ('index', 'pyunlocbox.tex', 'PyUNLocBoX documentation',
     u'Michaël Defferrard, EPFL LTS2', 'manual'),
]
