# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from plotly.io._sg_scraper import plotly_sg_scraper

sys.path.insert(0, os.path.abspath('..'))

project = 'mloptimizer'
copyright = '2023, Antonio Caparrini'
author = 'Antonio Caparrini'
release = '0.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx_mdinclude',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'autoapi.extension'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_dirs = ['../mloptimizer']
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_typehints = "signature"
autoapi_ignore = ['*test*']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

image_scrapers = ('matplotlib', plotly_sg_scraper,)

autodoc_default_flags = ['members']
autosummary_generate = True
autoclass_content = 'both'
html_show_sourcelink = False
autodoc_inherit_docstrings = True
set_type_checking_flag = True

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        # The module you locally document uses None
        'mloptimizer': None,
    },
    'image_scrapers': image_scrapers,
}

html_theme_options = {
    "repository_url": "https://github.com/Caparrini/mloptimizer",
    "use_repository_button": True,
}
