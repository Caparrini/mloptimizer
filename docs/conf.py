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
release = '0.5.9.24'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

image_scrapers = ('matplotlib', plotly_sg_scraper,)

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
