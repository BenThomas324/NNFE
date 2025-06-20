# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
sys.path.insert(0,'../../CARDIAX')

project = 'CARDIAX'
copyright = '2024, WCCMS'
author = 'WCCMS'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.napoleon',
'sphinx.ext.autosummary',
'sphinx_markdown_builder',
'myst_parser',
'sphinx.ext.mathjax',
'sphinx_toolbox.decorators',
'sphinx_math_dollar'
]


mathjax3_config = {
    "tex": {
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
    }
}

templates_path = ['_templates']
exclude_patterns = []

# adding this from the dolfinx conf.py
myst_enable_extensions = [
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'       # default theme
html_theme = "sphinx_rtd_theme" # Read the Docs theme
html_static_path = ['_static']

# process any tutorials by calling the method jupytext_process.process()
sys.path.insert(0,'.') # current directory; need to be able to find jupytext_process
import jupytext_process # from dolfinx conf.py
print("running juptext")
jupytext_process.process()