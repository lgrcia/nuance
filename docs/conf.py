# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "nuance"
copyright = "2023, Lionel Garcia"
author = "Lionel Garcia"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_design",
]


templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build/*"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Title
# get version number from pyproject.toml
# --------------------------------------
# import toml

# pyproject = toml.load("../pyproject.toml")
# version = pyproject["tool"]["poetry"]["version"]
# html_short_title = "nuance"
# html_title = f"{html_short_title}"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
# -----

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

root_doc = "index"

html_theme_options = {
    "repository_url": "https://github.com/lgrcia/nuance",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "default_mode": "light",
}

nb_render_image_options = {"align": "center"}

html_css_files = ["style.css"]

myst_enable_extensions = [
    "dollarmath",
]

autodoc_typehints = "signature"
autoclass_content = "both"

nb_execution_excludepatterns = [
    "_build/*",
    "notebooks/tutorials/*",
    "notebooks/_*.ipynb",
]
