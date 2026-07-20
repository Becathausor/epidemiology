"""Configuration Sphinx pour la documentation d'epidemio_sim."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from epidemio_sim import __version__  # noqa: E402

project = "epidemio-sim"
copyright = "2026, Thierry Bécart"
author = "Thierry Bécart"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
