# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ThermoElasticSim"
copyright = "2024, Gilbert Young"
author = "Gilbert Young"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "numpydoc", "breathe", "sphinx.ext.viewcode"]

templates_path = ["_templates"]
exclude_patterns = []

language = "zh_CN"

# -- Python source code path -------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../src/python"))

# -- Breathe configuration ---------------------------------------------------
breathe_projects = {"ThermoElasticSim": "../docs/xml"}
breathe_default_project = "ThermoElasticSim"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
