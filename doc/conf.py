# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import sys
from pathlib import Path


sys.path.insert(0, str(Path("..", "src").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AideForge"
copyright = "2026, Ferenc Vajda"
author = "Ferenc Vajda"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
    "enum_tools.autoenum",
]
templates_path = ["_templates"]
toc_object_entries_show_parents = "hide"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "bizstyle"
html_static_path = ["_static"]

# -- Autodoc Configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_type_aliases = {
    # structs.theme
    "ColorValue": "aideforge.structs.theme.ColorValue",
    "ColorType": "aideforge.structs.theme.ColorType",
    "WebColorValue": "aideforge.structs.theme.WebColorValue",
    "WebColorType": "aideforge.structs.theme.WebColorType",
}

# -- Napoleon Configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

#>napoleon_custom_sections = [["Class Metadata", "attributes"]]
napoleon_custom_sections = [["Options", "attributes"]]

# -- Autodoc Pydantic Configuration ------------------------------------------
# https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html

autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_field_doc_policy = "both"
