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
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
from io import StringIO
from os.path import basename

from docutils import nodes
from docutils.parsers.rst import Directive

sys.path.insert(0, os.path.abspath('../..'))
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# -- Project information -----------------------------------------------------

project = 'PyEchelle'
copyright = '2021, Julian Stuermer'
author = 'Julian Stuermer'

# The full version, including alpha/beta/rc tags
release = '0.1.7'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_annotation",
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
    'matplotlib.sphinxext.plot_directive',
    "sphinxarg.ext"
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
pygments_style = None
# napoleon_use_ivar = True

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------
# this is from here https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
# to fix https://github.com/sphinx-doc/sphinx/issues/2115
from sphinx.ext.napoleon.docstring import GoogleDocstring


# first, we define new methods for any new sections and add them to the class
def parse_keys_section(self, section):
    return self._format_fields('Keys', self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields('Attributes', self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields('Class Attributes', self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section


# we now patch the parse method to guarantee that the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_experimental_html5_writer = True


class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()
        try:
            exec('\n'.join(self.content))
            return [nodes.paragraph(text=sys.stdout.getvalue())]
        except Exception as e:
            return [nodes.error(None, nodes.paragraph(
                text="Unable to execute python code at %s:%d:" % (basename(self.src), self.srcline)),
                                nodes.paragraph(text=str(e)))]
        finally:
            sys.stdout = oldStdout


def setup(app):
    app.add_directive('exec', ExecDirective)
    app.add_javascript('https://cdn.plot.ly/plotly-latest.min.js')  # Custom JavaScript
    app.add_css_file('doc_theme.css')


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
