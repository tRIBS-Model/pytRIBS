import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

autodoc_default_options = {
    'exclude-members': '_*',
}

# Project information
project = 'pytRIBS'
author = 'Wren Raming'
# Derive the version from the installed package metadata so it never goes stale.
from importlib.metadata import version as _pkg_version
release = _pkg_version('pytRIBS')

# Add any Sphinx extension module names here
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

# HTML output theme
html_theme = 'sphinx_rtd_theme'

html_sidebars = {
    '**': ['localtoc.html', 'relations.html', 'searchbox.html', 'sourcelink.html'],
}

# -- Extensions to the  Napoleon GoogleDocstring class ---------------------

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

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse
