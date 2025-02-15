import macromol_voxelize

## General

project = 'Macromolecular Voxelization'
copyright = '2024, Kale Kundert'
version = macromol_voxelize.__version__
release = macromol_voxelize.__version__

master_doc = 'index'
source_suffix = '.rst'
templates_path = ['_templates']
exclude_patterns = ['_build']
html_static_path = ['_static']
default_role = 'any'
trim_footnote_reference_space = True
nitpicky = True

## Extensions

extensions = [
        'myst_parser',
        'sphinx_math_dollar',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.intersphinx',
        'sphinx.ext.mathjax',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'sphinx_rtd_theme',
]
intersphinx_mapping = {
        'python': ('https://docs.python.org/3', None),
        'numpy': ('https://numpy.org/doc/stable/', None),
        'polars': ('https://docs.pola.rs/api/python/stable', '_inv/polars.inv'),
}
autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_type_aliases = {
        'Image': 'Image',
}
mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}
html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'

## Custom Extensions

from sphinx.util.docutils import SphinxDirective
from docutils import nodes

class PyMolCommand(SphinxDirective):
    required_arguments = 1

    def run(self):
        import sys
        from unittest.mock import MagicMock

        sys.modules['pymol'] = MagicMock()
        sys.modules['pymol.cgo'] = MagicMock()

        import macromol_voxelize.pymol

        cmd = getattr(macromol_voxelize.pymol, self.arguments[0])
        return [nodes.literal_block(cmd.__doc__, cmd.__doc__)]

def setup(app):
    app.add_directive('pymol-cmd', PyMolCommand)
