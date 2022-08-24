import sys
from pathlib import Path
# -- Project information -----------------------------------------------------
project = 'TVM'
copyright = '2022, xinetzone'
author = 'xinetzone'

# The full version, including alpha/beta/rc tags
release = ''
tvm_path = Path(__file__).absolute().parents[1]
sys.path.extend([str(tvm_path/"python"),
                 str(tvm_path/"vta/python"),
                 str(tvm_path/"docs"),
                 str(tvm_path/"xin"),
                 str(tvm_path/"xinetzone"),
                 ])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    # 'sphinx_tabs.tabs',
    # "sphinxext.rediraffe",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/include']

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
    # "linkify",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../docs/_static', '_static']
html_css_files = ["default.css"]

html_logo = "../docs/_static/img/tvm-logo-small.png"
html_favicon = "../docs/_static/img/tvm-logo-square.png"

html_last_updated_fmt = '%Y-%m-%d, %H:%M:%S'

html_theme_options = {
    "footer_items": ["copyright", "last-updated", "sphinx-version", ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "tvm": ("https://tvm.apache.org/docs", None),
    # "numpy": ("https://numpy.org/doc/stable", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # "matplotlib": ("https://matplotlib.org/", None),
}


def setup(app):
    app.add_object_type('confval', 'confval',
                        objname='configuration value',
                        indextemplate='pair: %s; configuration value')


# MyST-NB 设置
# 如果你希望stderr和stdout中的每个输出都被合并成一个流，请使用以下配置。
# 避免将 jupter 执行报错的信息输出到 cmd
nb_merge_streams = True
execution_allow_errors = True
jupyter_execute_notebooks = "off" # "force"

nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    ),
    'gettext': ()
}

# -- 国际化输出 ----------------------------------------------------------------
gettext_compact = False
locale_dirs = [str(tvm_path /'xinetzone/locales/')]

# Napoleon 设置
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# ``pydata-sphinx-theme`` 配置
autosummary_generate = True
html_theme_options = {
    "footer_items": ["copyright", "last-updated", "sphinx-version", ],
    "repository_url": "https://github.com/daobook/tvm",
    "repository_branch": "doc",
    "path_to_docs": "docs/",
    "toc_title": "导航",
}

panels_add_bootstrap_css = False
