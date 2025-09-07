"""
Sphinx配置文件 - ThermoElasticSim项目文档

该配置文件设置了Sphinx文档生成系统的所有参数。
支持NumPy风格的docstring和LaTeX数学公式渲染。
"""

import os
import sys
from datetime import datetime

# 添加项目路径，以便autodoc能找到模块
sys.path.insert(0, os.path.abspath("../../src"))

# -- 项目信息 ----------------------------------------------------------------

project = "ThermoElasticSim"
copyright = f"{datetime.now().year}, Gilbert Young"
author = "Gilbert Young"
version = "4.0.0"
release = "4.0.0"
language = "zh_CN"

# -- 通用配置 ----------------------------------------------------------------

# Sphinx扩展列表
extensions = [
    "sphinx.ext.autodoc",  # 自动从代码提取文档
    "sphinx.ext.napoleon",  # 支持NumPy和Google风格的docstring
    "sphinx.ext.viewcode",  # 添加源代码链接
    "sphinx.ext.mathjax",  # LaTeX数学公式渲染
    "sphinx.ext.intersphinx",  # 链接到其他项目文档
    "sphinx.ext.coverage",  # 文档覆盖率检查
    "sphinx.ext.todo",  # TODO标记支持
    "myst_nb",  # MyST-NB：Jupyter Notebook集成（包含myst_parser功能）
    "sphinxcontrib.bibtex",  # BibTeX文献引用支持
]

# BibTeX配置
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# MyST-NB配置
nb_execution_mode = "cache"  # 缓存执行结果，避免重复计算
nb_execution_timeout = 30  # 轻量示例超时（秒）
nb_execution_excludepatterns = [  # 排除重计算案例
    "tutorial/gallery/advanced/*.ipynb",
    "tutorial/**/05b_wave_propagation.ipynb",
]
nb_execution_allow_errors = False  # 不允许执行错误
nb_execution_show_tb = True  # 显示traceback便于调试

# MyST-Parser配置
myst_enable_extensions = [
    "dollarmath",  # $...$ 数学公式
    "amsmath",  # LaTeX AMS扩展
    "deflist",  # 定义列表
    "tasklist",  # 任务列表
]

# 模板路径
templates_path = ["_templates"]

# 排除的文件模式
exclude_patterns = []

# 源文件后缀
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",  # MyST-NB统一处理Markdown和Notebook
    ".ipynb": "myst-nb",  # Jupyter Notebook支持
}

# 主文档
master_doc = "index"

# -- autodoc配置 --------------------------------------------------------------

# 自动文档选项
autodoc_default_options = {
    "members": True,  # 包含成员
    "member-order": "bysource",  # 按源码顺序
    "special-members": "__init__",  # 包含__init__
    "undoc-members": False,  # 不包含无文档成员
    "exclude-members": "__weakref__",  # 排除弱引用
    "show-inheritance": True,  # 显示继承关系
    "inherited-members": False,  # 不显示继承的成员
    "private-members": False,  # 不显示私有成员
    "imported-members": False,  # 不默认包含导入成员，避免重复
}

# Mock导入（对于C扩展等）
autodoc_mock_imports = [
    # C/C++ bindings
    "pybind11",
    "thermoelasticsim._cpp_core",
    # Heavy/optional runtime deps mocked for docs build
    "numba",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.animation",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    "plotly.subplots",
    "h5py",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    # Project subpackages that trigger heavy runtime side-effects
    # Note: we need to be careful not to mock modules that are used by API docs
    # "thermoelasticsim.visualization",  # Don't mock the whole package
    # "thermoelasticsim.visualization.elastic",  # benchmark module needs this
    "thermoelasticsim.visualization.web",  # Only mock web visualization
    "thermoelasticsim.utils.plot_config",
    "thermoelasticsim.utils.visualization",
    "thermoelasticsim.utils.modern_visualization",
    # Note: benchmark module should NOT be mocked - it contains API we need to document
]

# 类型提示配置
autodoc_typehints = "description"
autodoc_type_aliases = {
    "ArrayLike": "array_like",
    "Cell": "thermoelasticsim.core.structure.Cell",
    "Potential": "thermoelasticsim.potentials.base.Potential",
    "NeighborList": "thermoelasticsim.core.neighbor_list.NeighborList",
}

# -- Napoleon配置 (NumPy文档风格) --------------------------------------------

napoleon_google_docstring = False  # 不使用Google风格
napoleon_numpy_docstring = True  # 使用NumPy风格
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    # 解决常见类型的交叉引用歧义（映射到唯一目标）
    "Cell": "thermoelasticsim.core.structure.Cell",
    "Atom": "thermoelasticsim.core.structure.Atom",
    "Potential": "thermoelasticsim.potentials.base.Potential",
    "NeighborList": "thermoelasticsim.potentials.base.NeighborList",
}

# NumPy文档参数
napoleon_attr_annotations = True
napoleon_custom_sections = [
    "Theory",  # 理论基础
    "Algorithm",  # 算法说明
]
# 若启用 numpydoc，可恢复：numpydoc_show_class_members = False

# -- MathJax配置 (数学公式) --------------------------------------------------

# MathJax版本和配置
mathjax_version = "3"
mathjax3_config = {
    "tex": {
        "macros": {
            "bm": [r"\boldsymbol{#1}", 1],  # 粗体向量
            "avg": [r"\langle #1 \rangle", 1],  # 平均值
            "ket": [r"|#1\rangle", 1],  # 狄拉克ket
            "bra": [r"\langle #1|", 1],  # 狄拉克bra
        },
        "packages": {"[+]": ["ams", "physics"]},  # 加载额外包
    },
}

# -- HTML输出配置 -------------------------------------------------------------

# HTML主题
html_theme = "sphinx_rtd_theme"

# 主题选项
html_theme_options = {
    "navigation_depth": 4,  # 导航深度
    "collapse_navigation": False,  # 不折叠导航
    "sticky_navigation": True,  # 固定导航栏
    "includehidden": True,  # 包含隐藏的toctree
    "titles_only": False,  # 显示子标题
    "prev_next_buttons_location": "bottom",  # 上下页按钮位置
    "vcs_pageview_mode": "edit",  # GitHub编辑链接
    "style_external_links": True,  # 外部链接样式
}

# GitHub页面配置
html_context = {
    "display_github": True,  # 显示GitHub链接
    "github_user": "bud-primordium",  # GitHub用户名
    "github_repo": "ThermoElasticSim",  # 仓库名
    "github_version": "main",  # 分支名
    "conf_py_path": "/docs/source/",  # conf.py路径
    "source_suffix": source_suffix,  # 源文件后缀
}

# 静态文件路径
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# 网站图标
# html_favicon = '_static/favicon.ico'
# html_logo = '_static/logo.png'

# HTML标题
html_title = f"{project} v{version} 文档"

# 侧边栏
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# 显示源码链接
html_show_sourcelink = True

# -- 严格模式配置 -------------------------------------------------------------
# 启用nitpicky模式，任何未解析的交叉引用都会导致构建失败
nitpicky = True

# 忽略某些已知问题的引用（如果需要）
nitpick_ignore = [
    # 常见类型/占位名（不作为交叉引用目标）
    ("py:class", "array_like"),
    ("py:class", "optional"),
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "typing.Optional"),
    ("py:class", "Integrator"),
    ("py:class", "ElasticTrajectoryRecorder"),
    ("py:class", "optimizer"),
    # Internal optimizer cross-refs
    ("py:class", "thermoelasticsim.utils.optimizers.Optimizer"),
    ("py:class", "thermoelasticsim.utils.optimizers.LBFGSOptimizer"),
    # 外部异常类型
    ("py:exc", "yaml.YAMLError"),
    # 文档中使用的占位对象名
    ("py:obj", "dt"),
    ("py:obj", "target_temperature"),
    ("py:obj", "mode"),
    ("py:obj", "{'disp': True}"),
    ("py:obj", "cell"),
    # 常见基础类型名在不同子包下重复定义，避免歧义告警
    ("py:class", "Cell"),
    ("py:class", "Atom"),
    ("py:class", "Potential"),
    ("py:class", "NeighborList"),
    # 全限定名（当对应模块被 :noindex: 时，避免链接失败告警）
    ("py:class", "thermoelasticsim.core.structure.Cell"),
    ("py:class", "thermoelasticsim.core.structure.Atom"),
]
html_copy_source = True

# -- LaTeX输出配置 ------------------------------------------------------------

latex_engine = "xelatex"  # 支持中文
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
\usepackage{amsmath,amssymb}
\usepackage{physics}
% 注释掉字体设置，让Sphinx使用默认字体避免冲突
% 如果需要特定字体，请确保系统已安装
% \setmainfont{Times New Roman}
% \setsansfont{Arial}
% 或使用更通用的字体设置
\usepackage{fontspec}
\defaultfontfeatures{Ligatures=TeX}
% 中文字体设置（使用系统默认）
\usepackage[UTF8]{ctex}
""",
    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
    "printindex": r"\footnotesize\raggedright\printindex",
}

# LaTeX文档结构
latex_documents = [
    (
        master_doc,
        "ThermoElasticSim.tex",
        "ThermoElasticSim Documentation",
        "Gilbert Young",
        "manual",
    ),
]

# -- Intersphinx配置 (链接到其他项目) ----------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- 其他配置 ----------------------------------------------------------------

# TODO扩展配置
todo_include_todos = True

# 代码高亮风格
pygments_style = "sphinx"

# 保持成员顺序
autodoc_member_order = "bysource"

# 忽略的警告
suppress_warnings = [
    "autodoc.import_error",
    "ref.citation",
    "ref.python",
    "mystnb.unknown_mime_type",  # 抑制plotly等未知MIME类型警告
]

# 默认角色
# 不设置默认角色，避免将普通词语当作交叉引用解析
# default_role = "py:obj"


# 添加自定义CSS（如果需要）
def setup(app):
    """Sphinx应用设置钩子"""
    pass


# -- SimplePDF配置 ------------------------------------------------------------
# Sphinx-SimplePDF配置（如果扩展可用）
simplepdf_vars = {
    "primary-color": "#3498db",
    "secondary-color": "#2c3e50",
    "cover": True,
    "cover-title": "ThermoElasticSim 文档",
    "cover-subtitle": f"版本 {version}",
    "cover-author": author,
}

simplepdf_file_name = "ThermoElasticSim.pdf"
simplepdf_debug = False


# -- rst2pdf配置 --------------------------------------------------------------
# rst2pdf配置（轻量级PDF生成器，无需系统依赖）
pdf_documents = [
    ("index", "ThermoElasticSim", "ThermoElasticSim Documentation", author),
]

# rst2pdf样式配置
pdf_stylesheets = ["sphinx"]
pdf_language = "zh_CN"
pdf_fit_mode = "shrink"
pdf_break_level = 1
pdf_use_index = True
pdf_use_modindex = True
pdf_use_coverpage = True
