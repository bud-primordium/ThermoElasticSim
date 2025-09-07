安装指南
========

本页面提供ThermoElasticSim的完整安装说明。

系统要求
--------

**基本要求**
  - Python 3.9+
  - C++17兼容编译器（用于构建C++扩展）
  - CMake >= 3.15

**推荐工具**
  - `uv <https://github.com/astral-sh/uv>`_ ：快速Python包管理器
  - `git <https://git-scm.com>`_ ：版本控制工具

工具安装
--------

安装uv（推荐）
~~~~~~~~~~~~~~

uv是极速的Python包管理器，使用Rust编写：

**macOS/Linux**:

.. code-block:: bash

   # 使用官方安装脚本
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # 或使用homebrew
   brew install uv

**Windows**:

.. code-block:: powershell

   # 使用官方安装脚本
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # 或下载预编译二进制文件
   # https://github.com/astral-sh/uv/releases

**验证安装**:

.. code-block:: bash

   uv --version

安装编译工具
~~~~~~~~~~~~

**macOS**:

.. code-block:: bash

   # 安装Xcode命令行工具
   xcode-select --install

**Ubuntu/Debian**:

.. code-block:: bash

   sudo apt update
   sudo apt install build-essential cmake git

**Windows**:

- 确保包含C++构建工具和CMake

快速安装
--------

方法一：使用uv（推荐）
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. 克隆仓库
   git clone https://github.com/bud-primordium/ThermoElasticSim.git
   cd ThermoElasticSim

   # 2. 创建虚拟环境并安装
   uv venv
   uv pip install -e .

   # 3. 安装开发依赖（可选）
   uv pip install -e ".[dev]"

   # 4. 安装文档依赖（可选）
   uv pip install -e ".[docs]"

方法二：使用pip
~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. 克隆仓库
   git clone https://github.com/bud-primordium/ThermoElasticSim.git
   cd ThermoElasticSim

   # 2. 创建虚拟环境
   python -m venv .venv

   # 激活虚拟环境
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows

   # 3. 升级构建工具
   python -m pip install --upgrade pip setuptools wheel

   # 4. 安装项目
   python -m pip install -e .

验证安装
--------

运行测试套件
~~~~~~~~~~~~

.. code-block:: bash

   # 开发者：先安装测试依赖
   uv pip install -e ".[dev]"

   # 运行测试（推荐用模块方式，避免PATH混淆）
   python -m pytest
   # 或：uv run --no-sync python -m pytest

   # 可选：若坚持直接调用pytest，请确保指向venv：
   .venv/bin/pytest

.. note::

   **重要提示**: 如果你启用了Conda的base环境，可能存在 ``pytest`` 命令来自base的情况，
   从而绕过当前venv。使用 ``python -m pytest`` 可确保使用venv里的解释器与依赖。

测试应该全部通过（预期217个测试）。

验证核心功能
~~~~~~~~~~~~

.. code-block:: bash

   # 验证NVE能量守恒
   python debug/test_nve_conservation.py

   # 使用uv运行
   uv run --no-sync python debug/test_nve_conservation.py

运行示例
~~~~~~~~

.. code-block:: bash

   # CLI教学场景
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nve.yaml

   # 零温弹性常数基准测试
   python examples/legacy_py/zero_temp_al_benchmark.py

可选依赖
--------

项目提供多个可选依赖组：

.. code-block:: bash

   # 开发工具（测试、格式化、预提交钩子）
   uv pip install -e ".[dev]"

   # 或使用pip
   python -m pip install -e ".[dev]"

   # 文档构建（Sphinx、教程依赖）
   uv pip install -e ".[docs]"

**依赖组说明**:

- **dev**: pytest, ruff, pre-commit等开发工具
- **docs**: sphinx, myst-nb, jupyter等文档构建依赖

**主要依赖说明**:

- **核心计算**: numpy, scipy, pandas, scikit-learn
- **可视化**: matplotlib, plotly, kaleido
- **数据处理**: h5py, pyyaml
- **加速计算**: numba
- **C++绑定**: pybind11（构建时）

开发环境设置
--------

对于项目贡献者：

.. code-block:: bash

   # 1. Fork并克隆仓库
   git clone https://github.com/YOUR_USERNAME/ThermoElasticSim.git
   cd ThermoElasticSim

   # 2. 安装开发环境
   uv venv
   uv pip install -e ".[dev,docs]"

   # 3. 安装预提交钩子
   pre-commit install

   # 4. 运行完整测试
   python -m pytest -v
   # 或：uv run --no-sync python -m pytest -v

   # 5. 构建文档
   cd docs && make html

故障排除
--------

编译错误
~~~~~~~~

**问题**: C++编译失败
  - **macOS**: ``xcode-select --install``
  - **Ubuntu**: ``sudo apt install build-essential cmake``
  - **Windows**: 安装Visual Studio Build Tools

**问题**: CMake版本过低
  - 升级CMake至3.15+
  - 或使用conda: ``conda install cmake``

依赖冲突
~~~~~~~~

**问题**: numpy版本冲突
  - 项目需要numpy >= 2.0
  - 使用新的虚拟环境: ``uv venv --python 3.11``

**问题**: 可视化问题
  - 确保安装了kaleido: ``pip install kaleido``
  - 对于headless环境，设置: ``export MPLBACKEND=Agg``

虚拟环境问题
~~~~~~~~~~~~

**问题**: pytest找不到模块
  - 确认虚拟环境已激活
  - 使用 ``python -m pytest`` 而不是直接 ``pytest``
  - 检查 ``which python`` 和 ``which pytest`` 路径

**问题**: Conda base环境干扰
  - 禁用base自动激活: ``conda config --set auto_activate_base false``
  - 或使用完整路径: ``.venv/bin/python -m pytest``

性能问题
~~~~~~~~

**建议**: 确保使用C++后端
  - 项目自动检测C++扩展
  - 如果构建失败，会回退到纯Python（较慢）
  - 确保成功构建以获得最佳性能

**检查C++扩展**:

.. code-block:: bash

   python -c "import thermoelasticsim._cpp_core; print('C++ backend loaded')"

获取帮助
--------

如果遇到安装问题，请：

1. 检查 `GitHub Issues <https://github.com/bud-primordium/ThermoElasticSim/issues>`_
2. 搜索相似问题或创建新issue
3. 提供以下信息：

   - 操作系统版本
   - Python版本 (``python --version``)
   - 包管理器版本 (``uv --version`` 或 ``pip --version``)
   - 完整错误信息
   - 虚拟环境状态 (``which python``, ``pip list``)

4. 参与 `GitHub Discussions <https://github.com/bud-primordium/ThermoElasticSim/discussions>`_
