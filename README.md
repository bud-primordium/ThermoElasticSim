# ThermoElasticSim

## 项目简介

**ThermoElasticSim** 是一个用于计算金属铝（Al）和绝缘体金刚石（Diamond）在零温和有限温度下弹性常数的模拟工具。该项目结合了 Python 的易用性和 C++ 的高性能计算能力，通过结构优化和分子动力学（MD）模拟，采用显式变形法和应力涨落法，全面研究材料的弹性性质。此外，支持弹性波在不同晶向上的传播特性分析。

## 主要功能

1. **零温下弹性常数计算**
   - 确定平衡晶格构型
   - 施加独立应变，计算应力
   - 建立应力-应变关系，求解弹性常数

2. **有限温度下弹性常数计算**
   - 通过分子动力学（MD）模拟确定有限温度下的平衡晶格构型
   - 施加应变，计算应力并进行统计平均
   - 建立应力-应变关系，求解温度依赖的弹性常数

3. **弹性波传播模拟**
   - 模拟不同晶向（[100]、[110]、[111]）上的横波和纵波传播
   - 分析弹性波的传播特性

4. **应力-应变关系分析**
   - 模拟不同应变大小对应的应力，分析胡克定律的适用范围

## 目录结构

```plaintext
ThermoElasticSim/
├── LICENSE
├── Makefile
├── Makefile.libs
├── README.md
├── build/                   # Sphinx 构建输出目录
├── config/                  # 配置文件
│   └── pytest.ini
├── docs/                    # 文档生成目录，包含 Doxygen 和 Sphinx 配置
│   ├── Doxyfile
│   ├── custom.css
│   └── xml/
│       ├── combine.xslt
│       ├── compound.xsd
│       ├── doxyfile.xsd
│       ├── index.xsd
│       └── xml.xsd
├── examples/                # 使用示例
│   └── calculate_elastic_constants.py
├── logs/                    # 日志文件
├── scripts/                 # 脚本文件，自动化任务
│   ├── build_libs.bat
│   ├── collect.py
│   ├── datetime_updater.py
│   └── make.bat
├── setup.py                 # 项目安装脚本
├── source/                  # Sphinx 文档的源文件
│   ├── _static
│   ├── _templates
│   ├── conf.py
│   └── index.rst
├── src/                     # 项目源代码
│   ├── ThermoElasticSim.egg-info/
│   │   └── PKG-INFO
│   ├── cpp/                 # C++ 文件实现物理计算
│   │   ├── lennard_jones.cpp
│   │   ├── nose_hoover.cpp
│   │   ├── nose_hoover_chain.cpp
│   │   ├── parrinello_rahman_hoover.cpp
│   │   └── stress_calculator.cpp
│   ├── lib/                 # 动态库，编译后的 C++ 文件
│   │   ├── lennard_jones.dll
│   │   ├── nose_hoover.dll
│   │   ├── nose_hoover_chain.dll
│   │   ├── parrinello_rahman_hoover.dll
│   │   └── stress_calculator.dll
│   └── python/              # Python 代码，用于控制流程和集成 C++ 计算
│       ├── __init__.py
│       ├── __pycache__
│       ├── barostats.py
│       ├── config.py
│       ├── deformation.py
│       ├── elasticity.py
│       ├── integrators.py
│       ├── interfaces/
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   └── cpp_interface.py
│       ├── md_simulator.py
│       ├── mechanics.py
│       ├── optimizers.py
│       ├── ploy_lj.py
│       ├── potentials.py
│       ├── structure.py
│       ├── thermostats.py
│       ├── utils.py
│       └── visualization.py
└── tests/                   # 自动化测试
    ├── __pycache__
    ├── config.yaml
    ├── test_cpp_interface.py
    ├── test_deformation.py
    ├── test_elasticity.py
    ├── test_lj.py
    ├── test_md.py
    ├── test_mechanics.py
    ├── test_nose_hoover.py
    ├── test_optimizers.py
    ├── test_pbc.py
    ├── test_potentials.py
    ├── test_simple_optimizer.py
    ├── test_structure.py
    ├── test_utils.py
    └── test_visualization.py
```

## 安装指南

### 前置条件

- **Python 3.8+**
- **C++ 编译器**（如 `g++` 或 `clang++`）
- **pip** 包管理工具
- **make** 工具（Windows 用户可使用 MinGW 或 MSYS2）

### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/bud-primordium/ThermoElasticSim.git
   cd ThermoElasticSim
   ```

2. **创建并激活虚拟环境**

   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
   ```

3. **安装依赖**

   ```bash
   make install-deps
   ```

4. **编译 C++ 模块**

   使用 `Makefile.libs` 编译 C++ 文件并生成动态库：

   ```bash
   make build_libs
   ```

## 使用说明

### 运行示例

当前提供了一个示例脚本，用于计算弹性常数：

```bash
python examples/calculate_elastic_constants.py
```

该脚本将根据实现的算法进行弹性常数的计算。更多功能正在开发中，后续将提供更多示例。

### 文档生成

项目使用 Sphinx 和 Doxygen 生成详细的文档。通过 `Makefile` 可以自动化文档的生成过程：

```bash
make html
```

生成的文档位于 `build/` 目录中。

### 运行测试

自动化测试使用 pytest 进行管理，通过 `Makefile` 可以方便地运行所有测试：

```bash
make test
```

## 配置文件

项目使用 `YAML` 格式的配置文件 `tests/config.yaml` 来管理参数。以下是一个示例配置（**尚在测试中**）：

```yaml
# config.yaml

unit_cell:
  lattice_vectors: 
    - [3.615, 0.0, 0.0]
    - [0.0, 3.615, 0.0]
    - [0.0, 0.0, 3.615]
  atoms:
    - {symbol: 'Al', mass: 26.9815, position: [0.0, 0.0, 0.0]}
    - {symbol: 'Al', mass: 26.9815, position: [1.8075, 1.8075, 1.8075]}

potential:
  type: 'LennardJones'
  parameters:
    epsilon: 0.0103
    sigma: 3.405
  cutoff: 5.0

optimizer:
  method: 'ConjugateGradient'

deformation:
  delta: 0.01

stress_evaluator:
  type: 'LennardJones'

md_simulation:
  temperature: 300
  pressure: 0.0
  timestep: 1.0e-3
  thermostat: 'Nosé-Hoover'
  barostat: 'NoBarostat'
  steps: 10000
```

## 贡献指南

欢迎贡献代码、报告问题或提出建议！请遵循以下步骤：

1. **Fork 仓库**
2. **创建分支**

   ```bash
   git checkout -b feature/新功能名称
   ```

3. **提交更改**

   ```bash
   git commit -m "添加了新功能"
   ```

4. **推送到分支**

   ```bash
   git push origin feature/新功能名称
   ```

5. **创建 Pull Request**

## 许可证

本项目采用 [GNU GPLV3 许可证](LICENSE) 进行授权。有关详细信息，请参阅 `LICENSE` 文件。

## 联系方式

如有任何问题或建议，请通过 [issue](https://github.com/bud-primordium/ThermoElasticSim/issues) 与项目维护者联系。
