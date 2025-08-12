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
├── pyproject.toml          # 项目配置（依赖管理、构建设置）
├── CMakeLists.txt          # C++ 构建配置
├── README.md
├── src/
│   └── thermoelasticsim/
│       ├── _cpp/           # C++ 源码和 pybind11 绑定
│       │   ├── bindings/   # 模块化的绑定文件
│       │   └── *.cpp       # 核心计算实现
│       ├── core/           # 核心数据结构
│       ├── potentials/     # 势能模型
│       ├── elastic/        # 弹性常数计算
│       ├── md/             # 分子动力学
│       └── utils/          # 工具函数
├── tests/                  # 测试文件（镜像 src 结构）
├── examples/               # 使用示例
└── docs/                   # 文档
```

## 安装指南

### 前置条件

- **Python 3.9+**
- **C++ 编译器**（支持 C++11）
- **uv**（推荐）或 pip

### 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/bud-primordium/ThermoElasticSim.git
cd ThermoElasticSim

# 2. 清理旧的构建产物（如果有）
rm -rf .venv build .cmake src/thermoelasticsim/_cpp_core*.so

# 3. 创建虚拟环境
uv venv

# 4. 安装项目（editable 模式）
uv pip install -e .

# 5. 安装测试依赖（如需运行测试）
uv pip install pytest

# 6. 验证安装
uv run python -c "import thermoelasticsim._cpp_core; print('✓ 安装成功')"
```

### 运行测试

```bash
uv run pytest
```

### 安装说明

- **自动化构建**：使用 `scikit-build-core` + `pybind11` 自动编译 C++ 扩展
- **已知问题**：由于 scikit-build-core 与 uv 的兼容性问题，目前只支持 editable 安装（`-e`）
- **重要提示**：运行任何命令都需要加 `uv run` 前缀，确保使用正确的 Python 环境

## 使用说明

### 运行示例

```bash
uv run python examples/calculate_elastic_constants.py
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/potentials/

# 显示详细输出
uv run pytest -v
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
