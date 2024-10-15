# ThermoElasticSim

## 项目简介

**ThermoElasticSim** 是一个用于计算金属铝（Al）和绝缘体金刚石（Diamond）在零温和有限温度下弹性常数的模拟工具。该项目结合了Python的易用性和Fortran的高性能计算能力，通过结构优化和分子动力学（MD）模拟，实现了应力-应变方法和能量-体积（E-V）拟合法，全面研究材料的弹性性质。此外，项目还支持弹性波在不同晶向上的传播特性分析。

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
├── src/
│   ├── python/
│   │   ├── __init__.py
│   │   ├── structure.py
│   │   ├── potentials.py
│   │   ├── optimizers.py
│   │   ├── deformation.py
│   │   ├── stress_evaluator.py
│   │   ├── strain.py
│   │   ├── solver.py
│   │   ├── md_simulation.py
│   │   ├── visualization.py
│   │   ├── utilities.py
│   │   └── config.py
│   ├── fortran/
│   │   ├── stress_evaluator.f90
│   │   ├── structure_optimizer.f90
│   │   └── Makefile
│   └── docs/
│       └── Doxyfile
├── tests/
│   ├── test_structure.py
│   ├── test_potentials.py
│   ├── test_optimizers.py
│   ├── test_deformation.py
│   ├── test_stress_evaluator.py
│   ├── test_strain.py
│   ├── test_solver.py
│   ├── test_md_simulation.py
│   ├── test_visualization.py
│   ├── test_utilities.py
│   └── test_config.py
├── examples/
│   ├── zero_temperature_elastic_constants.py
│   ├── finite_temperature_elastic_constants.py
│   └── wave_propagation_simulation.py
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

## 安装指南

### 前置条件

- **Python 3.8+**
- **Fortran 编译器**（如 `gfortran`）
- **pip** 包管理工具

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
   pip install -r requirements.txt
   ```

4. **编译 Fortran 模块**

   ```bash
   cd src/fortran
   make
   cd ../../
   ```

## 使用说明

### 零温下弹性常数计算

```bash
python examples/zero_temperature_elastic_constants.py
```

### 有限温度下弹性常数计算

```bash
python examples/finite_temperature_elastic_constants.py
```

### 弹性波传播模拟

```bash
python examples/wave_propagation_simulation.py
```

## 配置文件

项目使用 `YAML` 格式的配置文件 `config.yaml` 来管理参数。以下是一个示例配置：

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
    # 添加更多原子

potential:
  type: 'EAM'  # 或 'LennardJones'
  parameters:
    epsilon: 0.0103  # 示例参数
    sigma: 3.405      # 示例参数
  cutoff: 5.0

optimizer:
  method: 'ConjugateGradient'  # 或 'NewtonRaphson'

deformation:
  delta: 0.01

stress_evaluator:
  type: 'EAM'  # 或 'LennardJones'

md_simulation:
  temperature: 300  # K
  pressure: 0.0      # GPa
  timestep: 1.0e-3    # ps
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

如有任何问题或建议，请通过 [issue](https://github.com/bud-primordium/ThermoElasticSim/issues) 与我们联系。
