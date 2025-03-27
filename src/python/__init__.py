# 文件名: __init__.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 初始化 ThermoElasticSim 项目，导入项目中的所有核心模块，
#          用于计算和模拟金属铝（Al）和金刚石（Diamond）的弹性常数。

"""
ThermoElasticSim 项目初始化模块

该模块提供项目核心功能的统一入口，主要包含以下功能：

1. 原子和晶胞结构定义 (Atom, Cell)
2. 势能函数 (Potential及其子类)
3. 积分器 (Integrator及其子类)
4. 恒温器和恒压器 (Thermostat, Barostat及其子类)
5. 应力和应变计算 (StressCalculator, StrainCalculator)
6. 弹性常数计算 (ElasticConstantsSolver)
7. 分子动力学模拟 (MDSimulator)
8. 数据可视化和分析 (Visualizer)

示例:
--------
>>> from python import Cell, Potential
>>> cell = Cell(...)
>>> potential = Potential(...)
"""

# 核心数据结构
from .structure import Atom, Cell

# 势能函数
from .potentials import (
    Potential,
    LennardJonesPotential,
    EAMAl1Potential,
)

# 数值积分器
from .integrators import (
    Integrator,
    VelocityVerletIntegrator,
    RK4Integrator,
)

# 恒温器
from .thermostats import (
    Thermostat,
    NoseHooverThermostat,
    NoseHooverChainThermostat,
)
from .barostats import Barostat, ParrinelloRahmanHooverBarostat
from .mechanics import (
    StressCalculator,
    StrainCalculator,
)
from .zeroelasticity import ElasticConstantsSolver, ElasticConstantsWorkflow
from .optimizers import Optimizer, GradientDescentOptimizer, BFGSOptimizer
from .deformation import Deformer
from .utils import TensorConverter, DataCollector, NeighborList
from .visualization import Visualizer
from .md_simulator import MDSimulator
from .config import ConfigManager
from .interfaces.cpp_interface import CppInterface
from .finite_temp_elasticity import FiniteTempElasticityWorkflow
