# src/python/__init__.py

"""
@package ThermoElasticSim
@brief 初始化 ThermoElasticSim 项目。

该项目包含用于计算和模拟金属铝（Al）和金刚石（Diamond）在不同温度下弹性常数的模块。
"""

# 导入子模块
from .structure import Atom, Cell
from .potentials import Potential, LennardJonesPotential
from .md import (
    MDSimulator,
    Integrator,
    VelocityVerletIntegrator,
    Thermostat,
    NoseHooverThermostat,
)
from .mechanics import (
    StressCalculator,
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from .optimizers import Optimizer, GradientDescentOptimizer
from .deformation import Deformer
from .utils import TensorConverter, DataCollector
from .visualization import Visualizer
from .config import ConfigManager
