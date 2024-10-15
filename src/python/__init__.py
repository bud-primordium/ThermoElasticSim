# src/python/__init__.py

"""
@package ThermoElasticSim
@brief 初始化ThermoElasticSim项目。

该项目包含用于计算和模拟金属铝（Al）和金刚石（Diamond）在不同温度下弹性常数的模块。
"""

# 导入子模块（可选）
from .structure import CrystalStructure, Atom
from .potentials import Potential, EAMPotential, LennardJonesPotential
from .optimizers import (
    StructureOptimizer,
    ConjugateGradientOptimizer,
    NewtonRaphsonOptimizer,
)
from .deformation import Deformer
from .stress_evaluator import (
    StressEvaluator,
    EAMStressEvaluator,
    LennardJonesStressEvaluator,
)
from .strain import StrainCalculator
from .solver import ElasticConstantSolver
from .md_simulation import MDSimulator
from .visualization import Visualizer
from .utilities import IOHandler, TensorConverter
from .config import ConfigManager
