#!/usr/bin/env python3
"""
ThermoElasticSim - 势能模块

这是一个包，提供了各种用于分子动力学模拟的原子间势。
采用延迟导入模式以避免循环依赖并提高加载性能。

.. moduleauthor:: Gilbert Young
.. created:: 2024-10-14
.. modified:: 2025-07-07
.. version:: 4.0.0
"""

# 1. 定义公开接口
__all__ = [
    "Potential",
    "LennardJonesPotential",
    "EAMAl1Potential",
    "TersoffPotential",
    "MLPotential",
]

# 2. 使用 __getattr__ 实现延迟加载
def __getattr__(name):
    if name == "Potential":
        from .base import Potential
        return Potential
    elif name == "LennardJonesPotential":
        from .lennard_jones import LennardJonesPotential
        return LennardJonesPotential
    elif name == "EAMAl1Potential":
        from .eam import EAMAl1Potential
        return EAMAl1Potential
    elif name == "TersoffPotential":
        from .tersoff import TersoffPotential
        return TersoffPotential
    elif name == "MLPotential":
        from .mlp import MLPotential
        return MLPotential
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
