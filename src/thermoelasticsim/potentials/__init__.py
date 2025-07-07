"""
势能函数模块
"""

__all__ = ["Potential", "LennardJonesPotential", "EAMAl1Potential"]

# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "Potential":
        from .potentials import Potential
        return Potential
    elif name == "LennardJonesPotential":
        from .potentials import LennardJonesPotential
        return LennardJonesPotential
    elif name == "EAMAl1Potential":
        from .potentials import EAMAl1Potential
        return EAMAl1Potential
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")