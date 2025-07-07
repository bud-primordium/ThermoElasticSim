"""
弹性常数计算模块
"""

# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "ElasticConstantsSolver":
        from .zeroelasticity import ElasticConstantsSolver
        return ElasticConstantsSolver
    elif name == "ElasticConstantsWorkflow":
        from .zeroelasticity import ElasticConstantsWorkflow
        return ElasticConstantsWorkflow
    elif name == "FiniteTempElasticityWorkflow":
        from .finite_temp_elasticity import FiniteTempElasticityWorkflow
        return FiniteTempElasticityWorkflow
    elif name == "Deformer":
        from .deformation import Deformer
        return Deformer
    elif name == "StrainCalculator":
        from .deformation import StrainCalculator
        return StrainCalculator
    elif name == "StressCalculator":
        from .mechanics import StressCalculator
        return StressCalculator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "ElasticConstantsSolver", "ElasticConstantsWorkflow",
    "FiniteTempElasticityWorkflow",
    "Deformer", "StrainCalculator", "StressCalculator"
]