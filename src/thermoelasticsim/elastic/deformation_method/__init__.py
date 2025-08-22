#!/usr/bin/env python3
"""
显式形变法子包

该子包实现了基于显式形变的弹性常数计算方法，包括零温和有限温两种情况。

.. moduleauthor:: Gilbert Young
.. created:: 2025-07-08
.. modified:: 2025-07-11
.. version:: 4.0.0
"""

# 零温显式形变法模块
from .zero_temp import (
    DeformationResult,
    ElasticConstantsSolver,
    StructureRelaxer,
    ZeroTempDeformationCalculator,
    calculate_zero_temp_elastic_constants,
)

# 有限温显式形变法模块（待实现）
try:
    from .finite_temp import FiniteTempElasticityWorkflow
except ImportError:
    # 如果finite_temp模块还未实现，创建占位符
    class FiniteTempElasticityWorkflow:
        """有限温弹性常数计算工作流（待实现）"""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("有限温显式形变法尚未实现")

# 为了向后兼容，保留旧的类名作为别名
ElasticConstantsWorkflow = ZeroTempDeformationCalculator

__all__ = [
    # 新的三层架构
    "StructureRelaxer",
    "ZeroTempDeformationCalculator",
    "ElasticConstantsSolver",
    "DeformationResult",
    "calculate_zero_temp_elastic_constants",
    # 向后兼容
    "ElasticConstantsWorkflow",
    "FiniteTempElasticityWorkflow",
]
