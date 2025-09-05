"""弹性波模块

本包提供基于弹性常数的弹性波解析计算（阶段A）以及后续
动力学传播模拟（阶段B，待实现）。

当前可用组件：
- :class:`~thermoelasticsim.elastic.wave.analytical.ElasticWaveAnalyzer`
"""

from .analytical import ElasticWaveAnalyzer

__all__ = ["ElasticWaveAnalyzer"]
