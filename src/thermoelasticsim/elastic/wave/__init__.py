"""弹性波模块

本包提供弹性波传播的解析计算和动力学模拟功能。

功能组件
--------
解析计算：
- :class:`~thermoelasticsim.elastic.wave.analytical.ElasticWaveAnalyzer`

动力学模拟：
- :class:`~thermoelasticsim.elastic.wave.dynamics.WaveExcitation`
- :class:`~thermoelasticsim.elastic.wave.dynamics.DynamicsConfig`
- :func:`~thermoelasticsim.elastic.wave.dynamics.simulate_plane_wave_mvp`

可视化：
- :func:`~thermoelasticsim.elastic.wave.visualization.plot_polar_plane`
- :func:`~thermoelasticsim.elastic.wave.visualization.plot_velocity_surface_3d`
"""

from .analytical import ElasticWaveAnalyzer
from .dynamics import DynamicsConfig, WaveExcitation, simulate_plane_wave_mvp
from .visualization import plot_polar_plane, plot_velocity_surface_3d

__all__ = [
    "ElasticWaveAnalyzer",
    "WaveExcitation",
    "DynamicsConfig",
    "simulate_plane_wave_mvp",
    "plot_polar_plane",
    "plot_velocity_surface_3d",
]
