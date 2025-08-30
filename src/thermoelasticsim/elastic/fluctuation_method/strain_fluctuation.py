#!/usr/bin/env python3
r"""
ThermoElasticSim - 应变涨落法模块

.. moduleauthor:: Gilbert Young

该模块将实现基于应变涨落理论的弹性常数计算方法。

此方法通过在NPT系综下长时间模拟，分析模拟盒子（应变）的自发涨落，
来计算材料的弹性常数。

.. math::
    C_{ijkl} = \frac{k_B T}{\langle V \rangle} \left[ \langle \epsilon_{ij} \epsilon_{kl} \rangle - \langle \epsilon_{ij} \rangle \langle \epsilon_{kl} \rangle \right]^{-1}

参考文献:
    Clavier, G., et al. (2017). Computation of elastic constants of solids
    using molecular simulation. Molecular Simulation, 43(13-16), 1133-1145.

.. warning::
    根据 Clavier et al. 的研究，此方法在MD模拟中对控压器参数极其敏感，
    可能导致结果不准确，但在MC模拟中表现良好。实现时需特别注意。
"""

# --- 模块内容将在未来实现 ---
