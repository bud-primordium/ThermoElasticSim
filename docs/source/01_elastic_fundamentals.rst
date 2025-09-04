==================
弹性常数基础理论
==================

本章介绍弹性常数的理论基础，包括线性弹性理论、能量二次型定义和立方晶系的特殊性质。

线性弹性理论
============

应力-应变关系
-------------

在小形变假设下，应力与应变呈线性关系（广义胡克定律）：

.. math::
   \sigma_{ij} = C_{ijkl} \varepsilon_{kl}

其中：

- :math:`\sigma_{ij}`：应力张量（二阶）
- :math:`\varepsilon_{kl}`：应变张量（二阶）
- :math:`C_{ijkl}`：弹性常数张量（四阶）

这一线性关系最早由Robert Hooke在1678年以拉丁文"ut tensio, sic vis"（拉伸如何，力亦如何）表述，后经Cauchy推广到连续介质的张量形式。弹性常数张量具有以下对称性：

.. math::
   C_{ijkl} = C_{jikl} = C_{ijlk} = C_{klij}

这些对称性将独立分量从81个减少到21个（最一般的各向异性材料）。

应变的严格定义
--------------

对于有限形变，需要使用Green-Lagrangian应变张量 :cite:`Born1954`：

.. math::
   E_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial X_j} + \frac{\partial u_j}{\partial X_i} + \sum_{k} \frac{\partial u_k}{\partial X_i} \frac{\partial u_k}{\partial X_j}\right)

其中 :math:`u_i` 为位移分量，:math:`X_j` 为参考构型坐标。在小应变假设下，二次项 :math:`\frac{\partial u_k}{\partial X_i} \frac{\partial u_k}{\partial X_j}` 远小于线性项，可以忽略，从而得到工程应变（无穷小应变）：

.. math::
   \varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial X_j} + \frac{\partial u_j}{\partial X_i}\right)

这一简化使得应变张量成为对称张量，是线性弹性理论成立的基础。

Voigt记号与剪切应变
-------------------

为简化张量运算，Voigt :cite:`Voigt1928` 提出将对称的6×6矩阵映射为6维向量。值得注意的是，Voigt记号中剪切应变分量定义为张量分量的两倍：

.. math::
   \varepsilon_4 = 2\varepsilon_{23}, \quad \varepsilon_5 = 2\varepsilon_{13}, \quad \varepsilon_6 = 2\varepsilon_{12}

这个因子2的引入并非任意，而是为了保持应变能密度表达式的简洁性。具体而言，应变能密度可以写为：

.. math::
   U = \frac{1}{2}C_{IJ}\varepsilon_I\varepsilon_J

其中求和遵循Einstein约定。若不引入因子2，则需要在剪切项中添加额外的系数，破坏了表达式的对称美。

能量二次型定义
==============

零温弹性常数
------------

在零温下，弹性常数可通过能量对应变的二阶导数定义：

.. math::
   C_{ijkl} = \frac{1}{V_0} \frac{\partial^2 E}{\partial \varepsilon_{ij} \partial \varepsilon_{kl}}

其中：

- :math:`E`：系统总能量
- :math:`V_0`：平衡体积
- 导数在零应变状态（平衡构型）处计算

从物理角度看，这一定义反映了晶格在平衡位置附近的恢复力特性。Born和黄昆 :cite:`Born1954` 指出，弹性常数本质上是原子间相互作用势在平衡位置的二阶导数，将微观的原子相互作用与宏观的弹性性质联系起来。

能量展开
--------

系统能量可按应变展开为Taylor级数：

.. math::
   E(\boldsymbol{\varepsilon}) = E_0 + V_0 \sigma_{ij}^0 \varepsilon_{ij} + \frac{V_0}{2} C_{ijkl} \varepsilon_{ij} \varepsilon_{kl} + O(\varepsilon^3)

在平衡态（:math:`\sigma_{ij}^0 = 0`），能量简化为：

.. math::
   \Delta E = \frac{V_0}{2} C_{ijkl} \varepsilon_{ij} \varepsilon_{kl}

这是弹性能的二次型表达式。二次型的正定性是晶体力学稳定的必要条件，即任意非零应变都应使能量增加。

立方晶系弹性常数
================

对称性约化
----------

立方晶系（如FCC、BCC、SC）属于高对称点群 :math:`O_h`，其48个对称操作将独立弹性常数减少到3个：

.. math::
   \begin{aligned}
   C_{11} &= C_{22} = C_{33} \\
   C_{12} &= C_{13} = C_{23} \\
   C_{44} &= C_{55} = C_{66}
   \end{aligned}

这一约化可通过群论严格证明：立方对称性要求弹性张量在所有立方对称操作下不变。

弹性矩阵形式
------------

使用Voigt记号，立方晶系的弹性矩阵为：

.. math::
   \mathbf{C} = \begin{bmatrix}
   C_{11} & C_{12} & C_{12} & 0 & 0 & 0 \\
   C_{12} & C_{11} & C_{12} & 0 & 0 & 0 \\
   C_{12} & C_{12} & C_{11} & 0 & 0 & 0 \\
   0 & 0 & 0 & C_{44} & 0 & 0 \\
   0 & 0 & 0 & 0 & C_{44} & 0 \\
   0 & 0 & 0 & 0 & 0 & C_{44}
   \end{bmatrix}

物理意义
--------

三个独立弹性常数的物理含义：

1. **C₁₁**：单轴应力下的纵向刚度

   .. math::
      C_{11} = \frac{\sigma_{11}}{\varepsilon_{11}} \quad \text{（其他应变为零）}

   从微观角度，C₁₁反映了沿晶轴方向原子键的刚度。

2. **C₁₂**：泊松效应的度量

   .. math::
      C_{12} = \frac{\sigma_{11}}{\varepsilon_{22}} \quad \text{（横向约束）}

   C₁₂描述了材料的横向耦合效应，与泊松比密切相关。

3. **C₄₄**：纯剪切刚度

   .. math::
      C_{44} = \frac{\sigma_{23}}{2\varepsilon_{23}} = \frac{\tau_{23}}{\gamma_{23}}

   C₄₄反映了晶体抵抗剪切形变的能力，与位错运动和塑性变形密切相关。

弹性模量关系
============

体积模量
--------

体积模量描述材料的抗压缩性：

.. math::
   B = \frac{C_{11} + 2C_{12}}{3}

对于立方晶系，这是精确关系。体积模量直接关联到晶体的状态方程和热力学性质。

剪切模量
--------

立方晶系有多个剪切模量定义：

1. C44：{100}面上的剪切模量
2. C'：{110}面上的剪切模量

   .. math::
      C' = \frac{C_{11} - C_{12}}{2}

   C'的软化常预示结构相变，如马氏体相变中的晶格失稳。

3. 各向同性平均（Voigt-Reuss-Hill）

   .. math::
      G_{VRH} = \frac{G_V + G_R}{2}

   其中Voigt平均假设应变均匀，Reuss平均假设应力均匀，Hill平均取两者算术平均。

杨氏模量和泊松比
----------------

通过弹性常数可计算等效各向同性参数：

.. math::
   E = \frac{9BG}{3B + G}, \quad \nu = \frac{3B - 2G}{2(3B + G)}

稳定性判据
==========

Born稳定性条件
--------------

立方晶系的力学稳定性要求 :cite:`Born1940`：

.. math::
   \begin{aligned}
   C_{11} - C_{12} &> 0 \\
   C_{11} + 2C_{12} &> 0 \\
   C_{44} &> 0
   \end{aligned}

这些条件确保弹性能为正定二次型。从物理上讲，第一个条件保证了晶体对四方畸变的稳定性，第二个条件保证了体积稳定性，第三个条件保证了剪切稳定性。违反这些条件意味着晶格动力学不稳定，系统将自发发生结构相变。

Cauchy关系
----------

对于中心力势，Cauchy关系成立：

.. math::
   C_{12} = C_{44}

实际材料中的偏离反映了多体效应的重要性。金属材料由于电子的离域特性，通常表现出显著的Cauchy偏离。

计算方法概述
============

应力-应变法
-----------

1. 对晶体施加一组小应变 :math:`\{\varepsilon^{(n)}\}`
2. 优化内部坐标（保持晶格固定）
3. 计算相应的应力 :math:`\{\sigma^{(n)}\}`
4. 通过线性拟合提取弹性常数

能量-应变法
-----------

1. 对晶体施加一组小应变
2. 完全优化结构能量
3. 拟合能量-应变关系的二次项系数

两种方法在理论上等价，但应力-应变法通常收敛更快，因为应力是能量的一阶导数，对数值误差更敏感。

形变模式选择
------------

对于立方晶系，典型的形变模式包括：

- **正交形变**：提取 :math:`C_{11} + C_{12}` 和 :math:`C_{11} - 2C_{12}`
- **单斜形变**：直接得到 :math:`C_{44}`
- **三角形变**：同时获得多个弹性常数组合

选择合适的形变模式可以减少所需的计算量并提高数值稳定性。

温度效应
========

准静态近似
----------

有限温度弹性常数通过时间平均应力定义：

.. math::
   C_{ijkl}(T) = \frac{1}{V(T)} \left\langle \frac{\partial \sigma_{ij}}{\partial \varepsilon_{kl}} \right\rangle_T

在有限温度下，热涨落导致瞬时应力的涨落，因此需要通过时间平均获得热力学应力。

热膨胀修正
----------

温度引起的晶格膨胀需要考虑：

.. math::
   a(T) = a_0[1 + \alpha(T-T_0)]

其中 :math:`\alpha` 是线性热膨胀系数。热膨胀使原子间距增大，导致有效相互作用减弱，表现为弹性常数的软化。

非谐效应
--------

高温下的非谐效应导致：

1. 弹性常数的温度软化
2. 声子-声子相互作用
3. 热涨落对原子力常数的重整化

这些效应超出了准谐近似的范畴，需要通过分子动力学模拟或高阶微扰理论处理。

计算实现
========

应力计算方法
------------

在ThermoElasticSim中，应力张量通过 :meth:`~thermoelasticsim.core.structure.Cell.calculate_stress_tensor` 方法计算：

.. math::
   \sigma_{\alpha\beta} = -\frac{1}{V} \left( \sum_i m_i v_{i,\alpha} v_{i,\beta} + \sum_i r_{i,\alpha} F_{i,\beta} \right)

零温下动能项为零，仅使用维里项。维里应力的物理意义是原子间相互作用力与位置的张量积，反映了内应力的微观起源。单位转换通过 :data:`~thermoelasticsim.utils.utils.EV_TO_GPA` 常数实现。

最小可执行示例
--------------

以下示例演示如何对FCC铝施加微小单轴应变并计算应力响应：

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.elastic.deformation import Deformer
    from thermoelasticsim.utils.utils import EV_TO_GPA

    # 创建3×3×3铝超胞
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )

    # 初始化势函数和形变器
    potential = EAMAl1Potential()
    deformer = Deformer(delta=0.005, num_steps=5)

    # 施加单轴应变εxx=0.005，构造形变梯度F = I + ε
    strain_xx = np.array([[0.005, 0, 0], [0, 0, 0], [0, 0, 0]])
    F = np.eye(3) + strain_xx  # 形变梯度矩阵
    deformed_cell = cell.copy()
    deformer.apply_deformation(deformed_cell, F)

    # 计算应力响应
    stress_tensor = deformed_cell.calculate_stress_tensor(potential)
    stress_xx_GPa = stress_tensor[0, 0] * EV_TO_GPA

    print(f"施加应变εxx=0.005，应力响应σxx={stress_xx_GPa:.2f} GPa")

此示例展示了理论到实现的映射：应变→形变梯度→应力→弹性常数。完整的弹性常数计算可通过 :func:`~thermoelasticsim.elastic.benchmark.run_zero_temp_benchmark` 一键完成。

小结
====

本章建立了弹性常数的理论框架：

- 线性弹性理论提供应力-应变关系
- 能量二次型给出微观定义
- 立方对称性简化到3个独立常数
- Born稳定性确保力学稳定
- 温度效应通过统计平均引入

下一章将介绍如何在零温下计算这些弹性常数的具体方法和工具。
