==================
势函数拓展
==================

本章介绍 ThermoElasticSim 已有与计划支持的势函数类型。

Tersoff势（已实现 Carbon 1988 :cite:`PhysRevLett.61.2879`）
===========================================================

实现状态
--------

- 已实现 Tersoff C(1988) 对碳（diamond）的参数化：能量、力、维里三者均由 C++ 后端解析计算 :cite:`PhysRevLett.61.2879`。
- 三体维里采用团簇分解（成对 + 三元组）进行记账，保证与拉伸为正的应力约定一致。

基本形式
--------

.. math::
   V = \frac{1}{2}\sum_{i\ne j} f_C(r_{ij})\,\big[f_R(r_{ij}) + b_{ij}\,f_A(r_{ij})\big]

其中本实现采用如下记号（与代码一致）：

- :math:`f_R(r) = A\,e^{-\lambda_1 r}` （排斥）
- :math:`f_A(r) = -B\,e^{-\lambda_2 r}` （吸引，注意负号在函数里）
- :math:`f_C(r)` 为平滑截断函数：

  .. math::
     f_C(r)=\begin{cases}
       1,& r<R-D\\
       0.5\bigl(1-\sin[\tfrac{\pi}{2}\tfrac{(r-R)}{D}]\bigr),& |r-R|\le D\\
       0,& r>R+D
     \end{cases}

- 键序项 :math:`b_{ij}` 与局部配位相关：

  .. math::
     b_{ij}(\zeta)=\bigl(1+ (\beta^n)\,\zeta^n\bigr)^{-\tfrac{1}{2n}},\qquad
     \zeta_{ij}=\sum_{k\ne i,j} f_C(r_{ik})\,g(\cos\theta_{ijk})\,\exp\bigl[\lambda_3^{m}(r_{ij}-r_{ik})^{m}\bigr]

  其中角函数：

  .. math::
     g(c)=1+\frac{c^2}{d^2}-\frac{c^2}{d^2+(h-c)^2}

  本实现支持 :math:`m=3` 的常见情形；C(1988) 参数通常 :math:`\lambda_3=0`。

多体维里与应力（拉伸为正）
--------------------------

应力定义为张拉为正：

.. math::
   \sigma = -\frac{1}{V}\sum r\otimes F\,.

本实现将维里分解为以下可累计项：

- 两体斥力对：:math:`\;\Delta W_{2,\mathrm{rep}} = (r_j-r_i)\otimes F_{ij}`
- 键序配对项（由 :math:`b_{ij}` 产生的配对力）：

  :math:`\;\Delta W_{2,\mathrm{zeta}} = -\,(r_j-r_i)\otimes F^{(\mathrm{pair})}_{ij}`

  （注意负号）
- 三体吸引项：:math:`\;\Delta W_{3} = (r_j-r_i)\otimes F_j + (r_k-r_i)\otimes F_k`

最终 :math:`\sigma` 由 :math:`W` 取负并除体积得到。该记账方式与主流实现一致，数值上对称收敛良好。

适用材料
--------

- 硅（Si）：半导体器件
- 碳（C）：金刚石、石墨烯
- 锗（Ge）：光电材料
- SiC、SiGe等化合物

Stillinger-Weber势
==================

设计理念
--------

SW势专为硅的四面体结构设计，显式包含三体项：

.. math::
   V = \sum_{i<j} v_2(r_{ij}) + \sum_{i,j<k} v_3(r_{ij}, r_{ik}, \theta_{jik})

三体项稳定109.47°的四面体角：

.. math::
   v_3 = \lambda \exp\left[\frac{\gamma}{r_{ij}-a} + \frac{\gamma}{r_{ik}-a}\right] (\cos\theta_{jik} + 1/3)^2

特点优势
--------

- 精确描述Si的结构相变
- 计算效率高于Tersoff
- 熔化性质合理

机器学习势
==========

发展趋势
--------

机器学习势函数是当前材料模拟的前沿方向，结合了DFT精度和经典势效率。

主要类型
--------

**神经网络势（NNP）**
  - Behler-Parrinello网络
  - DeepMD
  - 特点：通用性强，需要大量训练数据

**高斯近似势（GAP）**
  - 基于高斯过程回归
  - SOAP描述符
  - 特点：不确定性量化

**图神经网络势（GNN）**
  - 消息传递机制
  - E(3)等变性
  - 特点：自然处理多体相互作用

**矩张量势（MTP）**
  - 基于矩不变量
  - 系统化展开
  - 特点：可解释性好

集成考虑
--------

未来集成机器学习势的技术路线：

1. 统一接口设计
2. 模型文件管理
3. GPU加速支持
4. 不确定性量化

注意事项
========

当前限制与提示
----------------

- Tersoff 已提供 C(1988) 参数化版本（碳）；其他材料参数可按相同接口扩展。
- 统一采用张拉为正的应力约定；请在阅读公式时留意符号一致性（本章以代码实现为准）。

开发计划
--------

势函数拓展的优先级（仅供参考）：

1. Tersoff势（硅材料需求）
2. 机器学习势接口（通用框架）
3. Stillinger-Weber势
4. 其他专用势函数

贡献指南
--------

欢迎社区贡献新势函数实现。基本要求：

- 继承Potential基类
- 实现calculate_forces和calculate_energy方法
- 提供完整的单元测试
- 包含文献引用和验证数据
