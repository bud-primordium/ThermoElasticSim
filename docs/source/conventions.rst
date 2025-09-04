==============
术语与约定
==============

本文档定义ThermoElasticSim中使用的术语、符号约定和单位系统。

小应变线性化前提
================

本软件采用 **小应变线性化**（非保体积）作为全局假设：

.. math::
   \mathbf{F} = \mathbf{I} + \boldsymbol{\varepsilon}

其中：

- :math:`\mathbf{F}`：形变梯度张量
- :math:`\mathbf{I}`：单位张量
- :math:`\boldsymbol{\varepsilon}`：应变张量（小量）

**关键假设**：

1. 二阶及更高阶小量被忽略：:math:`\boldsymbol{\varepsilon}^2 \approx 0`
2. 体积变化为高阶小量：:math:`\det(\mathbf{F}) \approx 1 + \text{tr}(\boldsymbol{\varepsilon})`
3. 应力-应变关系线性化：:math:`\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}`

单位系统
========

ThermoElasticSim采用原子单位与实用单位混合系统：

基本单位
--------

===================== ============ =================
物理量                 单位         说明
===================== ============ =================
长度                   Å            埃（Angstrom）
能量                   eV           电子伏特
时间                   fs           飞秒
质量                   amu          原子质量单位
温度                   K            开尔文
===================== ============ =================

导出单位
--------

===================== ======================== =================
物理量                 内部单位                  输出单位
===================== ======================== =================
力                     eV/Å                     同内部单位
速度                   Å/fs                     同内部单位
应力                   eV/Å³                    GPa（输出时转换）
弹性常数               eV/Å³                    GPa（输出时转换）
压强                   eV/Å³                    GPa（输出时转换）
===================== ======================== =================

单位转换
--------
:data:`~thermoelasticsim.utils.utils.EV_TO_GPA`
:data:`~thermoelasticsim.utils.utils.KB_IN_EV`
:data:`~thermoelasticsim.utils.utils.AMU_TO_EVFSA2`

Voigt记号约定
=============

弹性常数张量和应力/应变张量采用Voigt记号简化表示。

张量-向量映射
-------------

四阶张量 :math:`C_{ijkl}` → 6×6矩阵 :math:`C_{IJ}`：

.. math::
   \begin{array}{c|c}
   \text{张量指标} & \text{Voigt指标} \\
   \hline
   11 & 1 \\
   22 & 2 \\
   33 & 3 \\
   23, 32 & 4 \\
   13, 31 & 5 \\
   12, 21 & 6
   \end{array}

工程剪切应变
------------

**重要约定**：Voigt记号中的剪切应变是工程剪切应变，为张量剪切应变的2倍：

.. math::
   \begin{aligned}
   \varepsilon_1 &= \varepsilon_{11} \\
   \varepsilon_2 &= \varepsilon_{22} \\
   \varepsilon_3 &= \varepsilon_{33} \\
   \varepsilon_4 &= 2\varepsilon_{23} = \gamma_{23} \quad \text{（工程剪切）} \\
   \varepsilon_5 &= 2\varepsilon_{13} = \gamma_{13} \\
   \varepsilon_6 &= 2\varepsilon_{12} = \gamma_{12}
   \end{aligned}

立方晶系弹性常数
----------------

对于立方晶系（如FCC铝、铜），独立弹性常数仅有3个：

.. math::
   \mathbf{C} = \begin{bmatrix}
   C_{11} & C_{12} & C_{12} & 0 & 0 & 0 \\
   C_{12} & C_{11} & C_{12} & 0 & 0 & 0 \\
   C_{12} & C_{12} & C_{11} & 0 & 0 & 0 \\
   0 & 0 & 0 & C_{44} & 0 & 0 \\
   0 & 0 & 0 & 0 & C_{44} & 0 \\
   0 & 0 & 0 & 0 & 0 & C_{44}
   \end{bmatrix}

其中：

- :math:`C_{11}`：纵向弹性常数
- :math:`C_{12}`：横向耦合常数
- :math:`C_{44}`：剪切弹性常数

坐标系约定
==========

晶格向量和原子位置
------------------

- 晶格向量按**行向量**存储在3×3矩阵中
- 第i个晶格向量：``lattice_vectors[i, :]``
- 原子位置为笛卡尔坐标（Å）

周期性边界条件（PBC）
---------------------

最小镜像约定（Minimum Image Convention, MIC）：

.. code-block:: python

    def apply_mic(r_ij, cell):
        """应用最小镜像约定"""
        # 转换到分数坐标
        s_ij = np.dot(r_ij, cell.lattice_inv)
        # 映射到[-0.5, 0.5]
        s_ij -= np.floor(s_ij + 0.5)
        # 转回笛卡尔坐标
        return np.dot(s_ij, cell.lattice_vectors.T)

矩阵运算约定
============

形变施加
--------

形变通过右乘形变矩阵实现：

.. math::
   \mathbf{h}' = \mathbf{h} \cdot \mathbf{F}

其中：

- :math:`\mathbf{h}`：原始晶格向量矩阵（3×3）
- :math:`\mathbf{F}`：形变梯度张量（3×3）
- :math:`\mathbf{h}'`：形变后晶格向量矩阵

应力计算
--------

应力张量定义为：

.. math::
   \sigma_{ij} = -\frac{1}{V} \frac{\partial E}{\partial \varepsilon_{ij}}

计算方法：

1. **Virial方法**：通过原子力和位置计算
2. **有限差分**：通过能量对应变的数值导数

数据结构约定
============

Cell对象
--------
:py:class:`~thermoelasticsim.core.structure.Cell`
.. code-block:: python

    cell.lattice_vectors  # 3×3矩阵，行向量
    cell.atoms            # 原子列表
    cell.volume           # 晶胞体积（Å³）
    cell.num_atoms        # 原子数

Atom对象
--------
:py:class:`~thermoelasticsim.core.structure.Atom`
.. code-block:: python

    atom.position         # 3D向量（Å）
    atom.velocity         # 3D向量（Å/fs）
    atom.force           # 3D向量（eV/Å）
    atom.mass_amu            # 质量（amu）
    atom.symbol          # 元素符号（如'Al'）

文献引用约定
============

- 使用sphinxcontrib-bibtex管理文献
- 引用格式：``:cite:`key```
- 文献库：``references.bib``
- 章末集中显示引用列表
