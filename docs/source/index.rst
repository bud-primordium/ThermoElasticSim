.. ThermoElasticSim documentation master file

===========================================
ThermoElasticSim 文档
===========================================

.. image:: https://img.shields.io/badge/version-4.0.0-blue
   :alt: Version

.. image:: https://img.shields.io/badge/license-GPL--3.0-green
   :alt: License

欢迎使用ThermoElasticSim文档！

**ThermoElasticSim** 是一个用于计算材料弹性常数的分子动力学模拟教学软件。
它提供了零温和有限温度下的弹性常数计算功能，支持多种系综和恒温器算法。

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   quickstart/installation
   quickstart/first_simulation
   quickstart/examples

.. toctree::
   :maxdepth: 2
   :caption: 用户指南

   guide/basic_concepts
   guide/elastic_calculation
   guide/temperature_control
   guide/advanced_features

.. toctree::
   :maxdepth: 2
   :caption: 理论背景

   theory/md_fundamentals
   theory/ensemble_theory
   theory/elastic_theory
   theory/integrators

.. toctree::
   :maxdepth: 3
   :caption: API参考

   api/core
   api/elastic
   api/md
   api/potentials
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   development/contributing
   development/code_style
   development/testing

特性概览
========

核心功能
--------

* **弹性常数计算**
   - 零温显式形变法
   - 有限温度形变法
   - 支持立方晶系 (C11, C12, C44)

* **分子动力学引擎**
   - NVE微正则系综
   - NVT正则系综（多种恒温器）
   - NPT等温等压系综（MTK算法）

* **势函数支持**
   - EAM嵌入原子势（铝）
   - Lennard-Jones势
   - 可扩展势函数接口

* **恒温器算法**
   - Berendsen恒温器
   - Andersen随机碰撞
   - Langevin动力学
   - Nosé-Hoover链

技术特点
--------

* 基于算符分离的模块化架构
* JIT编译优化的关键算法
* C++扩展加速的势函数计算
* 完整的单元测试覆盖
* NumPy风格的文档

快速示例
========

计算FCC铝的弹性常数::

    from thermoelasticsim.core.structure import Cell
    from thermoelasticsim.elastic.deformation_method import ZeroTempElasticCalculator
    from thermoelasticsim.potentials import EAMAl1Potential
    import numpy as np

    # 创建FCC铝晶胞
    a = 4.05  # 晶格常数
    lattice = a * np.eye(3)
    atoms = create_fcc_atoms("Al", a)
    cell = Cell(lattice, atoms)

    # 设置势函数
    potential = EAMAl1Potential()

    # 计算弹性常数
    calculator = ZeroTempElasticCalculator(potential)
    C11, C12, C44 = calculator.calculate(cell)

    print(f"C11 = {C11:.1f} GPa")
    print(f"C12 = {C12:.1f} GPa")
    print(f"C44 = {C44:.1f} GPa")

获取帮助
========

* GitHub: https://github.com/yourusername/ThermoElasticSim
* 问题反馈: https://github.com/yourusername/ThermoElasticSim/issues
* 邮件: gilbertyoung0015@gmail.com

索引与搜索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
