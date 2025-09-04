.. ThermoElasticSim documentation master file

===========================================
ThermoElasticSim 文档
===========================================

.. only:: html

   .. image:: https://img.shields.io/badge/version-4.0.0-blue
      :alt: Version

   .. image:: https://img.shields.io/badge/license-GPL--3.0-green
      :alt: License

欢迎使用ThermoElasticSim文档！

**ThermoElasticSim** 是一个用于计算材料弹性常数的分子动力学模拟教学软件。
它提供了零温和有限温度下的弹性常数计算功能，支持多种系综和恒温器算法。

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   quickstart
   conventions

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: 理论与方法

   01_elastic_fundamentals
   02_zero_temperature_elastic
   03_md_and_ensembles
   04_finite_temperature_elastic

.. toctree::
   :maxdepth: 2
   :caption: 拓展内容

   05_potential_extensions
   06_advanced_topics
   07_howto

.. toctree::
   :maxdepth: 3
   :caption: API参考

   api/core
   api/elastic
   api/md
   api/potentials
   api/interfaces
   api/utils

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
   - EAM嵌入原子势（铝、铜）
   - Lennard-Jones势
   - Tersoff势
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

    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential

    # 一键计算
    results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=(3, 3, 3)
    )

    # 输出结果
    print(f"C11 = {results['elastic_constants']['C11']:.1f} GPa")
    print(f"C12 = {results['elastic_constants']['C12']:.1f} GPa")
    print(f"C44 = {results['elastic_constants']['C44']:.1f} GPa")

获取帮助
========

* GitHub: https://github.com/bud-primordium/ThermoElasticSim
* 问题反馈: https://github.com/bud-primordium/ThermoElasticSim/issues
* 邮件: gilbertyoung0015@gmail.com

索引与搜索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

参考文献
========

.. bibliography:: references.bib
   :style: unsrt
   :all:
