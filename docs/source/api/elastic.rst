弹性常数计算模块 (elastic)
===========================

.. currentmodule:: thermoelasticsim.elastic

本模块提供材料弹性常数的计算功能。

显式形变法
----------

零温计算
~~~~~~~~

.. automodule:: thermoelasticsim.elastic.deformation_method.zero_temp
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, Potential, NeighborList
   :noindex:

有限温计算
~~~~~~~~~~

.. automodule:: thermoelasticsim.elastic.deformation_method.finite_temp
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, Potential, NeighborList
   :noindex:

基础工具与材料参数
------------------

.. automodule:: thermoelasticsim.elastic.deformation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: thermoelasticsim.elastic.materials
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ALUMINUM_FCC, COPPER_FCC, GOLD_FCC

材料常量
--------

预定义示例：`ALUMINUM_FCC`, `COPPER_FCC` 等（详见模块源码）。为便于交叉引用，以下显式导出常量：

.. autodata:: thermoelasticsim.elastic.materials.ALUMINUM_FCC
   :no-value:

.. autodata:: thermoelasticsim.elastic.materials.COPPER_FCC
   :no-value:

.. autodata:: thermoelasticsim.elastic.materials.GOLD_FCC
   :no-value:

基准与工作流
------------

.. automodule:: thermoelasticsim.elastic.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

顶层导出（便于交叉引用）
------------------------

.. automodule:: thermoelasticsim.elastic
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

力学计算
--------

.. automodule:: thermoelasticsim.elastic.mechanics
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: CppInterface


涨落法（开发中）
----------------

.. automodule:: thermoelasticsim.elastic.fluctuation_method
   :members:
   :undoc-members:
   :show-inheritance:
