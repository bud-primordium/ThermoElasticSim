势函数模块 (potentials)
=======================

.. currentmodule:: thermoelasticsim.potentials

顶层包
------

.. automodule:: thermoelasticsim.potentials
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList

本模块提供各种原子间相互作用势函数。

基类
----

.. automodule:: thermoelasticsim.potentials.base
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList

EAM势
-----

.. automodule:: thermoelasticsim.potentials.eam
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList, CppInterface, Potential

Lennard-Jones势
---------------

.. automodule:: thermoelasticsim.potentials.lennard_jones
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList, CppInterface, Potential

Tersoff势
---------

.. automodule:: thermoelasticsim.potentials.tersoff
   :members: TersoffC1988Potential
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList, CppInterface, Potential

机器学习势（开发中）
--------------------

.. automodule:: thermoelasticsim.potentials.mlp
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Cell, NeighborList, CppInterface, Potential
