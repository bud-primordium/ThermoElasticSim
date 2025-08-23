核心模块 (core)
================

.. currentmodule:: thermoelasticsim.core

本模块提供分子动力学模拟的基础数据结构。

结构模块
--------

.. automodule:: thermoelasticsim.core.structure
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :special-members: __init__

Atom类
~~~~~~

.. autoclass:: thermoelasticsim.core.structure.Atom
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: 方法

   .. autosummary::
      :nosignatures:

      ~Atom.move_by
      ~Atom.accelerate_by
      ~Atom.copy

Cell类
~~~~~~

.. autoclass:: thermoelasticsim.core.structure.Cell
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: 核心方法

   .. autosummary::
      :nosignatures:

      ~Cell.apply_periodic_boundary
      ~Cell.minimum_image
      ~Cell.calculate_temperature
      ~Cell.calculate_kinetic_energy
      ~Cell.build_supercell

   .. rubric:: 形变相关

   .. autosummary::
      :nosignatures:

      ~Cell.apply_deformation
      ~Cell.lock_lattice_vectors
      ~Cell.unlock_lattice_vectors

   .. rubric:: 坐标操作

   .. autosummary::
      :nosignatures:

      ~Cell.get_positions
      ~Cell.set_positions
      ~Cell.get_fractional_coordinates
      ~Cell.set_fractional_coordinates

配置模块
--------

.. automodule:: thermoelasticsim.core.config
   :members:
   :undoc-members:
   :show-inheritance:
