#!/usr/bin/env python3
"""
测试弹性力学模块
"""

import numpy as np
import pytest

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.elastic.mechanics import StressCalculator
from thermoelasticsim.potentials.lennard_jones import LennardJonesPotential
from thermoelasticsim.utils.utils import NeighborList

# 检查 pybind11 模块是否可用
try:
    import thermoelasticsim._cpp_core as _cpp_core

    HAS_PYBIND = hasattr(_cpp_core, "compute_stress")
except Exception:
    HAS_PYBIND = False


@pytest.fixture
def simple_cell():
    """创建一个简单的双原子晶胞"""
    atom1 = Atom(id=1, symbol="Ar", mass_amu=39.948, position=np.array([0, 0, 0]))
    atom2 = Atom(id=2, symbol="Ar", mass_amu=39.948, position=np.array([3.0, 0, 0]))
    # 设置速度
    atom1.velocity = np.array([1.0, 0.0, 0.0])
    atom2.velocity = np.array([-1.0, 0.0, 0.0])
    cell_vectors = np.diag([10, 10, 10])
    return Cell(cell_vectors, [atom1, atom2])


@pytest.fixture
def lj_potential():
    """创建LJ势实例"""
    return LennardJonesPotential(epsilon=0.01, sigma=3.405, cutoff=10.0)


def test_stress_calculator_basic(simple_cell, lj_potential):
    """测试StressCalculator的基本功能"""
    stress_calc = StressCalculator()

    # 创建邻居列表
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(simple_cell)

    # 计算力
    lj_potential.calculate_forces(simple_cell, neighbor_list)

    # 计算应力
    stress_tensor = stress_calc.calculate_stress_basic(simple_cell, lj_potential)

    # 检查应力张量的形状
    assert stress_tensor.shape == (3, 3)

    # 检查对称性
    assert np.allclose(stress_tensor, stress_tensor.T, atol=1e-10)
