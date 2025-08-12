#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试弹性力学模块（包括应力计算器的迁移一致性测试）
"""
import pytest
import numpy as np
import os
import ctypes
from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.elastic.mechanics import StressCalculator
from thermoelasticsim.potentials.lennard_jones import LennardJonesPotential
from thermoelasticsim.utils.utils import NeighborList

# 检查 pybind11 模块是否可用
try:
    import thermoelasticsim._cpp_core as _cpp_core
    HAS_PYBIND = hasattr(_cpp_core, 'compute_stress')
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


@pytest.mark.migration
def test_stress_pybind_ctypes_consistency(simple_cell, lj_potential):
    """测试pybind11和ctypes计算应力的一致性"""
    if not HAS_PYBIND:
        pytest.skip("pybind11 module _cpp_core not available")
    
    # 尝试定位旧 ctypes 库
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    project_root = None
    while cur != os.path.dirname(cur):
        if os.path.exists(os.path.join(cur, "pyproject.toml")):
            project_root = cur
            break
        cur = os.path.dirname(cur)
    
    if not project_root:
        pytest.skip("Cannot locate project root to find ctypes library")
    
    ctypes_lib_path = os.path.join(project_root, "src", "thermoelasticsim", "lib", "libstress_calculator.so")
    if not os.path.exists(ctypes_lib_path):
        # macOS 使用 .dylib
        ctypes_lib_path = ctypes_lib_path.replace(".so", ".dylib")
    
    if not os.path.exists(ctypes_lib_path):
        pytest.skip(f"ctypes library not found: {ctypes_lib_path}")
    
    # 先创建邻居列表并计算力
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(simple_cell)
    lj_potential.calculate_forces(simple_cell, neighbor_list)
    
    # 准备输入数据
    num_atoms = len(simple_cell.atoms)
    positions = np.array([atom.position for atom in simple_cell.atoms], dtype=np.float64).flatten()
    velocities = np.array([atom.velocity for atom in simple_cell.atoms], dtype=np.float64).flatten()
    forces = np.array([atom.force for atom in simple_cell.atoms], dtype=np.float64).flatten()
    masses = np.array([atom.mass_amu for atom in simple_cell.atoms], dtype=np.float64)
    volume = simple_cell.volume
    box_lengths = np.array([simple_cell.lattice_vectors[i, i] for i in range(3)], dtype=np.float64)
    
    # 加载 ctypes 库
    ctypes_lib = ctypes.CDLL(ctypes_lib_path)
    ctypes_lib.compute_stress.argtypes = [
        ctypes.c_int,                                      # num_atoms
        ctypes.POINTER(ctypes.c_double),                   # positions
        ctypes.POINTER(ctypes.c_double),                   # velocities
        ctypes.POINTER(ctypes.c_double),                   # forces
        ctypes.POINTER(ctypes.c_double),                   # masses
        ctypes.c_double,                                   # volume
        ctypes.POINTER(ctypes.c_double),                   # box_lengths
        ctypes.POINTER(ctypes.c_double)                    # stress_tensor (output)
    ]
    ctypes_lib.compute_stress.restype = ctypes.c_void_p
    
    # 准备指针
    positions_ptr = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    velocities_ptr = velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    forces_ptr = forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    masses_ptr = masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    box_lengths_ptr = box_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    stress_ctypes = np.zeros(9, dtype=np.float64)  # 展平的3x3张量
    stress_ptr = stress_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # 调用 ctypes
    ctypes_lib.compute_stress(
        num_atoms,
        positions_ptr,
        velocities_ptr,
        forces_ptr,
        masses_ptr,
        volume,
        box_lengths_ptr,
        stress_ptr
    )
    
    # 调用 pybind11
    stress_pybind = np.zeros(9, dtype=np.float64)
    _cpp_core.compute_stress(
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        box_lengths,
        stress_pybind
    )
    
    # 比较结果
    assert np.allclose(stress_pybind, stress_ctypes, atol=1e-12)
    
    # 验证应力张量的对称性
    stress_3x3 = stress_pybind.reshape(3, 3)
    assert np.allclose(stress_3x3, stress_3x3.T, atol=1e-12)