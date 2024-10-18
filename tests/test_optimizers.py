# tests/test_optimizers.py

import pytest
import numpy as np
from python.optimizers import GradientDescentOptimizer, BFGSOptimizer
from python.potentials import LennardJonesPotential
from python.structure import Atom, Cell
from python.interfaces.cpp_interface import CppInterface

@pytest.fixture
def lj_potential_optim():
    """
    @fixture 定义 Lennard-Jones 势。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

def test_gradient_descent_optimizer(lj_potential_optim):
    """
    @brief 测试梯度下降优化器。
    """
    optimizer = GradientDescentOptimizer(max_steps=100, tol=1e-5, step_size=1e-3)
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 0.0, 0.0]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    optimizer.optimize(cell, lj_potential_optim)
    # 检查优化后晶胞的能量是否降低
    initial_energy = lj_potential_optim.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential_optim.epsilon, lj_potential_optim.sigma, lj_potential_optim.cutoff, cell.get_box_lengths())
    optimizer.optimize(cell, lj_potential_optim)
    final_energy = lj_potential_optim.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential_optim.epsilon, lj_potential_optim.sigma, lj_potential_optim.cutoff, cell.get_box_lengths())
    assert final_energy <= initial_energy

def test_bfgs_optimizer(lj_potential_optim):
    """
    @brief 测试 BFGS 优化器。
    """
    optimizer = BFGSOptimizer(tol=1e-5)
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 0.0, 0.0]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    optimizer.optimize(cell, lj_potential_optim)
    # 检查优化后晶胞的能量是否降低
    initial_energy = lj_potential_optim.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential_optim.epsilon, lj_potential_optim.sigma, lj_potential_optim.cutoff, cell.get_box_lengths())
    optimizer.optimize(cell, lj_potential_optim)
    final_energy = lj_potential_optim.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential_optim.epsilon, lj_potential_optim.sigma, lj_potential_optim.cutoff, cell.get_box_lengths())
    assert final_energy <= initial_energy

def test_optimizer_convergence():
    """
    @brief 测试优化器的收敛性。
    """
    optimizer = GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-4)
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 0.0, 0.0]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    lj_potential = LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)
    optimizer.optimize(cell, lj_potential)
    # 检查优化是否收敛（假设优化器设置了合适的收敛标准）
    assert optimizer.converged

def test_force_direction():
    """
    @brief 验证力的方向是否为负梯度方向。
    """
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 2.55, 2.55]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 8.0  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 计算初始能量和力
    initial_energy = lj_potential.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential.epsilon, lj_potential.sigma, lj_potential.cutoff, cell.get_box_lengths())
    lj_potential.calculate_forces(cell.num_atoms, cell.get_positions(), cell.get_forces(), lj_potential.epsilon, lj_potential.sigma, lj_potential.cutoff, cell.get_box_lengths())
    forces_initial = np.array([atom.force for atom in cell.atoms])

    # 优化结构
    optimizer = GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-4)
    optimizer.optimize(cell, lj_potential)

    # 计算优化后能量和力
    final_energy = lj_potential.calculate_energy(cell.num_atoms, cell.get_positions(), lj_potential.epsilon, lj_potential.sigma, lj_potential.cutoff, cell.get_box_lengths())
    lj_potential.calculate_forces(cell.num_atoms, cell.get_positions(), cell.get_forces(), lj_potential.epsilon, lj_potential.sigma, lj_potential.cutoff, cell.get_box_lengths())
    forces_final = np.array([atom.force for atom in cell.atoms])

    # 检查力的方向是否为负梯度方向
    # 即初始力与优化后力的方向相同
    for f_initial, f_final in zip(forces_initial, forces_final):
        assert np.dot(f_initial, f_final) > 0, "Force direction is not consistent with negative gradient."
