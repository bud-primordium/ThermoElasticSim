import pytest
import numpy as np
import logging
from python.integrators import VelocityVerletIntegrator
from python.thermostats import NoseHooverThermostat
from python.md_simulator import MDSimulator
from python.utils import DataCollector


def test_md_without_thermostat(simple_cell, lj_potential_with_neighbor_list):
    """
    测试不带恒温器的 MD 模拟器。
    """
    logger = logging.getLogger(__name__)

    # 创建积分器
    integrator = VelocityVerletIntegrator()

    # 创建不带恒温器的 MD 模拟器
    simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbor_list,
        integrator=integrator,
        thermostat=None,
    )

    # 运行模拟
    data_collector = DataCollector()
    simulator.run(steps=100, dt=0.1, data_collector=data_collector)  # 增大 dt

    # 记录调试信息
    atom1 = simple_cell.atoms[0]
    atom2 = simple_cell.atoms[1]
    logger.debug(f"Atom1 Position: {atom1.position}")
    logger.debug(f"Atom1 Velocity: {atom1.velocity}")
    logger.debug(f"Atom2 Position: {atom2.position}")
    logger.debug(f"Atom2 Velocity: {atom2.velocity}")

    # 断言检查
    assert not np.allclose(atom1.position, [0.0, 0.0, 0.0]), "Atom 1 位置未发生变化。"
    assert not np.allclose(atom2.position, [2.55, 0.0, 0.0]), "Atom 2 位置未发生变化。"
    assert not np.allclose(atom1.velocity, [0.0, 0.0, 0.0]), "Atom 1 速度未发生变化。"
    assert not np.allclose(atom2.velocity, [0.0, 0.0, 0.0]), "Atom 2 速度未发生变化。"


def test_md_with_thermostat(simple_cell, lj_potential_with_neighbor_list):
    """
    测试带 Nose-Hoover 恒温器的 MD 模拟器。
    """
    logger = logging.getLogger(__name__)

    # 创建积分器和恒温器
    integrator = VelocityVerletIntegrator()
    thermostat = NoseHooverThermostat(target_temperature=300, time_constant=0.1)

    # 创建带恒温器的 MD 模拟器
    simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbor_list,
        integrator=integrator,
        thermostat=thermostat,
    )

    # 运行模拟
    data_collector = DataCollector()
    simulator.run(steps=100, dt=0.1, data_collector=data_collector)  # 增大 dt

    # 记录调试信息
    atom1 = simple_cell.atoms[0]
    atom2 = simple_cell.atoms[1]
    logger.debug(f"Atom1 Position: {atom1.position}")
    logger.debug(f"Atom1 Velocity: {atom1.velocity}")
    logger.debug(f"Atom2 Position: {atom2.position}")
    logger.debug(f"Atom2 Velocity: {atom2.velocity}")
    logger.debug(f"Thermostat xi: {thermostat.xi}")

    # 获取原子的最终位置
    atom1_final_position = simple_cell.atoms[0].position
    atom2_final_position = simple_cell.atoms[1].position

    # 计算位置的变化
    atom1_position_change = atom1_final_position - [0.0, 0.0, 0.0]
    atom2_position_change = atom2_final_position - [2.55, 0.0, 0.0]

    # 输出变化量
    logger.info(f"Atom 1 final position: {atom1_final_position}")
    logger.info(f"Atom 1 position change: {atom1_position_change}")

    logger.info(f"Atom 2 final position: {atom2_final_position}")
    logger.info(f"Atom 2 position change: {atom2_position_change}")

    # 断言检查
    assert not np.allclose(atom1.position, [0.0, 0.0, 0.0]), "Atom 1 位置未发生变化。"
    assert not np.allclose(atom2.position, [2.55, 0.0, 0.0]), "Atom 2 位置未发生变化。"
    assert not np.allclose(atom1.velocity, [0.0, 0.0, 0.0]), "Atom 1 速度未发生变化。"
    assert not np.allclose(atom2.velocity, [0.0, 0.0, 0.0]), "Atom 2 速度未发生变化。"
    assert thermostat.xi[0] != 0.0, "xi 未更新。"


# def test_energy_conservation(simple_cell, lj_potential_with_neighbor_list):
#     """
#     测试分子动力学模拟中的能量守恒
#     """
#     from python.integrators import VelocityVerletIntegrator
#     from python.md_simulator import MDSimulator

#     # 设置初始速度（温度约300K）
#     kB = 8.617333262e-5  # eV/K
#     T = 300  # K
#     mass = simple_cell.atoms[0].mass
#     velocity_scale = np.sqrt(2 * kB * T / mass)

#     for atom in simple_cell.atoms:
#         # 设置随机方向的初始速度
#         direction = np.random.randn(3)
#         direction /= np.linalg.norm(direction)
#         atom.velocity = direction * velocity_scale

#     # 创建模拟器
#     integrator = VelocityVerletIntegrator()
#     simulator = MDSimulator(
#         cell=simple_cell,
#         potential=lj_potential_with_neighbor_list,
#         integrator=integrator,
#     )

#     # 记录初始能量
#     initial_potential = lj_potential_with_neighbor_list.calculate_energy(simple_cell)
#     initial_kinetic = sum(
#         0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
#         for atom in simple_cell.atoms
#     )
#     initial_total = initial_potential + initial_kinetic

#     # 运行模拟
#     energies = []
#     steps = 1000
#     dt = 0.1  # fs

#     for step in range(steps):
#         simulator.run(steps=1, dt=dt)

#         # 计算当前能量
#         potential = lj_potential_with_neighbor_list.calculate_energy(simple_cell)
#         kinetic = sum(
#             0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
#             for atom in simple_cell.atoms
#         )
#         total = potential + kinetic
#         energies.append([potential, kinetic, total])

#     # 转换为numpy数组便于分析
#     energies = np.array(energies)

#     # 检查总能量的波动
#     total_energy_fluctuation = np.std(energies[:, 2]) / np.mean(energies[:, 2])

#     # 在较短时间尺度上，能量波动应该很小（通常小于1%）
#     assert (
#         total_energy_fluctuation < 0.01
#     ), f"Energy fluctuation {total_energy_fluctuation:.2%} exceeds 1%"

#     # 检查平均总能量与初始总能量的偏差
#     average_total = np.mean(energies[:, 2])
#     energy_drift = abs(average_total - initial_total) / initial_total

#     assert energy_drift < 0.01, f"Energy drift {energy_drift:.2%} exceeds 1%"
