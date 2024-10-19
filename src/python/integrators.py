# 文件名: integrators.py
# 作者: Gilbert Young
# 修改日期: 2024年10月19日
# 文件描述: 实现分子动力学模拟中的积分器，包括速度 Verlet 和四阶 Runge-Kutta 积分器。

"""
积分器模块。

包含 Integrator 基类及 VelocityVerlet 和四阶 Runge-Kutta (RK4) 积分器的实现。
"""

import numpy as np


class Integrator:
    """
    积分器基类，定义积分方法的接口。
    """

    def integrate(self, cell, potential, dt):
        """应用积分器，更新晶胞和原子状态。"""
        raise NotImplementedError


class VelocityVerletIntegrator(Integrator):
    """
    速度 Verlet 积分器的实现。

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象。
    potential : Potential
        势能对象，用于计算作用力。
    dt : float
        时间步长。
    """

    def integrate(self, cell, potential, dt):
        """
        使用速度 Verlet 算法进行分子动力学积分。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        potential : Potential
            势能对象，用于计算作用力。
        dt : float
            时间步长。
        """
        atoms = cell.atoms
        # 第一半步：更新位置
        for atom in atoms:
            atom.position += atom.velocity * dt + 0.5 * atom.force / atom.mass * dt**2
            # 应用周期性边界条件
            atom.position = cell.apply_periodic_boundary(atom.position)
        # 保存旧的力
        forces_old = [atom.force.copy() for atom in atoms]
        # 计算新的力
        potential.calculate_forces(cell)
        # 第二半步：更新速度
        for atom, force_old in zip(atoms, forces_old):
            atom.velocity += 0.5 * (atom.force + force_old) / atom.mass * dt


class RK4Integrator(Integrator):
    """
    四阶 Runge-Kutta (RK4) 积分器的实现。

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象。
    potential : Potential
        势能对象，用于计算作用力。
    dt : float
        时间步长。
    """

    def integrate(self, cell, potential, dt):
        """
        使用四阶 Runge-Kutta (RK4) 算法进行分子动力学积分。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        potential : Potential
            势能对象，用于计算作用力。
        dt : float
            时间步长。
        """

        def get_state():
            """获取当前原子位置和速度的状态。"""
            return np.concatenate(
                [atom.position for atom in cell.atoms]
                + [atom.velocity for atom in cell.atoms]
            )

        def set_state(state):
            """根据给定状态更新原子的位置和速度。"""
            num_atoms = len(cell.atoms)
            positions = state[: 3 * num_atoms].reshape((num_atoms, 3))
            velocities = state[3 * num_atoms :].reshape((num_atoms, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                atom.velocity = velocities[i]

        def compute_derivatives(state):
            """计算位置和速度的导数。"""
            num_atoms = len(cell.atoms)
            positions = state[: 3 * num_atoms].reshape((num_atoms, 3))
            velocities = state[3 * num_atoms :].reshape((num_atoms, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                atom.velocity = velocities[i]
            # 计算力
            potential.calculate_forces(cell)
            # 计算导数
            derivatives = np.zeros_like(state)
            for i, atom in enumerate(cell.atoms):
                derivatives[3 * i : 3 * i + 3] = atom.velocity
                derivatives[3 * num_atoms + 3 * i : 3 * num_atoms + 3 * i + 3] = (
                    atom.force / atom.mass
                )
            return derivatives

        state = get_state()
        k1 = compute_derivatives(state)
        k2 = compute_derivatives(state + 0.5 * dt * k1)
        k3 = compute_derivatives(state + 0.5 * dt * k2)
        k4 = compute_derivatives(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        set_state(new_state)
