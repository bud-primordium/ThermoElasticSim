# 文件名: integrators.py
# 作者: Gilbert Young
# 修改日期: 2024-11-02
# 文件描述: 实现分子动力学模拟中的积分器，包括速度 Verlet、RK4 和 MTK 积分器。

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class Integrator:
    """积分器基类，定义积分方法的接口"""

    def integrate(self, cell, potential, dt):
        """应用积分器，更新晶胞和原子状态"""
        raise NotImplementedError

    def calculate_kinetic_energy(self, atoms):
        """计算系统总动能"""
        return sum(
            0.5 * atom.mass * np.dot(atom.velocity, atom.velocity) for atom in atoms
        )


class VelocityVerletIntegrator(Integrator):
    """
    速度 Verlet 积分器的实现

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        势能对象，用于计算作用力
    dt : float
        时间步长
    """

    def integrate(self, cell, potential, dt):
        """
        使用速度 Verlet 算法进行分子动力学积分

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力
        dt : float
            时间步长
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


class MTKIntegrator(Integrator):
    """
    Martyna-Tobias-Klein (MTK) 积分器实现

    该积分器专门设计用于NPT系综，可以正确处理热浴和压浴的耦合。
    使用多时间尺度积分方案，确保系统的哈密顿量守恒。

    Parameters
    ----------
    thermostat : Optional[Thermostat]
        恒温器对象
    barostat : Optional[Barostat]
        恒压器对象
    n_chains : int
        Nose-Hoover链的长度，默认为3
    n_yoshida : int
        Yoshida积分器的阶数，默认为4
    """

    def __init__(
        self, thermostat=None, barostat=None, n_chains: int = 3, n_yoshida: int = 4
    ):
        self.thermostat = thermostat
        self.barostat = barostat
        self.n_chains = n_chains
        self.n_yoshida = n_yoshida

        # Yoshida积分器权重
        if n_yoshida == 4:
            self.w = np.zeros(4)
            self.w[0] = self.w[3] = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
            self.w[1] = self.w[2] = 1.0 - 2.0 * self.w[0]
        else:
            raise ValueError("Currently only 4th order Yoshida integrator is supported")

    def integrate(self, cell, potential, dt):
        """
        执行MTK积分

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能对象
        dt : float
            时间步长
        """
        try:
            # 多时间尺度积分
            for w in self.w:
                dt_sub = dt * w
                self._single_step_integration(cell, potential, dt_sub)

        except Exception as e:
            logger.error(f"MTK integration error: {e}")
            raise

    def _single_step_integration(self, cell, potential, dt):
        """执行单步MTK积分"""
        # 1. 更新热浴变量（1/2步）
        if self.thermostat is not None:
            self._update_thermostat_half_step(cell, dt)

        # 2. 更新压浴变量（1/4步）
        if self.barostat is not None:
            self._update_barostat_quarter_step(cell, dt)

        # 3. 更新粒子动量（1/2步）
        self._update_particle_momenta_half_step(cell, potential, dt)

        # 4. 更新粒子位置和晶胞参数（1步）
        self._update_positions_and_cell(cell, dt)

        # 5. 计算新的力
        potential.calculate_forces(cell)

        # 6. 更新粒子动量（1/2步）
        self._update_particle_momenta_half_step(cell, potential, dt)

        # 7. 更新压浴变量（1/4步）
        if self.barostat is not None:
            self._update_barostat_quarter_step(cell, dt)

        # 8. 更新热浴变量（1/2步）
        if self.thermostat is not None:
            self._update_thermostat_half_step(cell, dt)

    def _update_thermostat_half_step(self, cell, dt):
        """更新热浴变量半步"""
        dof = 3 * len(cell.atoms) - 3  # 考虑质心运动约束
        kinetic = self.calculate_kinetic_energy(cell.atoms)

        # 更新Nose-Hoover链变量
        if hasattr(self.thermostat, "xi_chain"):
            self._update_nose_hoover_chain(cell, dt / 2, dof, kinetic)
        else:
            # 单个Nose-Hoover热浴
            self.thermostat.apply(cell.atoms, dt / 2)

    def _update_barostat_quarter_step(self, cell, dt):
        """更新压浴变量四分之一步"""
        if hasattr(self.barostat, "xi"):
            stress_tensor = cell.calculate_stress_tensor()
            current_pressure = self.barostat.calculate_internal_pressure(stress_tensor)

            # 更新压浴动量
            dp = (current_pressure - self.barostat.target_pressure) * cell.volume
            self.barostat.xi += dp * dt / 4

    def _update_particle_momenta_half_step(self, cell, potential, dt):
        """更新粒子动量半步"""
        for atom in cell.atoms:
            atom.momentum += atom.force * dt / 2
            atom.velocity = atom.momentum / atom.mass

    def _update_positions_and_cell(self, cell, dt):
        """更新位置和晶胞参数"""
        if self.barostat is not None:
            # 计算体积缩放因子
            mu = self.barostat.xi.mean()
            scale = np.exp(mu * dt)

            # 更新晶格向量
            cell.lattice_vectors *= scale
            cell.update_volume()

            # 更新原子位置
            for atom in cell.atoms:
                atom.position *= scale
                atom.position = cell.apply_periodic_boundary(atom.position)
        else:
            # 常规位置更新
            for atom in cell.atoms:
                atom.position += atom.velocity * dt
                atom.position = cell.apply_periodic_boundary(atom.position)

    def _update_nose_hoover_chain(self, cell, dt, dof, kinetic):
        """更新Nose-Hoover链"""
        # 计算热浴动能
        Q = self.thermostat.Q
        xi_chain = self.thermostat.xi_chain

        # Suzuki-Yoshida方案更新链变量
        for i in reversed(range(self.n_chains)):
            if i == 0:
                # 第一个热浴与粒子耦合
                dxi = (
                    2 * kinetic
                    - dof * self.thermostat.kb * self.thermostat.target_temperature
                ) / Q[i]
            else:
                # 链中的其他热浴
                dxi = (
                    Q[i - 1] * xi_chain[i - 1] ** 2
                    - self.thermostat.kb * self.thermostat.target_temperature
                ) / Q[i]

            xi_chain[i] += dxi * dt

            # 更新运动能量
            if i > 0:
                kinetic *= np.exp(-xi_chain[i] * dt)

        # 更新粒子速度
        scale = np.exp(-xi_chain[0] * dt)
        for atom in cell.atoms:
            atom.velocity *= scale
            atom.momentum = atom.mass * atom.velocity


class RK4Integrator(Integrator):
    """
    四阶 Runge-Kutta (RK4) 积分器的实现

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        势能对象，用于计算作用力
    dt : float
        时间步长
    """

    def integrate(self, cell, potential, dt):
        """
        使用四阶 Runge-Kutta (RK4) 算法进行分子动力学积分

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力
        dt : float
            时间步长
        """

        def get_state():
            """获取当前原子位置和速度的状态"""
            return np.concatenate(
                [atom.position for atom in cell.atoms]
                + [atom.velocity for atom in cell.atoms]
            )

        def set_state(state):
            """根据给定状态更新原子的位置和速度"""
            num_atoms = len(cell.atoms)
            positions = state[: 3 * num_atoms].reshape((num_atoms, 3))
            velocities = state[3 * num_atoms :].reshape((num_atoms, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                atom.velocity = velocities[i]

        def compute_derivatives(state):
            """计算位置和速度的导数"""
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
