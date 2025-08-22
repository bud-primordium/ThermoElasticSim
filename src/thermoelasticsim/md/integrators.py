# 文件名: integrators.py
# 作者: Gilbert Young
# 修改日期: 2025-08-18
# 文件描述: 实现分子动力学模拟中的积分器，包括速度 Verlet、RK4 和 MTK 积分器。

import logging

import numpy as np

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.potentials import Potential

logger = logging.getLogger(__name__)


class Integrator:
    """积分器基类，定义积分方法的接口和通用功能"""

    def __init__(self):
        # 用于记录积分器性能和稳定性的数据
        self.energy_history = []
        self.time_history = []
        self.error_history = []
        self.initial_energy = None

    def integrate(self, cell: Cell, potential: Potential, dt: float) -> None:
        """应用积分器，更新晶胞和原子状态

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力
        dt : float
            时间步长

        Raises
        ------
        NotImplementedError
            如果子类没有实现该方法
        """
        raise NotImplementedError

    def calculate_kinetic_energy(self, atoms: list) -> float:
        """计算系统总动能"""
        return sum(
            0.5 * atom.mass * np.dot(atom.velocity, atom.velocity) for atom in atoms
        )

    def calculate_total_energy(self, cell: Cell, potential: Potential) -> float:
        """计算系统总能量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        float
            系统总能量(动能+势能)，单位为eV
        """
        kinetic = self.calculate_kinetic_energy(cell.atoms)
        potential_energy = potential.calculate_energy(cell)
        return kinetic + potential_energy

    def monitor_energy_conservation(
        self, cell: Cell, potential: Potential, current_time: float
    ) -> float:
        """监控能量守恒情况

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象
        current_time : float
            当前模拟时间

        Returns
        -------
        float
            相对能量误差

        Notes
        -----
        当相对能量误差超过1e-3时会记录警告日志
        """
        current_energy = self.calculate_total_energy(cell, potential)

        if self.initial_energy is None:
            self.initial_energy = current_energy

        # 计算相对能量误差
        relative_error = abs(
            (current_energy - self.initial_energy) / self.initial_energy
        )

        # 记录数据
        self.energy_history.append(current_energy)
        self.time_history.append(current_time)
        self.error_history.append(relative_error)

        # 如果相对误差过大，发出警告
        if relative_error > 1e-3:  # 可配置的阈值
            logger.warning(
                f"Large energy drift detected: {relative_error:.2e} at time {current_time:.2f}"
            )

        return relative_error

    def suggest_timestep(self, cell: Cell, potential: Potential) -> float:
        """基于系统特性建议时间步长

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        float
            建议的时间步长，单位为fs

        Notes
        -----
        时间步长基于系统最高频率周期的1/20计算
        """
        # 计算系统最高频率（基于力常数的近似）
        max_frequency = self._estimate_max_frequency(cell, potential)

        # 建议时间步长为最高频率周期的1/20
        suggested_dt = 1.0 / (20.0 * max_frequency)

        return suggested_dt

    def _estimate_max_frequency(self, cell: Cell, potential: Potential) -> float:
        """估算系统最高频率

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        float
            估算的最高频率，单位为fs^-1

        Notes
        -----
        使用有限差分方法近似计算力常数矩阵
        """
        # 计算力常数矩阵的近似
        # 这里使用简单的有限差分方法
        epsilon = 1e-5
        original_positions = np.array([atom.position for atom in cell.atoms])
        original_forces = np.array([atom.force for atom in cell.atoms])

        max_freq = 0.0
        for i in range(len(cell.atoms)):
            for dim in range(3):
                # 扰动位置
                cell.atoms[i].position[dim] += epsilon
                potential.calculate_forces(cell)
                new_forces = np.array([atom.force for atom in cell.atoms])

                # 恢复位置
                cell.atoms[i].position[dim] -= epsilon

                # 计算力常数
                force_constant = (
                    -(new_forces[i][dim] - original_forces[i][dim]) / epsilon
                )

                # 估算频率
                freq = np.sqrt(abs(force_constant) / cell.atoms[i].mass)
                max_freq = max(max_freq, freq)

        # 恢复原始状态
        for i, pos in enumerate(original_positions):
            cell.atoms[i].position = pos
        potential.calculate_forces(cell)

        return max_freq


class VelocityVerletIntegrator(Integrator):
    """
    改进的速度Verlet积分器实现

    特点：
    1. 能量守恒监控
    2. 自适应时间步长建议
    3. 详细的错误处理和日志记录
    """

    def __init__(self):
        super().__init__()
        self.name = "Velocity Verlet"

    def integrate(self, cell: Cell, potential: Potential, dt: float) -> None:
        """使用速度Verlet算法进行分子动力学积分

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力
        dt : float
            时间步长，单位为fs

        Raises
        ------
        RuntimeError
            如果积分过程中出现严重错误
        Exception
            如果其他未捕获的异常发生

        Notes
        -----
        该方法执行标准的Velocity Verlet积分步骤：
        1. 速度更新半步
        2. 位置更新整步
        3. 计算新力
        4. 速度再更新半步
        """
        try:
            atoms = cell.atoms

            # 记录初始状态用于错误恢复
            initial_state = self._save_state(atoms)

            # 速度先更新半步
            self._update_velocities_half_step(atoms, dt)

            # 位置整步更新
            self._update_positions_full_step(atoms, cell, dt)

            # 更新力
            potential.calculate_forces(cell)

            # 速度再更新半步
            self._update_velocities_half_step(atoms, dt)

            # 监控能量守恒
            current_time = len(self.time_history) * dt
            energy_error = self.monitor_energy_conservation(
                cell, potential, current_time
            )

            # # 如果能量误差过大，回滚到初始状态
            # if energy_error > 1e-2:  # 可配置的阈值
            #     logger.error(f"Critical energy conservation error: {energy_error:.2e}")
            #     self._restore_state(atoms, initial_state)
            #     raise RuntimeError(
            #         "Integration step failed due to energy conservation violation"
            #     )

        except Exception as e:
            logger.error(f"Error in Velocity Verlet integration: {str(e)}")
            raise

    def _update_velocities_half_step(self, atoms, dt):
        """更新速度半步"""
        dt_half = 0.5 * dt
        for atom in atoms:
            atom.velocity += dt_half * atom.force / atom.mass

    def _update_positions_full_step(self, atoms, cell, dt):
        """更新位置整步"""
        for atom in atoms:
            atom.position += dt * atom.velocity
            atom.position = cell.apply_periodic_boundary(atom.position)

    def _save_state(self, atoms):
        """保存当前状态"""
        return [
            (atom.position.copy(), atom.velocity.copy(), atom.force.copy())
            for atom in atoms
        ]

    def _restore_state(self, atoms, state):
        """恢复到保存的状态"""
        for atom, (pos, vel, force) in zip(atoms, state, strict=False):
            atom.position = pos
            atom.velocity = vel
            atom.force = force


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


class SymplecticIntegrator(Integrator):
    """
    辛积分器实现，使用构造方法保持哈密顿系统的几何结构

    Parameters
    ----------
    order : int
        积分器阶数 (2 或 4)
    """

    def __init__(self, order=2):
        super().__init__()
        self.order = order
        self.name = f"{order}阶辛积分器"
        self._initialize_coefficients()

    def _initialize_coefficients(self):
        """初始化积分器系数"""
        if self.order == 2:
            # 二阶方法（等价于Velocity Verlet）
            self.position_coeffs = np.array([0.5, 0.5])
            self.momentum_coeffs = np.array([1.0, 0.0])
        elif self.order == 4:
            # 四阶Forest-Ruth方法
            theta = 1.351207191959657
            self.position_coeffs = np.array(
                [theta / 2, (1 - theta) / 2, (1 - theta) / 2, theta / 2]
            )
            self.momentum_coeffs = np.array([theta, 1 - 2 * theta, theta, 0.0])
        else:
            raise ValueError(f"不支持{self.order}阶辛积分器")

    def integrate(self, cell, potential, dt):
        """执行辛积分步骤"""
        try:
            atoms = cell.atoms
            n_steps = len(self.position_coeffs)

            # 保存初始状态
            initial_state = self._save_state(atoms)

            # 初始力计算
            potential.calculate_forces(cell)

            # 执行分步积分
            for i in range(n_steps):
                # 位置更新
                self._update_positions(atoms, cell, dt * self.position_coeffs[i])

                # 如果不是最后一步，更新力
                if i < n_steps - 1:
                    potential.calculate_forces(cell)

                # 速度更新
                if self.momentum_coeffs[i] != 0:
                    self._update_velocities(atoms, dt * self.momentum_coeffs[i])

            # 计算最终力
            potential.calculate_forces(cell)

            # 监控能量守恒
            current_time = len(self.time_history) * dt
            energy_error = self.monitor_energy_conservation(
                cell, potential, current_time
            )

            if energy_error > 1e-2:
                self._restore_state(atoms, initial_state)
                raise RuntimeError("积分步骤因能量守恒违反而失败")

        except Exception as e:
            logger.error(f"辛积分器错误: {str(e)}")
            raise

    def _update_positions(self, atoms, cell, dt):
        """更新位置"""
        for atom in atoms:
            atom.position += dt * atom.velocity
            atom.position = cell.apply_periodic_boundary(atom.position)

    def _update_velocities(self, atoms, dt):
        """更新速度"""
        for atom in atoms:
            atom.velocity += dt * atom.force / atom.mass

    def _save_state(self, atoms):
        """保存状态"""
        return [
            (atom.position.copy(), atom.velocity.copy(), atom.force.copy())
            for atom in atoms
        ]

    def _restore_state(self, atoms, state):
        """恢复状态"""
        for atom, (pos, vel, force) in zip(atoms, state, strict=False):
            atom.position = pos
            atom.velocity = vel
            atom.force = force


class RK4Integrator(Integrator):
    """
    优化的四阶Runge-Kutta (RK4) 积分器实现

    虽然RK4不是辛积分器，但在某些情况下仍然有用：
    1. 系统不是哈密顿系统时
    2. 需要高精度的短期积分时
    3. 处理强非线性问题时
    """

    def __init__(self):
        super().__init__()
        self.name = "RK4"

    def integrate(self, cell, potential, dt):
        """执行RK4积分"""
        try:
            # 保存初始状态用于出错恢复
            initial_state = self._get_state(cell.atoms)

            # 执行RK4步骤
            k1 = self._compute_derivatives(cell, potential)

            self._advance_state(cell, k1, dt / 2)
            k2 = self._compute_derivatives(cell, potential)

            self._restore_state(cell.atoms, initial_state)
            self._advance_state(cell, k2, dt / 2)
            k3 = self._compute_derivatives(cell, potential)

            self._restore_state(cell.atoms, initial_state)
            self._advance_state(cell, k3, dt)
            k4 = self._compute_derivatives(cell, potential)

            # 最终更新
            self._restore_state(cell.atoms, initial_state)
            self._final_update(cell, k1, k2, k3, k4, dt)

            # 更新力
            potential.calculate_forces(cell)

            # 监控能量守恒
            current_time = len(self.time_history) * dt
            energy_error = self.monitor_energy_conservation(
                cell, potential, current_time
            )

            if energy_error > 1e-2:
                self._restore_state(cell.atoms, initial_state)
                raise RuntimeError("积分步骤因能量守恒违反而失败")

        except Exception as e:
            logger.error(f"RK4积分器错误: {str(e)}")
            raise

    def _get_state(self, atoms):
        """获取当前状态"""
        return [(atom.position.copy(), atom.velocity.copy()) for atom in atoms]

    def _restore_state(self, atoms, state):
        """恢复到保存的状态"""
        for atom, (pos, vel) in zip(atoms, state, strict=False):
            atom.position = pos
            atom.velocity = vel

    def _compute_derivatives(self, cell, potential):
        """计算导数"""
        potential.calculate_forces(cell)
        derivatives = []
        for atom in cell.atoms:
            derivatives.append(
                (
                    atom.velocity.copy(),  # dr/dt = v
                    atom.force.copy() / atom.mass,  # dv/dt = F/m
                )
            )
        return derivatives

    def _advance_state(self, cell, derivatives, dt):
        """
        基于导数前进系统状态

        Parameters
        ----------
        cell : Cell
            晶胞对象
        derivatives : list
            导数列表，每个元素包含位置和速度的导数
        dt : float
            时间步长
        """
        for atom, (dr_dt, dv_dt) in zip(cell.atoms, derivatives, strict=False):
            atom.position += dr_dt * dt
            atom.position = cell.apply_periodic_boundary(atom.position)
            atom.velocity += dv_dt * dt

    def _final_update(self, cell, k1, k2, k3, k4, dt):
        """
        使用RK4方法执行最终状态更新

        采用经典的RK4权重组合:
        y(t + dt) = y(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)

        Parameters
        ----------
        cell : Cell
            晶胞对象
        k1, k2, k3, k4 : list
            四个RK步骤的导数
        dt : float
            时间步长
        """
        dt6 = dt / 6.0
        for (
            atom,
            (dr_dt1, dv_dt1),
            (dr_dt2, dv_dt2),
            (dr_dt3, dv_dt3),
            (dr_dt4, dv_dt4),
        ) in zip(cell.atoms, k1, k2, k3, k4, strict=False):

            # 更新位置
            dr = dt6 * (dr_dt1 + 2 * dr_dt2 + 2 * dr_dt3 + dr_dt4)
            atom.position += dr
            atom.position = cell.apply_periodic_boundary(atom.position)

            # 更新速度
            dv = dt6 * (dv_dt1 + 2 * dv_dt2 + 2 * dv_dt3 + dv_dt4)
            atom.velocity += dv
