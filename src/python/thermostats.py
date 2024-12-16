# 文件名: thermostats.py
# 作者: Gilbert Young
# 修改日期: 2024-11-01
# 文件描述: 实现分子动力学模拟中的各种恒温器。

import numpy as np
import logging
from typing import List, Dict, Optional
from .interfaces.cpp_interface import CppInterface
import ctypes
from numpy.ctypeslib import ndpointer
from .utils import KB_IN_EV

# 配置日志记录
logger = logging.getLogger(__name__)


class Thermostat:
    """
    恒温器基类，定义恒温器的接口和通用属性

    Parameters
    ----------
    target_temperature : float
        目标温度，单位K
    """

    def __init__(self, target_temperature: float):
        self.target_temperature = target_temperature
        self.kb = KB_IN_EV  # Boltzmann常数，单位eV/K
        # 用于记录历史数据
        self.temperature_history = []
        self.time_history = []
        self.kinetic_energy_history = []
        self.xi_history = []  # 初始化xi_history

    def apply(self, atoms: List, dt: float) -> None:
        """
        应用恒温器，更新原子速度

        Parameters
        ----------
        atoms : list
            原子列表
        dt : float
            时间步长
        """
        raise NotImplementedError

    def get_kinetic_energy(self, atoms: List) -> float:
        """
        计算系统动能

        Parameters
        ----------
        atoms : list
            原子列表

        Returns
        -------
        float
            系统总动能
        """
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64)
        kinetic_energy = 0.5 * masses * np.sum(velocities**2, axis=1)
        return np.sum(kinetic_energy)

    def get_temperature(self, atoms: List) -> float:
        """
        计算当前系统温度，扣除质心运动

        Parameters
        ----------
        atoms : list
            原子列表

        Returns
        -------
        float
            当前温度，单位K
        """
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64)
        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
        relative_velocities = velocities - com_velocity
        kinetic_energy = 0.5 * masses * np.sum(relative_velocities**2, axis=1)
        dof = 3 * len(atoms) - 3  # 考虑质心运动约束
        temperature = 2.0 * np.sum(kinetic_energy) / (dof * self.kb)
        return temperature

    def remove_com_motion(self, atoms: List) -> None:
        """
        移除系统质心运动

        Parameters
        ----------
        atoms : list
            原子列表
        """
        masses = np.array([atom.mass for atom in atoms])
        velocities = np.array([atom.velocity for atom in atoms])
        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass

        for atom in atoms:
            atom.velocity -= com_velocity

    def record_state(self, atoms: List, time: float) -> None:
        """记录系统状态用于后续分析"""
        temp = self.get_temperature(atoms)
        kinetic = self.get_kinetic_energy(atoms)

        self.temperature_history.append(temp)
        self.time_history.append(time)
        self.kinetic_energy_history.append(kinetic)
        # 由子类决定是否记录 xi_chain


class BerendsenThermostat(Thermostat):
    """
    Berendsen 恒温器

    通过速度缩放实现温度控制，使系统快速达到目标温度。
    该方法不产生严格的正则系综，但具有良好的数值稳定性。

    Parameters
    ----------
    target_temperature : float
        目标温度，单位为开尔文 (K)。
    tau : float
        耦合时间常数，控制温度达到目标值的速率。
        较小的 `\\tau` 值表示更快的温度响应，但可能导致数值不稳定；
        较大的 `\\tau` 值则响应较慢，温度调整更加平缓。
    """

    def __init__(self, target_temperature: float, tau: float):
        super().__init__(target_temperature)
        self.tau = tau

    def apply(self, cell, dt: float, potential) -> None:
        """
        应用 Berendsen 恒温器以控制系统温度。

        该恒温器通过以下缩放因子 `\\lambda` 来调整原子速度：

        .. math::
            \\lambda = \\sqrt{1 + \\frac{\\Delta t}{\\tau} \\left(\\frac{T_{\\text{target}}}{T_{\\text{current}}} - 1\\right)}

        其中：
        - `\\Delta t` 为时间步长 `dt`，
        - `T_{\\text{target}}` 为目标温度 `target_temperature`，
        - `T_{\\text{current}}` 为当前系统温度。

        然后通过 `\\lambda` 缩放原子速度，以达到近似的温度控制。

        Parameters
        ----------
        atoms : list
            原子对象的列表。
        dt : float
            时间步长。
        """
        atoms = cell.atoms
        current_temp = self.get_temperature(atoms)
        if current_temp > 0:
            # 计算缩放因子 lambda
            lambda_scale = np.sqrt(
                1.0 + (dt / self.tau) * (self.target_temperature / current_temp - 1.0)
            )

            # 缩放速度
            velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64)
            velocities *= lambda_scale
            for i, atom in enumerate(atoms):
                atom.velocity = velocities[i]

        # 记录状态
        current_time = len(self.time_history) * dt if self.time_history else 0.0
        self.record_state(atoms, current_time)


class AndersenThermostat(Thermostat):
    """
    Andersen 恒温器

    通过随机碰撞来控制温度，实现符合正则系综的采样。
    该方法适合平衡态采样，但由于速度随机化，动力学轨迹会被打断。

    Parameters
    ----------
    target_temperature : float
        目标温度，单位为开尔文 (K)。
    collision_frequency : float
        碰撞频率，单位为 `\\text{fs}^{-1}`。
        较高的碰撞频率会更频繁地随机分配速度，导致温度波动较大；
        较低的碰撞频率则速度随机化较少，温度控制更稳定。
    """

    def __init__(self, target_temperature: float, collision_frequency: float):
        super().__init__(target_temperature)
        self.collision_frequency = collision_frequency

    def apply(self, cell, dt: float, potential) -> None:
        """
        应用 Andersen 恒温器以控制系统温度。

        在每个时间步，针对每个原子，发生碰撞的概率为 `p`：

        .. math::
            p = \\text{collision\\_frequency} \\times \\Delta t

        若发生碰撞，则根据目标温度 `T_{\\text{target}}` 的 Maxwell-Boltzmann 分布重新分配原子的速度：

        .. math::
            v_i = \\mathcal{N}(0, \\sigma)

        其中 `\\sigma` 为速度分布的标准差，定义为：

        .. math::
            \\sigma = \\sqrt{\\frac{k_B \\times T_{\\text{target}}}{m}}

        其中 `k_B` 为玻尔兹曼常数，`m` 为原子质量。

        Parameters
        ----------
        atoms : list
            原子对象的列表。
        dt : float
            时间步长。
        """
        atoms = cell.atoms
        velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)

        collision_probs = self.collision_frequency * dt
        random_values = np.random.random(len(atoms))
        collision_indices = np.where(random_values < collision_probs)[0]

        if collision_indices.size > 0:
            # 计算 sigma
            sigma = np.sqrt(
                self.kb * self.target_temperature / masses[collision_indices]
            )
            # 重新分配速度
            velocities[collision_indices] = np.random.normal(
                0, sigma[:, np.newaxis], (collision_indices.size, 3)
            )

            # 更新原子速度
            for i in collision_indices:
                atoms[i].velocity = velocities[i]

        # # 移除整体平动
        # self.remove_com_motion(atoms)

        # 记录状态
        current_time = len(self.time_history) * dt if self.time_history else 0.0
        self.record_state(atoms, current_time)


class NoseHooverThermostat(Thermostat):
    """
    Nose-Hoover 恒温器 (Python直接实现版本)

    使用Velocity-Verlet + Nose-Hoover对称分解的方式：
    1. 半步更新xi
    2. 半步速度缩放
    3. 半步力更新速度
    4. 更新位置
    5. 调用potential.calculate_forces(cell)以更新力
    6. 半步力更新速度
    7. 半步速度缩放
    8. 半步更新xi
    """

    def __init__(
        self, target_temperature: float, time_constant: float, Q: Optional[float] = None
    ):
        super().__init__(target_temperature)
        self.time_constant = time_constant

        if Q is None:
            self.Q = self._calculate_Q()
        else:
            self.Q = Q

        # Nose-Hoover热浴变量xi初始化
        self.xi = np.array([0.0], dtype=np.float64)

    def _calculate_Q(self) -> float:
        """计算合适的热浴质量"""
        return 3.0 * self.kb * self.target_temperature * self.time_constant**2

    def apply(self, cell, dt: float, potential) -> None:
        """
        应用 Nose-Hoover 恒温器 (Velocity-Verlet 对称分解实现)

        Parameters
        ----------
        cell : Cell
            包含原子信息的Cell对象，可以通过cell.atoms访问原子列表
        dt : float
            时间步长
        potential : Potential
            用于计算力场的对象，必须实现 potential.calculate_forces(cell)
        """

        atoms = cell.atoms
        num_atoms = len(atoms)
        dof = 3 * num_atoms - 3  # 去除刚体平移自由度(如果需要，不需要则可用3*num_atoms)

        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()

        total_mass = np.sum(masses)

        # 可选：移除质心运动，如果不需要可省略
        com_velocity = np.zeros(3)
        for i, atom in enumerate(atoms):
            com_velocity += masses[i] * atom.velocity
        com_velocity /= total_mass
        for i in range(num_atoms):
            velocities[3 * i : 3 * i + 3] -= com_velocity

        def kinetic_energy_func(masses, velocities):
            v2 = velocities.reshape(num_atoms, 3) ** 2
            ke = 0.5 * np.sum(masses[:, np.newaxis] * v2)
            return ke

        # 当前动能
        kinetic_energy = kinetic_energy_func(masses, velocities)

        dt_half = 0.5 * dt

        # 计算G_xi
        G_xi = (2.0 * kinetic_energy - dof * self.kb * self.target_temperature) / self.Q

        # 半步更新xi
        self.xi[0] += dt_half * G_xi

        # 半步缩放速度
        scale = np.exp(-self.xi[0] * dt_half)
        velocities *= scale

        # 半步力更新速度
        for i in range(num_atoms):
            inv_mass = 1.0 / masses[i]
            velocities[3 * i] += dt_half * forces[3 * i] * inv_mass
            velocities[3 * i + 1] += dt_half * forces[3 * i + 1] * inv_mass
            velocities[3 * i + 2] += dt_half * forces[3 * i + 2] * inv_mass

        # 更新位置
        velocities_reshaped = velocities.reshape(num_atoms, 3)
        for i, atom in enumerate(atoms):
            atom.velocity = velocities_reshaped[i] + com_velocity
            atom.position += atom.velocity * dt

        # 重新计算力
        potential.calculate_forces(cell)

        # 获取更新后的力并再次移除质心运动（可选）
        forces = np.array([atom.force for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        com_velocity = np.zeros(3)
        for i, atom in enumerate(atoms):
            com_velocity += masses[i] * atom.velocity
        com_velocity /= total_mass
        for i in range(num_atoms):
            velocities[3 * i : 3 * i + 3] -= com_velocity

        # 第二次半步力更新速度
        for i in range(num_atoms):
            inv_mass = 1.0 / masses[i]
            velocities[3 * i] += dt_half * forces[3 * i] * inv_mass
            velocities[3 * i + 1] += dt_half * forces[3 * i + 1] * inv_mass
            velocities[3 * i + 2] += dt_half * forces[3 * i + 2] * inv_mass

        # 第二次半步缩放速度
        velocities *= scale

        # 再次计算动能和G_xi，更新xi
        kinetic_energy = kinetic_energy_func(masses, velocities)
        G_xi = (2.0 * kinetic_energy - dof * self.kb * self.target_temperature) / self.Q
        self.xi[0] += dt_half * G_xi

        # 更新原子速度（加回COM）
        velocities_reshaped = velocities.reshape(num_atoms, 3)
        for i, atom in enumerate(atoms):
            atom.velocity = velocities_reshaped[i] + com_velocity

        # 记录状态
        current_time = len(self.time_history) * dt if self.time_history else 0.0
        self.record_state(atoms, current_time)

        # 调试日志
        logger.debug(f"Nose-Hoover xi: {self.xi[0]}")
        current_temp = self.get_temperature(atoms)
        logger.debug(f"Current Temperature: {current_temp}K")


class NoseHooverChainThermostat(Thermostat):
    """
    Nose-Hoover 链恒温器

    通过多个热浴形成链来改善遍历性，适合复杂系统。
    采用C++后端实现核心积分步骤。

    Parameters
    ----------
    target_temperature : float
        目标温度，单位K
    time_constant : float
        时间常数，控制热浴耦合强度。
        较大的time_constant意味着热浴链的响应较慢，温度调整更加平缓。
        较小的time_constant则响应较快，但可能导致温度波动。
    chain_length : int, optional
        链的长度，默认为3。
        链长度增加可以改善相空间的遍历性，但会增加计算复杂度。
    Q : numpy.ndarray, optional
        热浴质量参数数组，长度应等于 `chain_length`。
        如果未提供，将根据默认方式初始化。
    """

    def __init__(
        self,
        target_temperature: float,
        time_constant: float,
        chain_length: int = 3,
        Q: Optional[np.ndarray] = None,
    ):
        super().__init__(target_temperature)
        self.time_constant = time_constant
        self.chain_length = chain_length
        self.cpp_interface = CppInterface("nose_hoover_chain")

        if Q is not None:
            Q = np.asarray(Q, dtype=np.float64)
            if Q.size != self.chain_length:
                raise ValueError(
                    f"Q array must have length equal to chain_length ({self.chain_length})."
                )
            self.Q = Q
        else:
            self.Q = self._initialize_chain_masses()

        # 初始化链变量
        self.xi_chain = np.zeros(self.chain_length, dtype=np.float64)

        # 记录链变量历史
        self.xi_history = []  # 添加这个列表用于存储链变量历史

    def _initialize_chain_masses(self) -> np.ndarray:
        """初始化热浴链的质量"""
        Q = np.empty(self.chain_length, dtype=np.float64)
        Q[0] = 3.0 * self.kb * self.target_temperature * self.time_constant**2
        for i in range(1, self.chain_length):
            Q[i] = self.kb * self.target_temperature * self.time_constant**2
        return Q

    def apply(self, atoms: List, dt: float) -> None:
        """
        应用 Nose-Hoover 链恒温器

        Parameters
        ----------
        atoms : list
            原子列表
        dt : float
            时间步长
        """
        num_atoms = len(atoms)

        # 提前分配并填充 numpy 数组以避免重复分配
        masses = np.empty(num_atoms, dtype=np.float64)
        velocities = np.empty(3 * num_atoms, dtype=np.float64)
        forces = np.empty(3 * num_atoms, dtype=np.float64)

        for i, atom in enumerate(atoms):
            masses[i] = atom.mass
            velocities[3 * i : 3 * i + 3] = atom.velocity
            forces[3 * i : 3 * i + 3] = atom.force

        try:
            # 调用 C++ 实现
            self.cpp_interface.nose_hoover_chain(
                dt,
                num_atoms,
                masses,
                velocities,
                forces,
                self.xi_chain,
                self.Q,
                self.chain_length,
                self.target_temperature,
            )

            # 更新原子速度
            velocities = velocities.reshape((num_atoms, 3))
            for i, atom in enumerate(atoms):
                atom.velocity = velocities[i]

            # # 移除质心运动
            # self.remove_com_motion(atoms)

        except Exception as e:
            logger.error(f"Nose-Hoover chain thermostat error: {e}")
            raise

        # 记录状态
        current_time = len(self.time_history) * dt if self.time_history else 0.0
        self.record_state(atoms, current_time)

        # 添加调试日志
        logger.debug(f"Nose-Hoover Chain xi_chain: {self.xi_chain}")
        current_temp = self.get_temperature(atoms)
        logger.debug(f"Current Temperature: {current_temp}K")
        # 记录当前 xi_chain 状态到历史中
        self.xi_history.append(self.xi_chain.copy())

    def get_chain_state(self) -> Dict:
        """获取热浴链的当前状态"""
        return {
            "xi_chain": self.xi_chain.copy(),
            "Q": self.Q.copy(),
            "chain_length": self.chain_length,
        }
