# 文件名: structure.py
# 作者: Gilbert Young
# 修改日期: 2025-03-30
# 文件描述: 提供原子和晶胞类，用于分子动力学模拟中的结构表示和操作。

"""
结构模块

包含 Atom 和 Cell 类，用于描述分子动力学模拟中的原子和晶胞结构

Classes:
    Atom: 表示单个原子，包含位置、速度和质量等属性
    Cell: 表示晶胞结构，包含晶格矢量和原子列表
"""

import numpy as np
from typing import Optional  # 用于类型注解
from .utils import AMU_TO_EVFSA2
import logging
from .mechanics import StressCalculator
from numba import jit

# 配置日志记录
logger = logging.getLogger(__name__)


# 在类外部定义可JIT优化的函数
@jit(nopython=True)
def _apply_pbc_numba(positions, lattice_inv, lattice_vectors):
    fractional = np.dot(positions, lattice_inv)
    fractional = fractional % 1.0
    return np.dot(fractional, lattice_vectors.T)


class Atom:
    """
    原子类，包含原子的属性和操作

    Parameters
    ----------
    id : int
        原子的唯一标识符
    symbol : str
        原子符号，如 'H', 'O', 'C' 等
    mass_amu : float
        原子的质量，以 amu 为单位
    position : array_like
        原子的初始位置，3D 笛卡尔坐标
    velocity : array_like, optional
        原子的初始速度，3D 笛卡尔坐标，默认为 0

    Attributes
    ----------
    position : numpy.ndarray
        原子的当前位置
    velocity : numpy.ndarray
        原子的当前速度
    force : numpy.ndarray
        作用在原子上的力
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        mass_amu: float,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        self.id = id
        self.symbol = symbol
        self.mass_amu = mass_amu  # 保留原始质量
        self.mass = mass_amu * AMU_TO_EVFSA2  # 质量转换为 eV/fs^2
        self.position = np.array(position, dtype=np.float64)
        self.velocity = (
            np.zeros(3, dtype=np.float64)
            if velocity is None
            else np.array(velocity, dtype=np.float64)
        )
        self.force = np.zeros(3, dtype=np.float64)

    def update_position(self, delta_r):
        """更新原子的位置"""
        if not isinstance(delta_r, np.ndarray):
            delta_r = np.array(delta_r, dtype=np.float64)

        if delta_r.shape != (3,):
            raise ValueError(f"位置增量必须是3D向量，当前形状: {delta_r.shape}")

        self.position += delta_r

    def update_velocity(self, delta_v: np.ndarray) -> None:
        """更新原子的速度

        Parameters
        ----------
        delta_v : numpy.ndarray
            速度增量 (3D 笛卡尔坐标)

        Raises
        ------
        ValueError
            如果delta_v不是3D向量
        """
        self.velocity += delta_v

    def copy(self) -> "Atom":
        """创建 Atom 的深拷贝

        Returns
        -------
        Atom
            新的Atom对象，包含当前原子的所有属性的副本
        """
        return Atom(
            id=self.id,
            symbol=self.symbol,
            mass_amu=self.mass_amu,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )


class Cell:
    """
    晶胞类，包含晶格矢量和原子列表

    Parameters
    ----------
    lattice_vectors : array_like
        3x3 矩阵，表示晶胞的晶格矢量
    atoms : list of Atom
        原子列表，表示晶胞中的原子
    pbc_enabled : bool, optional
        是否启用周期性边界条件，默认为 True

    Attributes
    ----------
    lattice_vectors : numpy.ndarray
        晶胞的晶格矢量
    atoms : list of Atom
        原子列表
    volume : float
        晶胞的体积
    pbc_enabled : bool
        是否启用周期性边界条件
    lattice_locked : bool
        晶格矢量是否被锁定
    """

    def __init__(
        self, lattice_vectors: np.ndarray, atoms: list["Atom"], pbc_enabled: bool = True
    ) -> None:
        # 验证晶格向量
        if not self._validate_lattice_vectors(lattice_vectors):
            raise ValueError("Invalid lattice vectors")

        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self.atoms = atoms
        self.pbc_enabled = pbc_enabled
        self.volume = self.calculate_volume()
        self.lattice_locked = False

        # 计算最小图像所需的辅助矩阵
        self.lattice_inv = np.linalg.inv(self.lattice_vectors.T)

        # 验证周期性边界条件的合理性
        self._validate_pbc_conditions()

        self.stress_calculator = StressCalculator()

    def _validate_lattice_vectors(self, lattice_vectors: np.ndarray) -> bool:
        """验证晶格向量的有效性

        Parameters
        ----------
        lattice_vectors : numpy.ndarray
            待验证的晶格向量矩阵

        Returns
        -------
        bool
            如果晶格向量有效返回True，否则返回False

        Notes
        -----
        检查内容包括：
        1. 是否为3x3矩阵
        2. 是否可逆
        3. 体积是否为正
        """
        if not isinstance(lattice_vectors, np.ndarray):
            lattice_vectors = np.array(lattice_vectors)

        # 检查维度
        if lattice_vectors.shape != (3, 3):
            return False

        # 检查是否可逆
        try:
            np.linalg.inv(lattice_vectors)
        except np.linalg.LinAlgError:
            return False

        # 检查体积是否为正
        if np.linalg.det(lattice_vectors) <= 0:
            return False

        return True

    def _validate_pbc_conditions(self) -> None:
        """验证周期性边界条件的合理性

        Raises
        ------
        ValueError
            如果盒子尺寸非正
        """
        if not self.pbc_enabled:
            return

        # 检查盒子尺寸是否足够大
        box_lengths = self.get_box_lengths()
        min_length = np.min(box_lengths)

        if min_length <= 0:
            raise ValueError("Box dimensions must be positive")

        # 警告可能的物理不合理情况
        for atom1 in self.atoms:
            for atom2 in self.atoms:
                if atom1.id >= atom2.id:
                    continue

                rij = atom2.position - atom1.position
                dist = np.linalg.norm(self.minimum_image(rij))

                if dist < 0.1:  # 原子间距过小
                    logger.warning(
                        f"Atoms {atom1.id} and {atom2.id} are too close: {dist:.3f} Å"
                    )

    def calculate_volume(self) -> float:
        """计算晶胞的体积

        Returns
        -------
        float
            晶胞体积，单位为Å^3
        """
        return np.linalg.det(self.lattice_vectors)

    def get_box_lengths(self) -> np.ndarray:
        """返回模拟盒子在 x、y、z 方向的长度

        Returns
        -------
        numpy.ndarray
            包含三个方向长度的数组，单位为Å
        """
        box_lengths = np.linalg.norm(self.lattice_vectors, axis=1)
        return box_lengths

    def calculate_stress_tensor(self, potential) -> np.ndarray:
        """计算应力张量

        Parameters
        ----------
        potential : Potential
            用于计算应力的势能对象

        Returns
        -------
        numpy.ndarray
            3x3应力张量矩阵，单位为eV/Å^3
        """
        return self.stress_calculator.compute_stress(self, potential)

    def lock_lattice_vectors(self) -> None:
        """锁定晶格向量，防止在优化过程中被修改

        Notes
        -----
        锁定后，晶格向量将不能被修改，直到调用unlock_lattice_vectors()
        """
        self.lattice_locked = True
        logger.debug("Lattice vectors have been locked.")

    def unlock_lattice_vectors(self) -> None:
        """解锁晶格向量，允许在需要时修改

        Notes
        -----
        解锁后，晶格向量可以被修改，直到再次调用lock_lattice_vectors()
        """
        self.lattice_locked = False
        logger.debug("Lattice vectors have been unlocked.")

    def apply_deformation(self, deformation_matrix):
        """
        对晶胞和原子坐标施加变形矩阵F。

        当晶格未锁定时：
        1. 更新晶格矢量： L_new = F * L_old
        2. 更新原子坐标： r_new = r_old * F^T （因为r是行向量形式表示坐标）

        当晶格已锁定时：
        仅对原子坐标施加F变形： r_new = r_old * F^T
        不修改晶格矢量。
        """
        logger = logging.getLogger(__name__)

        positions = self.get_positions()  # shape (N, 3)

        if self.lattice_locked:
            # 晶格矢量不变，仅对原子坐标进行变换
            logger.debug(
                "Lattice vectors are locked. Applying deformation only to atomic positions."
            )

            # 将F作用在笛卡尔坐标上
            # x_new = F * x_old => 若x_old以行向量表示，则 x_new = x_old * F^T
            new_positions = positions @ deformation_matrix.T

            if self.pbc_enabled:
                new_positions = self.apply_periodic_boundary(new_positions)

            for i, atom in enumerate(self.atoms):
                atom.position = new_positions[i]

        else:
            logger.debug(
                "Applying deformation to lattice vectors and atomic positions."
            )

            # 晶格可变形：先更新晶格矢量
            # L_new = F * L_old
            self.lattice_vectors = deformation_matrix @ self.lattice_vectors
            logger.debug(f"Updated lattice vectors:\n{self.lattice_vectors}")

            # 更新逆矩阵
            self.lattice_inv = np.linalg.inv(self.lattice_vectors.T)

            # 原子坐标同样在笛卡尔坐标下被变形
            new_positions = positions @ deformation_matrix.T

            if self.pbc_enabled:
                new_positions = self.apply_periodic_boundary(new_positions)

            for i, atom in enumerate(self.atoms):
                atom.position = new_positions[i]

            # 更新体积
            self.volume = self.calculate_volume()
            logger.debug(f"Updated cell volume: {self.volume}")

    def apply_periodic_boundary(self, positions):
        """
        改进的周期性边界条件实现，增加数值稳定性检查

        Parameters
        ----------
        positions : numpy.ndarray
            原子位置坐标，形状为 (3,) 或 (N, 3)

        Returns
        -------
        numpy.ndarray
            应用周期性边界条件后的位置
        """
        if not self.pbc_enabled:
            return positions

        single_pos = positions.ndim == 1
        if single_pos:
            positions = positions.reshape(1, -1)

        if positions.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3) or (3,)")

        # 转换到分数坐标
        fractional = np.dot(positions, self.lattice_inv)

        # 数值稳定检查
        if not np.all(np.isfinite(fractional)):
            raise ValueError(
                "Non-finite values in fractional coordinates during PBC application"
            )

        # 对接近整数的值进行偏移，避免浮点误差
        tol = 1e-12
        fractional = fractional % 1.0
        # 将非常接近1的值强制稍微拉回
        fractional[(fractional > 1.0 - tol) & (fractional <= 1.0)] = 1.0 - tol
        fractional[(fractional < tol) & (fractional >= 0.0)] = tol

        # 转回笛卡尔坐标
        new_positions = _apply_pbc_numba(
            positions, self.lattice_inv, self.lattice_vectors
        )

        if not np.all(np.isfinite(new_positions)):
            raise ValueError(
                "Non-finite values in cartesian coordinates after PBC application"
            )

        return new_positions[0] if single_pos else new_positions

    def copy(self):
        """创建 Cell 的深拷贝"""
        atoms_copy = [atom.copy() for atom in self.atoms]
        cell_copy = Cell(self.lattice_vectors.copy(), atoms_copy, self.pbc_enabled)
        cell_copy.lattice_locked = self.lattice_locked  # 复制锁定状态
        return cell_copy

    def calculate_temperature(self):
        """
        计算当前系统的温度，扣除质心运动。
        对于多个原子：dof = 3*N - 3
        对于单个原子：dof = 3*N （不扣除质心自由度）
        """

        kb = 8.617333262e-5  # eV/K
        num_atoms = len(self.atoms)
        if num_atoms == 0:
            return 0.0  # 或 raise Exception("No atoms in system.")

        total_mass = sum(atom.mass for atom in self.atoms)
        total_momentum = sum(atom.mass * atom.velocity for atom in self.atoms)
        com_velocity = total_momentum / total_mass

        kinetic = sum(
            0.5
            * atom.mass
            * np.dot(atom.velocity - com_velocity, atom.velocity - com_velocity)
            for atom in self.atoms
        )

        # 根据原子数决定自由度计算方式
        if num_atoms > 1:
            dof = 3 * num_atoms - 3
        else:
            dof = 3 * num_atoms  # 对于单个原子不扣除3个自由度

        if dof <= 0:
            # 若无自由度可分配，温度定义无意义，按需处理
            return 0.0

        temperature = 2.0 * kinetic / (dof * kb)
        return temperature

    def get_positions(self):
        """
        获取所有原子的位置信息

        Returns
        -------
        numpy.ndarray
            原子位置数组，形状为 (num_atoms, 3)
        """
        return np.array([atom.position for atom in self.atoms], dtype=np.float64)

    def get_velocities(self):
        """
        获取所有原子的速度信息

        Returns
        -------
        numpy.ndarray
            原子速度数组，形状为 (num_atoms, 3)
        """
        return np.array([atom.velocity for atom in self.atoms], dtype=np.float64)

    def get_forces(self):
        """
        获取所有原子的力信息

        Returns
        -------
        numpy.ndarray
            原子力数组，形状为 (num_atoms, 3)
        """
        return np.array([atom.force for atom in self.atoms], dtype=np.float64)

    def minimum_image(self, displacement):
        """
        改进的最小镜像约定实现，增加数值检查与修正

        Parameters
        ----------
        displacement : numpy.ndarray
            原始位移向量 (3,)

        Returns
        -------
        numpy.ndarray
            最小镜像位移向量
        """
        if not isinstance(displacement, np.ndarray):
            displacement = np.array(displacement, dtype=np.float64)

        if displacement.shape != (3,):
            raise ValueError(f"位移向量必须是3D向量，当前形状: {displacement.shape}")

        # 转换到分数坐标
        fractional = np.dot(displacement, self.lattice_inv)

        if not np.all(np.isfinite(fractional)):
            raise ValueError(
                "Non-finite values in fractional coordinates for minimum image calculation"
            )

        # 应用最小镜像约定
        fractional -= np.round(fractional)

        # 二次检查数值范围与稳定性
        # 若仍有分量大于0.5或小于-0.5，视为数值不稳定，进行修正
        if np.any(np.abs(fractional) > 0.5 + 1e-10):
            logger.warning(
                "Possible numerical instability in minimum image convention, applying correction."
            )
            fractional = np.clip(fractional, -0.5, 0.5)

        # 转回笛卡尔坐标
        min_image_vector = np.dot(self.lattice_vectors, fractional)

        if not np.all(np.isfinite(min_image_vector)):
            raise ValueError(
                "Non-finite values in cartesian coordinates after minimum image calculation"
            )

        return min_image_vector

    def build_supercell(self, repetition):
        """
        构建超胞，返回一个新的 Cell 对象

        Parameters
        ----------
        repetition : tuple of int
            在 x, y, z 方向上的重复次数，例如 (2, 2, 2)

        Returns
        -------
        Cell
            新的超胞对象
        """
        nx, ny, nz = repetition

        # 获取并复制基本晶胞的晶格矢量
        lattice_vectors = self.lattice_vectors.copy()

        # 计算超胞的晶格矢量
        super_lattice_vectors = lattice_vectors * np.array([nx, ny, nz])[:, np.newaxis]

        # 获取原始原子的分数坐标
        positions = np.array([atom.position for atom in self.atoms])
        # 确保正确的矩阵运算顺序
        fractional = np.dot(positions, np.linalg.inv(lattice_vectors))
        # 确保分数坐标在 [0,1) 范围内
        fractional = fractional % 1.0

        super_atoms = []
        atom_id = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 使用整数偏移
                    # 对分数坐标进行拓展：加上(i,j,k)，然后除以(nx,ny,nz)
                    # 这样就保证了扩展后的分数坐标仍然落在[0,1)内，并且分布正确
                    for idx, atom in enumerate(self.atoms):
                        frac_coord = (fractional[idx] + np.array([i, j, k])) / np.array(
                            [nx, ny, nz]
                        )
                        new_position = np.dot(frac_coord, super_lattice_vectors)

                        new_atom = Atom(
                            id=atom_id,
                            symbol=atom.symbol,
                            mass_amu=atom.mass_amu,
                            position=new_position,
                            velocity=atom.velocity.copy(),
                        )
                        super_atoms.append(new_atom)
                        atom_id += 1

        # 创建新的超胞对象
        super_cell = Cell(
            lattice_vectors=super_lattice_vectors,
            atoms=super_atoms,
            pbc_enabled=self.pbc_enabled,
        )

        return super_cell

    def get_com_velocity(self):
        """
        计算系统质心速度。

        Returns
        -------
        numpy.ndarray
            质心速度向量
        """
        masses = np.array([atom.mass for atom in self.atoms])
        velocities = np.array([atom.velocity for atom in self.atoms])
        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
        return com_velocity

    def remove_com_motion(self):
        """
        移除系统的质心运动。

        该方法计算并移除系统的净平动，保证总动量为零。
        在应用恒温器和恒压器时应该调用此方法。
        """
        com_velocity = self.get_com_velocity()
        for atom in self.atoms:
            atom.velocity -= com_velocity

    def get_com_position(self):
        """
        计算系统质心位置。

        Returns
        -------
        numpy.ndarray
            质心位置向量
        """
        masses = np.array([atom.mass for atom in self.atoms])
        positions = np.array([atom.position for atom in self.atoms])
        total_mass = np.sum(masses)
        com_position = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
        return com_position

    @property
    def num_atoms(self):
        """返回原子数量"""
        return len(self.atoms)
