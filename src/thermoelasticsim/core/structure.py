#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThermoElasticSim - 结构模块

该模块提供原子和晶胞类，用于分子动力学模拟中的结构表示和操作。

.. moduleauthor:: Gilbert Young
.. created:: 2024-10-14
.. modified:: 2025-08-18
.. version:: 4.0.0

Classes:
    Atom: 表示单个原子，包含位置、速度和质量等属性
    Cell: 表示晶胞结构，包含晶格矢量和原子列表

Examples:
    创建原子和晶胞的基本用法::

        >>> from thermoelasticsim.core.structure import Atom, Cell
        >>> atom = Atom(id=1, symbol="Al", mass_amu=26.98, position=[0, 0, 0])
        >>> print(atom.symbol)
        'Al'

Notes:
    本模块是 ThermoElasticSim 包的核心组件，提供了分子动力学模拟的基础数据结构。

.. versionadded:: 4.0.0
   重构版本，增强了数值稳定性和异常处理。
"""

__version__ = "4.0.0"
__author__ = "Gilbert Young"
__license__ = "GPL-3.0"
__copyright__ = "Copyright 2025, Gilbert Young"

import numpy as np
from typing import Optional  # 用于类型注解
from thermoelasticsim.utils.utils import AMU_TO_EVFSA2, KB_IN_EV
import logging
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

    def move_by(self, displacement: np.ndarray) -> None:
        """通过位置增量移动原子

        Args:
            displacement: 位置增量向量，形状为(3,)

        Raises:
            ValueError: 如果位置增量不是3D向量

        Examples:
            >>> atom = Atom(1, 'H', 1.0, [0, 0, 0])
            >>> atom.move_by([0.1, 0.2, 0.3])
            >>> print(atom.position)
            [0.1 0.2 0.3]
        """
        if not isinstance(displacement, np.ndarray):
            displacement = np.array(displacement, dtype=np.float64)

        if displacement.shape != (3,):
            raise ValueError(f"位置增量必须是3D向量，当前形状: {displacement.shape}")

        self.position += displacement

    def accelerate_by(self, velocity_change: np.ndarray) -> None:
        """通过速度增量改变原子速度

        Args:
            velocity_change: 速度增量向量，形状为(3,)

        Raises:
            ValueError: 如果velocity_change不是3D向量

        Examples:
            >>> atom = Atom(1, 'H', 1.0, [0, 0, 0])
            >>> atom.accelerate_by([0.1, 0.2, 0.3])
            >>> print(atom.velocity)
            [0.1 0.2 0.3]
        """
        if not isinstance(velocity_change, np.ndarray):
            velocity_change = np.array(velocity_change, dtype=np.float64)

        if velocity_change.shape != (3,):
            raise ValueError(f"速度增量必须是3D向量，当前形状: {velocity_change.shape}")

        self.velocity += velocity_change

    def copy(self) -> "Atom":
        """创建 Atom 的深拷贝

        Returns:
            新的Atom对象，包含当前原子的所有属性的副本

        Examples:
            >>> atom1 = Atom(1, 'H', 1.0, [0, 0, 0])
            >>> atom2 = atom1.copy()
            >>> atom2.id == atom1.id
            True
            >>> atom2.position is atom1.position
            False
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
        # 验证输入参数
        if not atoms:
            raise ValueError("原子列表不能为空")

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

        # 验证原子属性
        self._validate_atoms()

        # 验证周期性边界条件的合理性
        self._validate_pbc_conditions()

        # self.stress_calculator = StressCalculator()  # 移除循环导入

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

    def _validate_atoms(self) -> None:
        """验证原子属性的有效性

        Raises
        ------
        ValueError
            如果原子属性无效
        """
        atom_ids = set()
        for atom in self.atoms:
            # 检查原子ID唯一性
            if atom.id in atom_ids:
                raise ValueError(f"原子ID {atom.id} 重复")
            atom_ids.add(atom.id)

            # 检查原子质量
            if atom.mass_amu <= 0:
                raise ValueError(
                    f"原子 {atom.id} 的质量必须为正数，当前: {atom.mass_amu}"
                )

            # 检查位置向量
            if not np.all(np.isfinite(atom.position)):
                raise ValueError(f"原子 {atom.id} 的位置包含无效值")

            # 检查速度向量
            if not np.all(np.isfinite(atom.velocity)):
                raise ValueError(f"原子 {atom.id} 的速度包含无效值")

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
        from thermoelasticsim.elastic.mechanics import StressCalculator

        stress_calculator = StressCalculator()
        return stress_calculator.compute_stress(self, potential)

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

    def apply_deformation(self, deformation_matrix: np.ndarray) -> None:
        """对晶胞和原子坐标施加变形矩阵F

        Args:
            deformation_matrix: 3x3变形矩阵F

        Notes:
            当晶格未锁定时：
            1. 更新晶格矢量： L_new = F * L_old
            2. 更新原子坐标： r_new = r_old * F^T

            当晶格已锁定时：
            仅对原子坐标施加F变形： r_new = r_old * F^T

        Examples:
            >>> import numpy as np
            >>> # 小幅变形矩阵
            >>> F = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> cell.apply_deformation(F)
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

        if not np.all(np.isfinite(fractional)):
            raise ValueError(
                "Non-finite values in fractional coordinates during PBC application"
            )

        # 应用周期性边界条件 - 使用最小镜像原理
        fractional = fractional - np.floor(fractional + 0.5)

        # 转回笛卡尔坐标
        new_positions = np.dot(fractional, self.lattice_vectors.T)

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

    def calculate_temperature(self) -> float:
        """计算当前系统的温度，扣除质心运动
        
        Returns:
            系统温度，单位为K
            
        Notes:
            对于多个原子：dof = 3*N - 3 (扣除质心运动)
            对于单个原子：dof = 3*N (不扣除质心运动)
            
        Examples:
            >>> cell = Cell(lattice_vectors, atoms)
            >>> temp = cell.calculate_temperature()
            >>> print(f"系统温度: {temp:.2f} K")
        """
        
        kb = KB_IN_EV  # 使用utils.py中的标准常数
        num_atoms = len(self.atoms)
        if num_atoms == 0:
            return 0.0
        
        if num_atoms == 1:
            # 单原子系统：直接使用原子动能，不扣除质心运动
            atom = self.atoms[0]
            kinetic = 0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
            dof = 3  # 3个平动自由度
        else:
            # 多原子系统：扣除质心运动
            total_mass = sum(atom.mass for atom in self.atoms)
            total_momentum = sum(atom.mass * atom.velocity for atom in self.atoms)
            com_velocity = total_momentum / total_mass
            
            kinetic = sum(
                0.5
                * atom.mass
                * np.dot(atom.velocity - com_velocity, atom.velocity - com_velocity)
                for atom in self.atoms
            )
            dof = 3 * num_atoms - 3  # 扣除3个质心平动自由度
        
        if dof <= 0:
            return 0.0
            
        temperature = 2.0 * kinetic / (dof * kb)
        return temperature

    def calculate_kinetic_energy(self) -> float:
        """计算当前系统的总动能
        
        Returns:
            系统总动能，单位为eV
            
        Notes:
            计算所有原子的动能总和，包含质心运动
            
        Examples:
            >>> cell = Cell(lattice_vectors, atoms)
            >>> kinetic = cell.calculate_kinetic_energy()
            >>> print(f"系统动能: {kinetic:.6f} eV")
        """
        num_atoms = len(self.atoms)
        if num_atoms == 0:
            return 0.0
        
        # 计算所有原子动能总和（不扣除质心运动）
        total_kinetic = 0.0
        for atom in self.atoms:
            kinetic = 0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
            total_kinetic += kinetic
        
        return total_kinetic

    def get_positions(self) -> np.ndarray:
        """获取所有原子的位置信息

        Returns:
            原子位置数组，形状为(num_atoms, 3)

        Examples:
            >>> positions = cell.get_positions()
            >>> print(f"原子数量: {positions.shape[0]}")
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

        # 确保数值稳定性
        fractional = np.clip(fractional, -0.5, 0.5)

        # 转回笛卡尔坐标
        min_image_vector = np.dot(fractional, self.lattice_vectors.T)

        if not np.all(np.isfinite(min_image_vector)):
            raise ValueError(
                "Non-finite values in cartesian coordinates after minimum image calculation"
            )

        return min_image_vector

    def build_supercell(self, repetition: tuple) -> "Cell":
        """
        构建超胞，返回一个新的 Cell 对象。
        采用更简单、更健壮的算法，直接在笛卡尔坐标系下操作。

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

        # 1. 计算新的超胞晶格矢量
        super_lattice_vectors = self.lattice_vectors.copy()
        super_lattice_vectors[0] *= nx
        super_lattice_vectors[1] *= ny
        super_lattice_vectors[2] *= nz

        super_atoms = []
        atom_id = 0

        # 2. 循环创建新的原子
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 3. 计算每个单胞的平移向量（笛卡尔坐标）
                    translation_vector = (
                        i * self.lattice_vectors[0]
                        + j * self.lattice_vectors[1]
                        + k * self.lattice_vectors[2]
                    )

                    # 4. 复制并平移原子
                    for atom in self.atoms:
                        new_position = atom.position + translation_vector
                        new_atom = Atom(
                            id=atom_id,
                            symbol=atom.symbol,
                            mass_amu=atom.mass_amu,
                            position=new_position,
                            velocity=atom.velocity.copy(),
                        )
                        super_atoms.append(new_atom)
                        atom_id += 1

        # 5. 创建新的超胞对象
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

    def set_positions(self, positions: np.ndarray):
        """
        设置所有原子的位置。

        Parameters
        ----------
        positions : np.ndarray
            原子位置数组，形状为 (num_atoms, 3)。
        """
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"位置数组形状应为 ({self.num_atoms}, 3)，但得到 {positions.shape}"
            )
        for i, atom in enumerate(self.atoms):
            atom.position = positions[i]

    def set_lattice_vectors(self, lattice_vectors: np.ndarray):
        """
        设置新的晶格矢量并更新相关属性。

        Parameters
        ----------
        lattice_vectors : np.ndarray
            新的 3x3 晶格矢量矩阵。
        """
        if not self._validate_lattice_vectors(lattice_vectors):
            raise ValueError("设置的晶格矢量无效。")

        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self.volume = self.calculate_volume()
        self.lattice_inv = np.linalg.inv(self.lattice_vectors.T)
        logger.debug("晶格矢量已更新。")

    def get_fractional_coordinates(self) -> np.ndarray:
        """获取所有原子的分数坐标

        Returns
        -------
        numpy.ndarray
            原子分数坐标数组，形状为(num_atoms, 3)

        Notes
        -----
        分数坐标：r_frac = (L^T)^-1 * r_cart
        其中L是晶格矢量矩阵，r_cart是笛卡尔坐标
        """
        positions = self.get_positions()  # shape (N, 3)
        # 转换到分数坐标：(L^T)^-1 * r
        fractional_coords = np.dot(positions, self.lattice_inv)
        return fractional_coords

    def set_fractional_coordinates(self, fractional_coords: np.ndarray) -> None:
        """根据分数坐标设置所有原子的笛卡尔坐标

        Parameters
        ----------
        fractional_coords : numpy.ndarray
            分数坐标数组，形状为(num_atoms, 3)

        Notes
        -----
        笛卡尔坐标：r_cart = L^T * r_frac
        其中L是晶格矢量矩阵，r_frac是分数坐标
        """
        if fractional_coords.shape != (len(self.atoms), 3):
            raise ValueError(
                f"分数坐标数组形状错误: 期望({len(self.atoms)}, 3), 实际{fractional_coords.shape}"
            )

        # 转换到笛卡尔坐标：L^T * r_frac
        cartesian_coords = np.dot(fractional_coords, self.lattice_vectors.T)

        # 更新原子位置
        for i, atom in enumerate(self.atoms):
            atom.position = cartesian_coords[i].copy()

    def set_positions(self, positions: np.ndarray) -> None:
        """设置所有原子的笛卡尔坐标

        Parameters
        ----------
        positions : numpy.ndarray
            笛卡尔坐标数组，形状为(num_atoms, 3)
        """
        if positions.shape != (len(self.atoms), 3):
            raise ValueError(
                f"位置数组形状错误: 期望({len(self.atoms)}, 3), 实际{positions.shape}"
            )

        for i, atom in enumerate(self.atoms):
            atom.position = positions[i].copy()

    def get_volume(self) -> float:
        """计算晶胞体积
        
        Returns
        -------
        float
            晶胞体积 (Å³)
            
        Notes
        -----
        体积通过晶格矢量的标量三重积计算：
        V = |a · (b × c)|
        """
        # 计算混合积：a · (b × c)
        a, b, c = self.lattice_vectors
        volume = np.abs(np.dot(a, np.cross(b, c)))
        return volume
    
    def scale_lattice(self, scale_factor: float) -> None:
        """按给定因子等比例缩放晶格
        
        Parameters
        ----------
        scale_factor : float
            缩放因子，>1放大，<1缩小
            
        Notes
        -----
        这个方法只缩放晶格矢量，不改变原子坐标。
        通常与原子坐标的相应缩放一起使用。
        
        Examples
        --------
        >>> cell.scale_lattice(1.1)  # 放大10%
        >>> # 同时需要缩放原子坐标以保持相对位置
        >>> for atom in cell.atoms:
        ...     atom.position *= 1.1
        """
        if scale_factor <= 0:
            raise ValueError(f"缩放因子必须为正数，得到 {scale_factor}")
            
        # 缩放所有晶格矢量
        self.lattice_vectors *= scale_factor
        
        # 重新计算逆矩阵和体积
        self._update_cached_properties()
    
    def _update_cached_properties(self) -> None:
        """更新缓存的晶格属性（逆矩阵等）"""
        # 重新计算晶格逆矩阵
        try:
            self.lattice_inv = np.linalg.inv(self.lattice_vectors.T)
        except np.linalg.LinAlgError:
            raise ValueError("晶格矢量矩阵不可逆，可能存在线性相关的矢量")
