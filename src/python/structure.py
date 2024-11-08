# 文件名: structure.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 提供原子和晶胞类，用于分子动力学模拟中的结构表示和操作。

"""
结构模块

包含 Atom 和 Cell 类，用于描述分子动力学模拟中的原子和晶胞结构
"""

import numpy as np
from .utils import AMU_TO_EVFSA2
import logging

# 配置日志记录
logger = logging.getLogger(__name__)


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

    def __init__(self, id, symbol, mass_amu, position, velocity=None):
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
        self.position += delta_r

    def update_velocity(self, delta_v):
        """更新原子的速度"""
        self.velocity += delta_v

    def copy(self):
        """创建 Atom 的深拷贝"""
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

    def __init__(self, lattice_vectors, atoms, pbc_enabled=True):
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

    def _validate_lattice_vectors(self, lattice_vectors):
        """验证晶格向量的有效性"""
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

    def _validate_pbc_conditions(self):
        """验证周期性边界条件的合理性"""
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

    def calculate_volume(self):
        """计算晶胞的体积"""
        return np.linalg.det(self.lattice_vectors)

    def get_box_lengths(self):
        """返回模拟盒子在 x、y、z 方向的长度"""
        box_lengths = np.linalg.norm(self.lattice_vectors, axis=1)
        return box_lengths

    def lock_lattice_vectors(self):
        """锁定晶格向量，防止在优化过程中被修改"""
        self.lattice_locked = True
        logger.debug("Lattice vectors have been locked.")

    def unlock_lattice_vectors(self):
        """解锁晶格向量，允许在需要时修改"""
        self.lattice_locked = False
        logger.debug("Lattice vectors have been unlocked.")

    def apply_deformation(self, deformation_matrix):
        """
        对晶胞和原子坐标施加变形矩阵

        Parameters
        ----------
        deformation_matrix : array_like
            3x3 变形矩阵
        """
        if self.lattice_locked:
            logger.debug(
                "Lattice vectors are locked. Only applying deformation to atomic positions."
            )
            # 批量更新原子坐标
            positions = np.array([atom.position for atom in self.atoms])  # (N, 3)
            # 直接计算分数坐标
            fractional = np.linalg.solve(self.lattice_vectors.T, positions.T)  # (3, N)
            # 应用变形
            fractional = np.dot(deformation_matrix, fractional)  # (3, N)
            # 转回笛卡尔坐标
            new_positions = np.dot(self.lattice_vectors.T, fractional).T  # (N, 3)

            if self.pbc_enabled:
                # 应用周期性边界条件
                new_positions = self.apply_periodic_boundary(new_positions)

            # 更新原子位置
            for i, atom in enumerate(self.atoms):
                atom.position = new_positions[i]
        else:
            logger.debug(
                "Applying deformation to lattice vectors and atomic positions."
            )

            # 先获取当前分数坐标
            positions = np.array([atom.position for atom in self.atoms])  # (N, 3)
            fractional = np.linalg.solve(self.lattice_vectors.T, positions.T)  # (3, N)

            # 更新晶格矢量
            self.lattice_vectors = np.dot(self.lattice_vectors, deformation_matrix.T)
            logger.debug(f"Updated lattice vectors:\n{self.lattice_vectors}")

            # 重新计算逆晶格矢量矩阵
            self.lattice_inv = np.linalg.inv(self.lattice_vectors.T)

            # 直接使用原始分数坐标计算新的笛卡尔坐标
            new_positions = np.dot(self.lattice_vectors.T, fractional).T  # (N, 3)

            if self.pbc_enabled:
                new_positions = self.apply_periodic_boundary(new_positions)

            # 更新原子位置
            for i, atom in enumerate(self.atoms):
                atom.position = new_positions[i]

            # 更新体积
            self.volume = self.calculate_volume()
            logger.debug(f"Updated cell volume: {self.volume}")

    def apply_periodic_boundary(self, positions):
        """
        改进的周期性边界条件实现

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

        # 处理单个原子和多个原子的情况
        single_pos = positions.ndim == 1
        if single_pos:
            positions = positions.reshape(1, -1)

        # 确保输入的维度正确
        if positions.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3) or (3,)")

        try:
            # 转换到分数坐标
            fractional = np.dot(positions, self.lattice_inv)

            # 应用周期性边界条件
            fractional = fractional % 1.0

            # 转回笛卡尔坐标
            new_positions = np.dot(fractional, self.lattice_vectors.T)

            # 数值稳定性检查
            if np.any(np.isnan(new_positions)):
                raise ValueError("NaN values detected in position calculation")

        except Exception as e:
            logger.error(f"Error in periodic boundary application: {str(e)}")
            raise

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

        Returns
        -------
        float
            当前温度，单位K。
        """
        kb = 8.617333262e-5  # eV/K
        total_mass = sum(atom.mass for atom in self.atoms)
        total_momentum = sum(atom.mass * atom.velocity for atom in self.atoms)
        com_velocity = total_momentum / total_mass

        kinetic = sum(
            0.5
            * atom.mass
            * np.dot(atom.velocity - com_velocity, atom.velocity - com_velocity)
            for atom in self.atoms
        )
        dof = 3 * len(self.atoms) - 3  # 考虑质心运动约束
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
        改进的最小镜像约定实现

        Parameters
        ----------
        displacement : numpy.ndarray
            原始位移向量

        Returns
        -------
        numpy.ndarray
            最小镜像位移向量
        """
        if displacement.shape != (3,):
            raise ValueError("Displacement must be a 3-dimensional vector")

        # 转换到分数坐标
        try:
            fractional = np.dot(displacement, self.lattice_inv)
        except np.linalg.LinAlgError:
            logger.error("Failed to compute fractional coordinates")
            raise

        # 应用最小镜像约定
        fractional -= np.round(fractional)

        # 数值稳定性检查
        if np.any(np.abs(fractional) > 0.5 + 1e-10):
            logger.warning("Possible numerical instability in minimum image convention")

        # 转回笛卡尔坐标
        return np.dot(
            self.lattice_vectors,
            fractional,
        )

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
        # 获取基本晶胞的晶格矢量
        lattice_vectors = self.lattice_vectors.copy()
        # 计算超胞的晶格矢量
        super_lattice_vectors = np.dot(np.diag([nx, ny, nz]), lattice_vectors)

        # 获取原始原子的分数坐标
        positions = np.array([atom.position for atom in self.atoms])
        fractional = np.linalg.solve(lattice_vectors.T, positions.T).T  # Shape (N, 3)

        super_atoms = []
        atom_id = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 平移矢量（分数坐标）
                    shift = np.array([i, j, k])
                    for idx, atom in enumerate(self.atoms):
                        # 新的分数坐标
                        frac_coord = fractional[idx] + shift
                        # 检查是否在范围内，避免重复原子
                        if i == nx - 1 and frac_coord[0] >= nx:
                            continue
                        if j == ny - 1 and frac_coord[1] >= ny:
                            continue
                        if k == nz - 1 and frac_coord[2] >= nz:
                            continue
                        # 转换回笛卡尔坐标
                        new_position = np.dot(frac_coord, lattice_vectors)
                        # 创建新的原子
                        new_atom = Atom(
                            id=atom_id,
                            symbol=atom.symbol,
                            mass_amu=atom.mass_amu,
                            position=new_position,
                            velocity=atom.velocity.copy(),
                        )
                        super_atoms.append(new_atom)
                        atom_id += 1

        # 创建新的 Cell 对象
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
