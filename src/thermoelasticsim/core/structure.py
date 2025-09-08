#!/usr/bin/env python3
r"""
分子动力学结构模块

该模块提供分子动力学模拟中的基础数据结构，包括原子和晶胞的表示。
实现了周期性边界条件、最小镜像约定等关键算法。

理论基础
--------
分数/笛卡尔坐标定义（采用列向量记号）：

.. math::
    \mathbf{r} = \mathbf{L}\,\mathbf{s},\qquad
    \mathbf{s} = \mathbf{L}^{-1}\,\mathbf{r}

其中：

:math:`\mathbf{r}`
    笛卡尔坐标

:math:`\mathbf{s}`
    分数坐标

:math:`\mathbf{L}`
    晶格矢量矩阵（列向量为基矢）

最小镜像约定 (Minimum Image Convention)：

.. math::
    \mathbf{r}_{\min} = \mathbf{r} - \mathbf{L}\, \operatorname{round}(\mathbf{L}^{-1} \mathbf{r})

实现说明：本模块内部在代码中使用“行向量右乘”的等价实现。
例如上式可写作 :math:`\mathbf{s}^{\top} = \mathbf{r}^{\top}\,\mathbf{L}^{-1}` 与
:math:`\mathbf{r}^{\top} = \mathbf{s}^{\top}\,\mathbf{L}^{\top}`。

Classes
-------
Atom
    表示单个原子，包含位置、速度、力等属性
Cell
    表示晶胞结构，管理原子集合和晶格矢量

Functions
---------
_apply_pbc_numba
    JIT优化的周期性边界条件应用函数

Notes
-----
本模块是ThermoElasticSim的核心组件，为分子动力学模拟提供基础数据结构。
所有长度单位为埃(Å)，时间单位为飞秒(fs)，能量单位为电子伏特(eV)。

Examples
--------
创建简单的铝晶胞：

>>> from thermoelasticsim.core.structure import Atom, Cell
>>> import numpy as np
>>> # 创建FCC铝的原胞
>>> a = 4.05  # 晶格常数
>>> lattice = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
>>> atoms = [
...     Atom(0, "Al", 26.98, [0, 0, 0]),
...     Atom(1, "Al", 26.98, [a/2, a/2, 0]),
...     Atom(2, "Al", 26.98, [a/2, 0, a/2]),
...     Atom(3, "Al", 26.98, [0, a/2, a/2])
... ]
>>> cell = Cell(lattice, atoms)


"""

__version__ = "4.0.0"
__author__ = "Gilbert Young"
__license__ = "GPL-3.0"
__copyright__ = "Copyright 2025, Gilbert Young"

import logging

import numpy as np
from numba import jit

from thermoelasticsim.utils.utils import AMU_TO_EVFSA2, KB_IN_EV

# 配置日志记录
logger = logging.getLogger(__name__)


@jit(nopython=True)
def _apply_pbc_numba(positions, lattice_inv, lattice_vectors):
    """JIT优化的周期性边界条件应用

    Parameters
    ----------
    positions : numpy.ndarray
        原子位置数组 (N, 3)
    lattice_inv : numpy.ndarray
        晶格逆矩阵 (3, 3)
    lattice_vectors : numpy.ndarray
        晶格矢量矩阵 (3, 3)

    Returns
    -------
    numpy.ndarray
        应用PBC后的位置数组
    """
    fractional = np.dot(positions, lattice_inv)
    fractional = fractional % 1.0
    return np.dot(fractional, lattice_vectors.T)


class Atom:
    r"""原子对象，分子动力学模拟的基本单元

    表示一个原子的完整状态，包括位置、速度、受力等物理属性。
    在MD模拟中，原子遵循牛顿运动方程：

    .. math::
        m \frac{d^2\mathbf{r}}{dt^2} = \mathbf{F}

    Parameters
    ----------
    id : int
        原子的唯一标识符
    symbol : str
        元素符号 (如 'Al', 'Cu', 'Si')
    mass_amu : float
        原子质量，原子质量单位 (amu)
    position : array_like
        初始位置，3D笛卡尔坐标 (Å)
    velocity : array_like, optional
        初始速度，3D笛卡尔坐标 (Å/fs)，默认为零

    Attributes
    ----------
    id : int
        原子唯一标识符
    symbol : str
        元素符号
    mass_amu : float
        原始质量 (amu)
    mass : float
        转换后的质量 (eV·fs²/Å²)
    position : numpy.ndarray
        当前位置向量 (3,)
    velocity : numpy.ndarray
        当前速度向量 (3,)
    force : numpy.ndarray
        当前受力向量 (3,)

    Notes
    -----
    质量与常数采用全局统一定义（参见 ``thermoelasticsim.utils.utils`` 模块）：

    ``AMU_TO_EVFSA2``
        质量单位转换常量（amu → eV·fs²/Å²）。

    ``KB_IN_EV``
        玻尔兹曼常数（eV/K）。

    Examples
    --------
    创建铝原子：

    >>> atom = Atom(id=1, symbol="Al", mass_amu=26.98, position=[0, 0, 0])
    >>> atom.velocity = np.array([1.0, 0.0, 0.0])  # 设置速度
    >>> print(f"动能: {0.5 * atom.mass * np.dot(atom.velocity, atom.velocity):.4f} eV")
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        mass_amu: float,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
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

        Parameters
        ----------
        displacement : numpy.ndarray | array_like
            位置增量向量，形状为 (3,)

        Raises
        ------
        ValueError
            如果位置增量不是3D向量

        Examples
        --------
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

        Parameters
        ----------
        velocity_change : numpy.ndarray | array_like
            速度增量向量，形状为 (3,)

        Raises
        ------
        ValueError
            如果速度增量不是3D向量

        Examples
        --------
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

        Returns
        -------
        Atom
            新的 Atom 对象，包含当前原子的所有属性的副本

        Examples
        --------
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
    r"""晶胞对象，管理原子集合和晶格结构

    晶胞是分子动力学模拟的基本容器，定义了模拟盒子的几何形状
    和其中包含的原子。支持周期性边界条件和最小镜像约定。

    晶格矢量定义为行向量：

    .. math::
        \mathbf{L} = \begin{pmatrix}
            \mathbf{a}_1 \\
            \mathbf{a}_2 \\
            \mathbf{a}_3
        \end{pmatrix}

    晶胞体积计算：

    .. math::
        V = |\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)|

    Parameters
    ----------
    lattice_vectors : array_like
        3×3晶格矢量矩阵，每行为一个晶格矢量 (Å)
    atoms : list of Atom
        晶胞中的原子列表
    pbc_enabled : bool, optional
        是否启用周期性边界条件，默认True

    Attributes
    ----------
    lattice_vectors : numpy.ndarray
        晶格矢量矩阵 (3, 3)
    atoms : list of Atom
        原子对象列表
    volume : float
        晶胞体积 (Å³)
    pbc_enabled : bool
        周期性边界条件标志
    lattice_locked : bool
        晶格锁定标志（用于内部弛豫）
    lattice_inv : numpy.ndarray
        晶格逆矩阵，用于坐标转换
    num_atoms : int
        原子数量（属性）

    Methods
    -------
    apply_periodic_boundary(positions)
        应用周期性边界条件
    minimum_image(displacement)
        计算最小镜像位移
    calculate_temperature()
        计算瞬时温度
    build_supercell(repetition)
        构建超胞

    Notes
    -----
    分数/笛卡尔坐标转换（列向量记号）：

    .. math::
        \mathbf{r} = \mathbf{L}\,\mathbf{s},\qquad
        \mathbf{s} = \mathbf{L}^{-1}\,\mathbf{r}

    实现采用“行向量右乘”的等价式：

    .. math::
        \mathbf{s}^{\top} = \mathbf{r}^{\top}\,\mathbf{L}^{-1},\qquad
        \mathbf{r}^{\top} = \mathbf{s}^{\top}\,\mathbf{L}^{\top}

    Examples
    --------
    创建FCC铝晶胞：

    >>> a = 4.05  # 晶格常数
    >>> lattice = a * np.eye(3)  # 立方晶格
    >>> atoms = [
    ...     Atom(0, "Al", 26.98, [0, 0, 0]),
    ...     Atom(1, "Al", 26.98, [a/2, a/2, 0]),
    ...     Atom(2, "Al", 26.98, [a/2, 0, a/2]),
    ...     Atom(3, "Al", 26.98, [0, a/2, a/2])
    ... ]
    >>> cell = Cell(lattice, atoms)
    >>> print(f"晶胞体积: {cell.volume:.2f} Å³")
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
        return not np.linalg.det(lattice_vectors) <= 0

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
            晶胞体积 (Å³)
        """
        return np.linalg.det(self.lattice_vectors)

    def get_box_lengths(self) -> np.ndarray:
        """返回模拟盒子在 x、y、z 方向的长度

        Returns
        -------
        numpy.ndarray
            包含三个方向长度的数组 (Å)
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
            3×3 应力张量矩阵 (eV/Å³)
        """
        from thermoelasticsim.elastic.mechanics import StressCalculator

        stress_calculator = StressCalculator()
        return stress_calculator.compute_stress(self, potential)

    def lock_lattice_vectors(self) -> None:
        """锁定晶格向量，防止在优化过程中被修改

        Notes
        -----
        锁定后，晶格向量将不能被修改，直到调用 :meth:`unlock_lattice_vectors`。
        """
        self.lattice_locked = True
        logger.debug("Lattice vectors have been locked.")

    def unlock_lattice_vectors(self) -> None:
        """解锁晶格向量，允许在需要时修改

        Notes
        -----
        解锁后，晶格向量可以被修改，直到再次调用 :meth:`lock_lattice_vectors`。
        """
        self.lattice_locked = False
        logger.debug("Lattice vectors have been unlocked.")

    def apply_deformation(self, deformation_matrix: np.ndarray) -> None:
        r"""对晶格与原子坐标施加形变矩阵 :math:`\mathbf{F}`

        Parameters
        ----------
        deformation_matrix : numpy.ndarray
            形变矩阵 :math:`\mathbf{F}`，形状为 (3, 3)

        Notes
        -----
        采用列向量约定的连续介质力学记号：

        - 晶格未锁定（允许变胞）时：

          .. math::
             \mathbf{L}_{\text{new}} = \mathbf{F}\, \mathbf{L}_{\text{old}},\qquad
             \mathbf{r}_{\text{new}} = \mathbf{F}\, \mathbf{r}_{\text{old}}

        - 晶格已锁定（固定胞形）时：

          .. math::
             \mathbf{L}_{\text{new}} = \mathbf{L}_{\text{old}},\qquad
             \mathbf{r}_{\text{new}} = \mathbf{F}\, \mathbf{r}_{\text{old}}

        实现细节：本实现使用“行向量右乘”形式，等价写法为

        .. math::
           \mathbf{r}_{\text{new}}^{\top} = \mathbf{r}_{\text{old}}^{\top} \mathbf{F}^{\top}.

        Examples
        --------
        >>> import numpy as np
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
        r"""应用周期性边界条件

        将原子位置映射回主晶胞内，使用标准的最小镜像约定。

        算法（列向量记号）：

        .. math::
            \mathbf{s} = \mathbf{L}^{-1}\,\mathbf{r},\qquad
            \mathbf{s}' = \mathbf{s} - \lfloor \mathbf{s} + 0.5 \rfloor,\qquad
            \mathbf{r}' = \mathbf{L}\,\mathbf{s}'

        Parameters
        ----------
        positions : numpy.ndarray
            原子位置坐标，形状为 (3,) 或 (N, 3)

        Returns
        -------
        numpy.ndarray
            应用PBC后的位置，保持输入形状

        Raises
        ------
        ValueError
            如果位置数组形状不正确或包含非有限值

        Notes
        -----
        该实现使用分数坐标进行周期性映射，确保数值稳定性。
        对于三斜晶系也能正确处理。

        Examples
        --------
        >>> # 单个原子位置
        >>> pos = np.array([5.0, 6.0, 7.0])
        >>> new_pos = cell.apply_periodic_boundary(pos)
        >>> # 批量处理
        >>> positions = np.random.randn(100, 3) * 10
        >>> new_positions = cell.apply_periodic_boundary(positions)
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
        r"""计算系统的瞬时温度

        使用均分定理计算温度，扣除质心运动贡献：

        .. math::
            T = \frac{2 E_{kin}}{k_B \cdot N_{dof}}

        其中动能（扣除质心）：

        .. math::
            E_{kin} = \sum_i \frac{1}{2} m_i |\mathbf{v}_i - \mathbf{v}_{cm}|^2

        自由度数定义：

        .. math::
            N_{\mathrm{dof}} = \begin{cases}
                3N - 3, & N > 1, \\
                3, & N = 1.
            \end{cases}

        Returns
        -------
        float
            系统温度，单位为 :math:`\mathrm{K}`

        Notes
        -----
        瞬时温度会有涨落，通常需要时间平均来获得稳定值。
        对于NVE系综，温度是守恒量的函数。

        Examples
        --------
        >>> temp = cell.calculate_temperature()
        >>> print(f"瞬时温度: {temp:.1f} K")
        >>> # 计算时间平均温度
        >>> temps = [cell.calculate_temperature() for _ in range(1000)]
        >>> avg_temp = np.mean(temps)

        See Also
        --------
        calculate_kinetic_energy : 计算总动能（含质心运动）
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
        r"""计算当前系统的总动能

        Returns
        -------
        float
            系统总动能，单位为 :math:`\mathrm{eV}`

        Notes
        -----
            计算所有原子的动能总和，包含质心运动

        Examples
        --------
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

        Returns
        -------
        numpy.ndarray
            原子位置数组，形状为 (num_atoms, 3)

        Examples
        --------
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
        r"""计算最小镜像位移向量

        根据最小镜像约定，找到最近的周期镜像之间的位移。
        这对于正确计算周期性系统中的距离和力至关重要。

        数学原理（列向量记号）：

        .. math::
            \mathbf{d}_{\min} = \mathbf{d} - \mathbf{L}\, \operatorname{round}(\mathbf{L}^{-1} \mathbf{d})

        Parameters
        ----------
        displacement : numpy.ndarray
            原始位移向量 (3,)

        Returns
        -------
        numpy.ndarray
            最小镜像位移向量 (3,)

        Raises
        ------
        ValueError
            如果位移向量不是3D或包含非有限值

        Notes
        -----
        该方法确保返回的位移向量长度最小，对应于最近的周期镜像。
        在计算原子间相互作用时必须使用此方法。

        Examples
        --------
        >>> # 计算两原子间的最小距离
        >>> r12 = atom2.position - atom1.position
        >>> r12_min = cell.minimum_image(r12)
        >>> distance = np.linalg.norm(r12_min)

        See Also
        --------
        apply_periodic_boundary : 应用周期性边界条件
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
        r"""获取所有原子的分数坐标

        Returns
        -------
        numpy.ndarray
            原子分数坐标数组，形状为(num_atoms, 3)

        Notes
        -----
        坐标关系（列向量记号）：

        .. math::
            \mathbf{s} = \mathbf{L}^{-1}\,\mathbf{r}

        代码实现采用行向量右乘：:math:`\mathbf{s}^{\top} = \mathbf{r}^{\top}\,\mathbf{L}^{-1}`。
        """
        positions = self.get_positions()  # shape (N, 3)
        # 转换到分数坐标：(L^T)^-1 * r
        fractional_coords = np.dot(positions, self.lattice_inv)
        return fractional_coords

    def set_fractional_coordinates(self, fractional_coords: np.ndarray) -> None:
        r"""根据分数坐标设置所有原子的笛卡尔坐标

        Parameters
        ----------
        fractional_coords : numpy.ndarray
            分数坐标数组，形状为(num_atoms, 3)

        Notes
        -----
        坐标关系（列向量记号）：

        .. math::
            \mathbf{r} = \mathbf{L}\,\mathbf{s}

        代码实现采用行向量右乘：:math:`\mathbf{r}^{\top} = \mathbf{s}^{\top}\,\mathbf{L}^{\top}`。
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
        # 提醒：部分算法需要访问最近使用的势能对象以计算扩展哈密顿量
        # 若上层调用提供了 potential，可临时缓存到 Cell（教学用途）。
        # 具体设置由算法调用者完成：cell._last_potential_object = potential

    def get_volume(self) -> float:
        r"""计算晶胞体积

        Returns
        -------
        float
            晶胞体积 (Å³)

        Notes
        -----
        体积通过晶格矢量的标量三重积计算：

        .. math::
           V = \left|\, \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) \,\right|
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
        except np.linalg.LinAlgError as e:
            raise ValueError("晶格矢量矩阵不可逆，可能存在线性相关的矢量") from e
