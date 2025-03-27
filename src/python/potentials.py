# 文件名: potentials.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 实现 Lennard-Jones 势能，EAM势能及其相关方法。

"""
势能模块

包含 Potential 基类和 LennardJonesPotential 子类、EAMAl1Potential子类，用于计算原子势能

Classes:
    Potential: 势能计算基类，定义接口
    LennardJonesPotential: Lennard-Jones 势能实现
    EAMAl1Potential: 铝的EAM势能实现
"""

import numpy as np
from .utils import NeighborList
from .interfaces.cpp_interface import CppInterface
from .structure import Cell
import logging

logger = logging.getLogger(__name__)


class Potential:
    """势能计算基类，定义势能计算的接口

    Attributes
    ----------
    parameters : dict
        势能相关的参数字典
    cutoff : float
        势能的截断距离，单位为 Å
    neighbor_list : NeighborList or None
        邻居列表对象，用于计算原子间相互作用
    """

    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff
        self.neighbor_list = None  # 邻居列表

    def calculate_forces(self, cell: Cell) -> None:
        """计算作用力，需子类实现

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Raises
        ------
        NotImplementedError
            如果子类没有实现该方法
        """
        raise NotImplementedError

    def calculate_energy(self, cell: Cell) -> float:
        """计算能量，需子类实现

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Returns
        -------
        float
            计算的总势能，单位为eV

        Raises
        ------
        NotImplementedError
            如果子类没有实现该方法
        """
        raise NotImplementedError

    def set_neighbor_list(self, neighbor_list: "NeighborList") -> None:
        """设置邻居列表

        Parameters
        ----------
        neighbor_list : NeighborList
            邻居列表对象
        """
        self.neighbor_list = neighbor_list


class LennardJonesPotential(Potential):
    """Lennard-Jones 势的实现

    Attributes
    ----------
    parameters : dict
        包含epsilon和sigma参数的字典
    cutoff : float
        势能截断距离，单位为Å
    cpp_interface : CppInterface
        用于调用C++实现的接口
    neighbor_list : NeighborList
        邻居列表对象
    """

    def __init__(self, epsilon, sigma, cutoff):
        # 初始化父类参数
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.cpp_interface = CppInterface("lennard_jones")
        # 自动创建邻居列表
        self.neighbor_list = NeighborList(cutoff=self.cutoff)
        logger.debug(
            f"Lennard-Jones Potential initialized with epsilon={epsilon}, sigma={sigma}, cutoff={cutoff}."
        )

    def calculate_forces(self, cell: Cell) -> None:
        """计算并更新所有原子的作用力

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Notes
        -----
        使用C++接口进行高性能计算
        """
        # 如果邻居列表未构建，则构建它
        if self.neighbor_list.cell is None:
            logger.debug("Neighbor list not built yet. Building now.")
            self.neighbor_list.build(cell)
        else:
            # 检查是否需要更新邻居列表
            # logger.debug("Updating neighbor list.")
            self.neighbor_list.update()

        num_atoms = cell.num_atoms
        # 展平 避免报错
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 构建邻居对列表
        neighbor_pairs = [
            (i, j)
            for i in range(num_atoms)
            for j in self.neighbor_list.get_neighbors(i)
            if j > i
        ]

        neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
        neighbor_list_array = np.ascontiguousarray(neighbor_list_flat, dtype=np.int32)

        # logger.debug(f"Number of neighbor pairs: {len(neighbor_pairs)}.")

        # 初始化力数组
        forces = np.zeros_like(positions, dtype=np.float64)

        # 调用 C++ 接口计算作用力
        # logger.debug("Calling C++ interface to calculate forces.")
        self.cpp_interface.calculate_lj_forces(
            num_atoms,
            positions,
            forces,
            self.parameters["epsilon"],
            self.parameters["sigma"],
            self.cutoff,
            box_lengths,
            neighbor_list_array,
            len(neighbor_pairs),
        )

        # 更新原子力，按原子顺序存储计算结果
        forces = forces.reshape((num_atoms, 3))
        # logger.debug("Updating atomic forces.")
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[i]

        # logger.debug("Forces calculation and update completed.")

    def calculate_energy(self, cell: Cell) -> float:
        """计算系统的总能量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Returns
        -------
        float
            计算的总势能，单位为 eV

        Notes
        -----
        使用C++接口进行高性能计算
        """
        if self.neighbor_list.cell is None:
            logger.debug("Neighbor list not built yet. Building now.")
            self.neighbor_list.build(cell)
        else:
            # 检查是否需要更新邻居列表
            # logger.debug("Updating neighbor list.")
            self.neighbor_list.update()

        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 构建邻居对列表
        neighbor_pairs = [
            (i, j)
            for i in range(num_atoms)
            for j in self.neighbor_list.get_neighbors(i)
            if j > i
        ]

        neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
        neighbor_list_array = np.ascontiguousarray(neighbor_list_flat, dtype=np.int32)

        # logger.debug(
        #     f"Number of neighbor pairs for energy calculation: {len(neighbor_pairs)}."
        # )

        # 调用 C++ 接口计算能量
        # logger.debug("Calling C++ interface to calculate energy.")
        energy = self.cpp_interface.calculate_lj_energy(
            num_atoms,
            positions,
            self.parameters["epsilon"],
            self.parameters["sigma"],
            self.cutoff,
            box_lengths,
            neighbor_list_array,
            len(neighbor_pairs),
        )
        # logger.debug(f"Calculated potential energy: {energy} eV.")
        return energy


class EAMAl1Potential(Potential):
    """Al1 EAM (Embedded Atom Method) 势的实现

    基于 Mendelev et al. (2008) 的参数化。该势包含三个主要部分：
    1. 对势项 φ(r)
    2. 电子密度贡献 ψ(r)
    3. 嵌入能 F(ρ)

    Attributes
    ----------
    parameters : dict
        包含cutoff和type参数的字典
    cutoff : float
        势能截断距离，单位为Å
    cpp_interface : CppInterface
        用于调用C++实现的接口

    Notes
    -----
    总能量表达式：E = Σ_i Σ_{j>i} φ(r_ij) + Σ_i F(ρ_i)
    其中 ρ_i = Σ_{j≠i} ψ(r_ij)
    """

    def __init__(self, cutoff=6.5):
        parameters = {"cutoff": cutoff, "type": "Al1"}
        super().__init__(parameters=parameters, cutoff=cutoff)
        self.cpp_interface = CppInterface("eam_al1")
        logger.debug(f"EAM Al1 Potential initialized with cutoff={cutoff}.")

    def calculate_forces(self, cell: Cell) -> None:
        """计算并更新所有原子的作用力

        使用完整的EAM表达式计算力：
        F_i = -∇_i E = -Σ_j [φ'(r_ij) + (F'(ρ_i) + F'(ρ_j))ψ'(r_ij)] * r_ij/r_ij

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Notes
        -----
        使用C++接口进行高性能计算
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 初始化力数组
        forces = np.zeros_like(positions, dtype=np.float64)

        # 调用C++接口计算力
        self.cpp_interface.calculate_eam_al1_forces(
            num_atoms, positions, box_lengths, forces
        )

        # 更新原子力，按原子顺序存储计算结果
        forces = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = -forces[i]
        # 注，在新版本中发现漏洞，EAM应该是负的力

    def calculate_energy(self, cell: Cell) -> float:
        """计算系统的总能量

        包括对势能和嵌入能两部分：
        E = Σ_i Σ_{j>i} φ(r_ij) + Σ_i F(ρ_i)

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Returns
        -------
        float
            计算的总势能，单位为 eV

        Notes
        -----
        使用C++接口进行高性能计算
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 调用C++接口计算能量
        energy = self.cpp_interface.calculate_eam_al1_energy(
            num_atoms, positions, box_lengths
        )
        logger.debug(f"Calculated EAM potential energy: {energy} eV.")

        return energy

    def set_neighbor_list(self, neighbor_list: "NeighborList") -> None:
        """重写该方法以避免使用邻居列表

        Parameters
        ----------
        neighbor_list : NeighborList
            邻居列表对象，将被忽略

        Notes
        -----
        EAM Al1势不使用邻居列表，此方法仅用于兼容接口
        """
        logger.warning(
            "EAM Al1 potential does not use neighbor lists. This call will be ignored."
        )
        pass
