# 文件名: potentials.py
# 作者: Gilbert Young
# 修改日期: 2024年10月19日
# 文件描述: 实现 Lennard-Jones 势能及其相关方法。

"""
势能模块。

包含 Potential 基类和 LennardJonesPotential 类，用于计算原子间的 Lennard-Jones 势能。
"""

import numpy as np
from .interfaces.cpp_interface import CppInterface


class Potential:
    """
    势能基类，定义势能计算的接口。

    Parameters
    ----------
    parameters : dict
        势能相关的参数。
    cutoff : float
        势能的截断距离，单位为 Å。
    """

    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    def calculate_potential(self, cell):
        """
        计算势能，需子类实现。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        """
        raise NotImplementedError

    def calculate_forces(self, cell):
        """
        计算作用力，需子类实现。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        """
        raise NotImplementedError

    def calculate_energy(self, cell):
        """
        计算能量，需子类实现。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        """
        raise NotImplementedError


class LennardJonesPotential(Potential):
    """
    Lennard-Jones 势的实现。

    Parameters
    ----------
    epsilon : float
        Lennard-Jones 势深度参数，单位为 eV。
    sigma : float
        Lennard-Jones 势的零势距，单位为 Å。
    cutoff : float
        Lennard-Jones 势的截断距离，单位为 Å。
    """

    def __init__(self, epsilon, sigma, cutoff):
        # 初始化父类参数
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.epsilon = epsilon  # 势能深度，单位 eV
        self.sigma = sigma  # 势能零距，单位 Å
        self.cutoff = cutoff  # 截断距离，单位 Å
        self.cpp_interface = CppInterface("lennard_jones")

    def calculate_forces(self, cell):
        """
        计算并更新所有原子的作用力。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        """
        num_atoms = cell.num_atoms
        # 获取所有原子的位置信息并转换为连续的 NumPy 数组
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        # 初始化力数组
        forces = np.zeros_like(positions, dtype=np.float64)
        # 获取盒子长度信息
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 调用 C++ 接口计算作用力
        self.cpp_interface.calculate_forces(
            num_atoms,
            positions,
            forces,
            self.epsilon,
            self.sigma,
            self.cutoff,
            box_lengths,
        )

        # 更新原子力，按原子顺序存储计算结果
        forces = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[i]

    def calculate_energy(self, cell):
        """
        计算系统的总能量。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。

        Returns
        -------
        float
            计算的总势能，单位为 eV。
        """
        num_atoms = cell.num_atoms
        # 获取所有原子的位置信息并转换为连续的 NumPy 数组
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        # 获取盒子长度信息
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        # 调用 C++ 接口计算能量
        energy = self.cpp_interface.calculate_energy(
            num_atoms,
            positions,
            self.epsilon,
            self.sigma,
            self.cutoff,
            box_lengths,
        )
        return energy
