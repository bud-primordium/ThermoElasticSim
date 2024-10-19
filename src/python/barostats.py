# 文件名: barostats.py
# 作者: Gilbert Young
# 修改日期: 2024-10-19
# 文件描述: 实现分子动力学模拟中的 Parrinello-Rahman-Hoover 恒压器。

"""
恒压器模块

包含 Barostat 基类和 Parrinello-Rahman-Hoover (PRH) 恒压器的实现
"""

import numpy as np
from .interfaces.cpp_interface import CppInterface
from .mechanics import StressCalculatorLJ
from .structure import Cell


class Barostat:
    """
    恒压器基类，定义恒压器的接口
    """

    def apply(self, cell, dt):
        """应用恒压器，更新晶胞参数"""
        raise NotImplementedError


class ParrinelloRahmanHooverBarostat(Barostat):
    """
    Parrinello-Rahman-Hoover (PRH) 恒压器的实现

    Parameters
    ----------
    target_pressure : float
        目标压力
    time_constant : float
        控制压力调节的时间常数
    Qp : array_like, optional
        压力热浴质量矩阵，默认为 time_constant^2 的数组
    """

    def __init__(self, target_pressure, time_constant, Qp=None):
        self.target_pressure = target_pressure
        self.time_constant = time_constant
        self.Qp = Qp if Qp is not None else np.ones(6) * (time_constant**2)
        self.xi = np.zeros(6)  # 热浴变量数组
        self.cpp_interface = CppInterface("parrinello_rahman_hoover")

    def apply(self, cell, dt):
        """
        应用 PRH 恒压器，更新晶胞的晶格矢量和原子速度

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        dt : float
            时间步长
        """
        num_atoms = len(cell.atoms)
        masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.array(
            [atom.force for atom in cell.atoms], dtype=np.float64
        ).flatten()
        lattice_vectors = cell.lattice_vectors.flatten()

        # 调用 C++ PRH 函数
        self.cpp_interface.parrinello_rahman_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            lattice_vectors,
            self.xi,
            self.Qp,
            self.target_pressure,
        )

        # 更新晶格矢量
        cell.lattice_vectors = lattice_vectors.reshape((3, 3))

        # 更新原子速度
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[3 * i : 3 * i + 3]
