# src/python/barostats.py

import numpy as np
from .interfaces.cpp_interface import CppInterface
from .mechanics import StressCalculatorLJ


class Barostat:
    """
    @class Barostat
    @brief 恒压器基类
    """

    def apply(self, cell, dt):
        raise NotImplementedError


class ParrinelloRahmanHooverBarostat(Barostat):
    """
    @class ParrinelloRahmanHooverBarostat
    @brief Parrinello-Rahman-Hoover (PRH) 恒压器的实现
    """

    def __init__(self, target_pressure, time_constant, Qp=None):
        self.target_pressure = target_pressure
        self.time_constant = time_constant
        self.Qp = Qp if Qp is not None else np.ones(6) * (time_constant**2)
        self.xi = np.zeros(6)  # 热浴变量数组
        self.cpp_interface = CppInterface("parrinello_rahman_hoover")

    def apply(self, cell, dt):
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

        # 更新 lattice_vectors
        cell.lattice_vectors = lattice_vectors.reshape((3, 3))

        # 更新 atom velocities
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[3 * i : 3 * i + 3]
