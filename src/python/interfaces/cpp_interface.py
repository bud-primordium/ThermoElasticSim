# src/python/interfaces/cpp_interface.py

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
from python.utils import AMU_TO_EVFSA2  # 导入单位转换常量


class CppInterface:
    """
    @class CppInterface
    @brief 用于调用 C++ 实现的函数的接口类。
    """

    def __init__(self, lib_name):
        """
        @param lib_name 库的名称
        """
        if os.name == "nt":  # Windows
            lib_extension = ".dll"
            lib_prefix = ""
        else:  # Unix/Linux
            lib_extension = ".so"
            lib_prefix = "lib"
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录的绝对路径
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        # 构建库文件的绝对路径
        lib_path = os.path.join(
            project_root, "lib", lib_prefix + lib_name + lib_extension
        )
        # 检查库文件是否存在
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"无法找到库文件: {lib_path}")
        self.lib = ctypes.CDLL(lib_path)

        # 配置不同库的函数签名
        if lib_name == "nose_hoover":
            self.lib.nose_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (amu)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.POINTER(ctypes.c_double),  # xi
                ctypes.c_double,  # Q
                ctypes.c_double,  # target_temperature
            ]
            self.lib.nose_hoover.restype = None
        elif lib_name == "nose_hoover_chain":
            self.lib.nose_hoover_chain.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (amu)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # Q
                ctypes.c_int,  # chain_length
                ctypes.c_double,  # target_temperature
            ]
            self.lib.nose_hoover_chain.restype = None
        elif lib_name == "parrinello_rahman_hoover":
            self.lib.parrinello_rahman_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (amu)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # lattice_vectors
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # Q
                ctypes.c_double,  # target_pressure
            ]
            self.lib.parrinello_rahman_hoover.restype = None
        elif lib_name == "lennard_jones":
            self.lib.calculate_forces.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
            ]
            self.lib.calculate_forces.restype = None

            self.lib.calculate_energy.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
            ]
            self.lib.calculate_energy.restype = ctypes.c_double
        elif lib_name == "stress_calculator":
            self.lib.compute_stress.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (amu)
                ctypes.c_double,  # volume
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # stress_tensor (output)
            ]
            self.lib.compute_stress.restype = None

    def nose_hoover(
        self, dt, num_atoms, masses, velocities, forces, xi, Q, target_temperature
    ):
        """
        调用 C++ 实现的 Nose-Hoover 恒温器。

        masses: array in amu
        """
        # 转换质量单位
        masses_converted = masses * AMU_TO_EVFSA2
        xi_c = ctypes.c_double(xi)
        self.lib.nose_hoover(
            dt,
            num_atoms,
            masses_converted,
            velocities,
            forces,
            ctypes.byref(xi_c),
            Q,
            target_temperature,
        )
        return xi_c.value

    def parrinello_rahman_hoover(
        self,
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        lattice_vectors,
        xi,
        Q,
        target_pressure,
    ):
        """
        调用 C++ 实现的 Parrinello-Rahman-Hoover 恒压器。

        masses: array in amu
        """
        # 转换质量单位
        masses_converted = masses * AMU_TO_EVFSA2
        self.lib.parrinello_rahman_hoover(
            dt,
            num_atoms,
            masses_converted,
            velocities,
            forces,
            lattice_vectors,
            xi,
            Q,
            target_pressure,
        )
        # xi 和 lattice_vectors 已在 C++ 中更新

    def compute_stress(
        self,
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        box_lengths,
    ):
        """
        计算应力张量。

        masses: array in amu
        """
        # 转换质量单位
        masses_converted = masses * AMU_TO_EVFSA2
        stress_tensor = np.zeros(9, dtype=np.float64)
        self.lib.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses_converted,
            volume,
            box_lengths,
            stress_tensor,
        )
        return stress_tensor.reshape((3, 3))

    def calculate_energy(
        self,
        num_atoms,
        positions,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
    ):
        """
        计算系统的总 Lennard-Jones 势能。
        """
        energy = self.lib.calculate_energy(
            num_atoms,
            positions,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
        )
        return energy
