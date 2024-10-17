# src/python/interfaces/cpp_interface.py

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os


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

        if lib_name == "nose_hoover":
            self.lib.nose_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.POINTER(ctypes.c_double),  # xi
                ctypes.c_double,  # Q
                ctypes.c_double,  # target_temperature
            ]
            self.lib.nose_hoover.restype = None
        elif lib_name == "lennard_jones":
            self.lib.calculate_forces.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # lattice_vectors
            ]
            self.lib.calculate_forces.restype = None
        elif lib_name == "stress_calculator":
            self.lib.compute_stress.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ctypes.c_double,  # volume
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # lattice_vectors
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # stress_tensor (output)
            ]
            self.lib.compute_stress.restype = None

    def nose_hoover(
        self, dt, num_atoms, masses, velocities, forces, xi, Q, target_temperature
    ):
        xi_c = ctypes.c_double(xi)
        self.lib.nose_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            ctypes.byref(xi_c),
            Q,
            target_temperature,
        )
        return xi_c.value

    def calculate_forces(
        self,
        num_atoms,
        positions,
        forces,
        epsilon,
        sigma,
        cutoff,
        lattice_vectors,
    ):
        self.lib.calculate_forces(
            num_atoms,
            positions,
            forces,
            epsilon,
            sigma,
            cutoff,
            lattice_vectors,
        )

    def compute_stress(
        self,
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        epsilon,
        sigma,
        cutoff,
        lattice_vectors,
    ):
        stress_tensor = np.zeros(9, dtype=np.float64)
        self.lib.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            epsilon,
            sigma,
            cutoff,
            lattice_vectors,
            stress_tensor,
        )
        return stress_tensor.reshape((3, 3))
