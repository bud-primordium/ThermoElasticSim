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
        elif lib_name == "nose_hoover_chain":
            self.lib.nose_hoover_chain.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
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
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
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
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
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

        返回更新后的 xi。
        """
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

    def nose_hoover_chain(
        self,
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi,
        Q,
        chain_length,
        target_temperature,
    ):
        """
        调用 C++ 实现的 Nose-Hoover 链恒温器。

        返回更新后的 xi。
        """
        self.lib.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi,
            Q,
            chain_length,
            target_temperature,
        )
        return xi  # xi 已在 C++ 中更新

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
        调用 C++ 实现的 Parrinello-Rahman-Hoover 恒温器。
        """
        self.lib.parrinello_rahman_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            lattice_vectors,
            xi,
            Q,
            target_pressure,
        )
        # xi 已在 C++ 中更新

    def calculate_forces(
        self,
        num_atoms,
        positions,
        forces,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
    ):
        self.lib.calculate_forces(
            num_atoms,
            positions,
            forces,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
        )

    def calculate_energy(
        self,
        num_atoms,
        positions,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
    ):
        energy = self.lib.calculate_energy(
            num_atoms,
            positions,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
        )
        return energy

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
        stress_tensor = np.zeros(9, dtype=np.float64)
        self.lib.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            box_lengths,
            stress_tensor,
        )
        return stress_tensor.reshape((3, 3))
