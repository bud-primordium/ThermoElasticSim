# src/python/interfaces/cpp_interface.py

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
from python.utils import AMU_TO_EVFSA2  # 确保正确导入单位转换常量


class CppInterface:
    """
    @class CppInterface
    @brief 用于调用 C++ 实现的函数的接口类。
    """

    def __init__(self, lib_name):
        """
        @param lib_name 库的名称，不包括前缀和扩展名
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
        if lib_name == "stress_calculator":
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

        elif lib_name == "nose_hoover":
            self.lib.nose_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.c_double,  # xi
                ctypes.c_double,  # Q
                ctypes.c_double,  # target_temperature
            ]
            self.lib.nose_hoover.restype = ctypes.c_double

        elif lib_name == "nose_hoover_chain":
            self.lib.nose_hoover_chain.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi_chain
                ctypes.c_double,  # Q
                ctypes.c_double,  # target_temperature
            ]
            self.lib.nose_hoover_chain.restype = None

        else:
            raise ValueError(f"未知的库名称: {lib_name}")

    def compute_stress(
        self,
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        box_lengths,
        stress_tensor,
    ):
        """
        计算应力张量。
        """
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
        """
        计算系统的力。
        """
        self.lib.calculate_forces(
            num_atoms,
            positions,
            forces,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
        )

    def nose_hoover(
        self,
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi,
        Q,
        target_temperature,
    ):
        """
        实现 Nose-Hoover 恒温器算法
        """
        updated_xi = self.lib.nose_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi,
            Q,
            target_temperature,
        )
        return updated_xi

    def nose_hoover_chain(
        self,
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi_chain,
        Q,
        target_temperature,
    ):
        """
        实现 Nose-Hoover 链恒温器算法
        """
        self.lib.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi_chain,
            Q,
            target_temperature,
        )
