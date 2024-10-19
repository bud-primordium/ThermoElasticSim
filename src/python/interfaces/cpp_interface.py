# 文件名: cpp_interface.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 用于在 Python 中调用 C++ 实现的接口类。

"""
接口模块。

该模块定义了 `CppInterface` 类，用于通过 ctypes 调用外部 C++ 函数库。
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import sys

# from ..utils import AMU_TO_EVFSA2  # 如果需要，确保导入单位转换常量


class CppInterface:
    """
    @class CppInterface
    @brief 用于调用 C++ 实现的函数的接口类。

    Parameters
    ----------
    lib_name : str
        库的名称，不包括前缀和扩展名。
    """

    def __init__(self, lib_name):
        if os.name == "nt":  # Windows
            lib_extension = ".dll"
            lib_prefix = ""

            # 判断是否使用 MinGW 环境
            if "GCC" in sys.version:
                lib_prefix = "lib"  # MinGW 使用 'lib' 前缀
        elif sys.platform == "darwin":  # macOS
            lib_extension = ".dylib"
            lib_prefix = "lib"
        else:  # Unix/Linux
            lib_extension = ".so"
            lib_prefix = "lib"

        lib_path = os.path.join(
            os.path.abspath("../../lib"), lib_prefix + lib_name + lib_extension
        )

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
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ctypes.c_double,  # volume
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # stress_tensor
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
                ctypes.POINTER(ctypes.c_double),  # xi (input/output)
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
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi_chain
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # Q
                ctypes.c_int,  # chain_length
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

        Parameters
        ----------
        num_atoms : int
            原子数。
        positions : numpy.ndarray
            原子位置数组。
        velocities : numpy.ndarray
            原子速度数组。
        forces : numpy.ndarray
            原子受力数组。
        masses : numpy.ndarray
            原子质量数组。
        volume : float
            系统体积。
        box_lengths : numpy.ndarray
            盒子尺寸。
        stress_tensor : numpy.ndarray
            输出的应力张量。
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
        self, num_atoms, positions, epsilon, sigma, cutoff, box_lengths
    ):
        """
        计算系统的总 Lennard-Jones 势能。

        Parameters
        ----------
        num_atoms : int
            原子数。
        positions : numpy.ndarray
            原子位置数组。
        epsilon : float
            Lennard-Jones 势参数 ε。
        sigma : float
            Lennard-Jones 势参数 σ。
        cutoff : float
            截断距离。
        box_lengths : numpy.ndarray
            盒子尺寸。

        Returns
        -------
        float
            总 Lennard-Jones 势能。
        """
        energy = self.lib.calculate_energy(
            num_atoms, positions, epsilon, sigma, cutoff, box_lengths
        )
        return energy

    def calculate_forces(
        self, num_atoms, positions, forces, epsilon, sigma, cutoff, box_lengths
    ):
        """
        计算系统的力。

        Parameters
        ----------
        num_atoms : int
            原子数。
        positions : numpy.ndarray
            原子位置数组。
        forces : numpy.ndarray
            输出的力数组。
        epsilon : float
            Lennard-Jones 势参数 ε。
        sigma : float
            Lennard-Jones 势参数 σ。
        cutoff : float
            截断距离。
        box_lengths : numpy.ndarray
            盒子尺寸。
        """
        self.lib.calculate_forces(
            num_atoms, positions, forces, epsilon, sigma, cutoff, box_lengths
        )

    def nose_hoover(
        self, dt, num_atoms, masses, velocities, forces, xi_array, Q, target_temperature
    ):
        """
        实现 Nose-Hoover 恒温器算法。

        Parameters
        ----------
        dt : float
            时间步长。
        num_atoms : int
            原子数。
        masses : numpy.ndarray
            原子质量数组。
        velocities : numpy.ndarray
            原子速度数组。
        forces : numpy.ndarray
            原子受力数组。
        xi_array : numpy.ndarray
            Nose-Hoover 热浴变量数组。
        Q : float
            热浴质量参数。
        target_temperature : float
            目标温度。

        Returns
        -------
        float
            更新后的 Nose-Hoover 热浴变量。
        """
        if not isinstance(xi_array, np.ndarray) or xi_array.size != 1:
            raise ValueError("xi must be a numpy array with one element.")
        self.lib.nose_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Q,
            target_temperature,
        )
        return xi_array[0]

    def nose_hoover_chain(
        self,
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi_chain,
        Q,
        chain_length,
        target_temperature,
    ):
        """
        实现 Nose-Hoover 链恒温器算法。

        Parameters
        ----------
        dt : float
            时间步长。
        num_atoms : int
            原子数。
        masses : numpy.ndarray
            原子质量数组。
        velocities : numpy.ndarray
            原子速度数组。
        forces : numpy.ndarray
            原子受力数组。
        xi_chain : numpy.ndarray
            Nose-Hoover 链的热浴变量数组。
        Q : numpy.ndarray
            热浴质量参数数组。
        chain_length : int
            Nose-Hoover 链的长度。
        target_temperature : float
            目标温度。
        """
        self.lib.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi_chain,
            Q,
            chain_length,
            target_temperature,
        )
