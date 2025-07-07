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
import logging

logger = logging.getLogger(__name__)


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

        # --- 健壮的路径查找 ---
        # 从当前文件位置开始，向上查找项目根目录（以 pyproject.toml 为标志）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = None
        while current_dir != os.path.dirname(current_dir): # 循环直到文件系统的根目录
            if "pyproject.toml" in os.listdir(current_dir):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)

        if project_root is None:
            raise FileNotFoundError("无法定位项目根目录 (未找到 pyproject.toml)。")

        # 从项目根目录构建库文件的绝对路径
        lib_path = os.path.join(
            project_root, "src", "thermoelasticsim", "lib", lib_prefix + lib_name + lib_extension
        )

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"无法找到库文件: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # 配置不同库的函数签名
        if lib_name == "stress_calculator":
            self.lib.compute_stress.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # positions (3*num_atoms)
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # velocities (3*num_atoms)
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # forces (3*num_atoms)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (num_atoms)
                ctypes.c_double,  # volume
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths (3,)
                ndpointer(ctypes.c_double, flags="WRITEABLE"),  # stress_tensor (9,)
            ]
            self.lib.compute_stress.restype = None

        elif lib_name == "lennard_jones":
            self.lib.calculate_lj_forces.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # neighbor_pairs
                ctypes.c_int,  # num_pairs
            ]
            self.lib.calculate_lj_forces.restype = None

            self.lib.calculate_lj_energy.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ctypes.c_double,  # epsilon
                ctypes.c_double,  # sigma
                ctypes.c_double,  # cutoff
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # neighbor_pairs
                ctypes.c_int,  # num_pairs
            ]
            self.lib.calculate_lj_energy.restype = ctypes.c_double

        elif lib_name == "eam_al1":
            self.lib.calculate_eam_al1_forces.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
            ]
            self.lib.calculate_eam_al1_forces.restype = None

            self.lib.calculate_eam_al1_energy.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # box_lengths
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # energy (output)
            ]
            self.lib.calculate_eam_al1_energy.restype = None

        elif lib_name == "nose_hoover":
            self.lib.nose_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # velocities
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # xi_array (修改为 ndpointer)
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

        elif lib_name == "parrinello_rahman_hoover":
            self.lib.parrinello_rahman_hoover.argtypes = [
                ctypes.c_double,  # dt
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # masses (num_atoms)
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # velocities (3*num_atoms)
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # forces (3*num_atoms)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # lattice_vectors (9)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # xi (9)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # Q (9)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # total_stress (9)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # target_pressure (9)
                ctypes.c_double,  # W
            ]
            self.lib.parrinello_rahman_hoover.restype = None

        else:
            raise ValueError(f"未知的库名称: {lib_name}")

    def compute_stress(
        self,
        num_atoms: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        volume: float,
        box_lengths: np.ndarray,
        stress_tensor: np.ndarray,
    ):
        """
        计算应力张量。

        本函数允许输入的 positions, velocities, forces 既可以是 (num_atoms, 3) 也可以是 (3*num_atoms,) 的形状。
        同理，对于 stress_tensor，既可以是 (3,3) 也可以是 (9,)。

        Parameters
        ----------
        num_atoms : int
        positions : np.ndarray
            原子位置数组，可为 (num_atoms, 3) 或 (3*num_atoms, )
        velocities : np.ndarray
            原子速度数组，可为 (num_atoms, 3) 或 (3*num_atoms, )
        forces : np.ndarray
            原子力数组，可为 (num_atoms, 3) 或 (3*num_atoms, )
        masses : np.ndarray
            原子质量数组 (num_atoms,)
        volume : float
            晶胞体积
        box_lengths : np.ndarray
            晶胞长度数组 (3,)
        stress_tensor : np.ndarray
            输出应力张量，可为 (3,3) 或 (9,)

        Returns
        -------
        None
        """

        def ensure_flat(array, length):
            # 此内部函数将 (num_atoms,3) 转为 (3*num_atoms,) 的视图，或检查本就是(3*num_atoms,)
            if array.shape == (length,):
                return array
            elif array.ndim == 2 and array.shape == (length // 3, 3):
                return array.reshape(length)
            else:
                raise ValueError(
                    f"Array shape {array.shape} not compatible with length {length}."
                )

        # 对 positions/velocities/forces 进行统一处理
        positions = ensure_flat(positions, 3 * num_atoms)
        velocities = ensure_flat(velocities, 3 * num_atoms)
        forces = ensure_flat(forces, 3 * num_atoms)

        # masses 和 box_lengths 保持严格要求
        if masses.shape != (num_atoms,):
            raise ValueError(
                f"Expected masses shape {(num_atoms,)}, got {masses.shape}"
            )
        if box_lengths.shape != (3,):
            raise ValueError(
                f"Expected box_lengths shape (3,), got {box_lengths.shape}"
            )

        # 对应力张量同样灵活处理
        if stress_tensor.shape == (3, 3):
            stress_tensor_view = stress_tensor.reshape(9)
        elif stress_tensor.shape == (9,):
            stress_tensor_view = stress_tensor
        else:
            raise ValueError(
                f"stress_tensor must be shape (3,3) or (9,), got {stress_tensor.shape}"
            )

        # 确保所有数组是连续和浮点数类型
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        velocities = np.ascontiguousarray(velocities, dtype=np.float64)
        forces = np.ascontiguousarray(forces, dtype=np.float64)
        masses = np.ascontiguousarray(masses, dtype=np.float64)
        box_lengths = np.ascontiguousarray(box_lengths, dtype=np.float64)
        stress_tensor_view = np.ascontiguousarray(stress_tensor_view, dtype=np.float64)

        # 调用C++函数
        self.lib.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            box_lengths,
            stress_tensor_view,
        )

        # 若stress_tensor为(3,3)，数据已经更新，因为view共享内存
        stress_tensor[:] = stress_tensor_view.reshape(stress_tensor.shape)

    def calculate_lj_forces(
        self,
        num_atoms,
        positions,
        forces,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
        neighbor_pairs,
        num_pairs,
    ):
        """
        调用 C++ 接口计算作用力。

        Parameters
        ----------
        num_atoms : int
            原子数。
        positions : numpy.ndarray
            原子位置数组，形状为 (num_atoms, 3)。
        forces : numpy.ndarray
            力数组，形状为 (num_atoms, 3)，将被更新。
        epsilon : float
            Lennard-Jones 势参数 ε，单位 eV。
        sigma : float
            Lennard-Jones 势参数 σ，单位 Å。
        cutoff : float
            截断距离，单位 Å。
        box_lengths : numpy.ndarray
            盒子长度数组，形状为 (3,)。
        neighbor_pairs : numpy.ndarray
            邻居对数组，形状为 (2*num_pairs,)。
        num_pairs : int
            邻居对的数量。

        Returns
        -------
        None
        """
        self.lib.calculate_lj_forces(
            num_atoms,
            positions,
            forces,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
            neighbor_pairs,
            num_pairs,
        )

    def calculate_lj_energy(
        self,
        num_atoms,
        positions,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
        neighbor_pairs,
        num_pairs,
    ):
        """
        调用 C++ 接口计算能量。

        Parameters
        ----------
        num_atoms : int
            原子数。
        positions : numpy.ndarray
            原子位置数组，形状为 (num_atoms, 3)。
        epsilon : float
            Lennard-Jones 势参数 ε，单位 eV。
        sigma : float
            Lennard-Jones 势参数 σ，单位 Å。
        cutoff : float
            截断距离，单位 Å。
        box_lengths : numpy.ndarray
            盒子长度数组，形状为 (3,)。
        neighbor_pairs : numpy.ndarray
            邻居对数组，形状为 (2*num_pairs,)。
        num_pairs : int
            邻居对的数量。

        Returns
        -------
        float
            总势能，单位 eV。
        """
        energy = self.lib.calculate_lj_energy(
            num_atoms,
            positions,
            epsilon,
            sigma,
            cutoff,
            box_lengths,
            neighbor_pairs,
            num_pairs,
        )
        return energy

    def calculate_eam_al1_forces(
        self,
        num_atoms: int,
        positions: np.ndarray,
        box_lengths: np.ndarray,
        forces: np.ndarray,
    ) -> None:
        """
        计算EAM Al1势的原子力。

        Parameters
        ----------
        num_atoms : int
            原子数量
        positions : numpy.ndarray
            原子位置数组，形状为(num_atoms, 3)
        box_lengths : numpy.ndarray
            模拟盒子的长度，形状为(3,)
        forces : numpy.ndarray
            输出的力数组，形状为(num_atoms, 3)，将被更新

        Returns
        -------
        None
        """
        self.lib.calculate_eam_al1_forces(
            num_atoms,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(box_lengths, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
        )

    def calculate_eam_al1_energy(
        self, num_atoms: int, positions: np.ndarray, box_lengths: np.ndarray
    ) -> float:
        """
        计算EAM Al1势的总能量。

        Parameters
        ----------
        num_atoms : int
            原子数量
        positions : numpy.ndarray
            原子位置数组，形状为(num_atoms, 3)
        box_lengths : numpy.ndarray
            模拟盒子的长度，形状为(3,)

        Returns
        -------
        float
            系统的总能量，单位为eV
        """
        energy = np.zeros(1, dtype=np.float64)
        self.lib.calculate_eam_al1_energy(
            num_atoms,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(box_lengths, dtype=np.float64),
            energy,
        )
        return energy[0]

    def nose_hoover(
        self, dt, num_atoms, masses, velocities, forces, xi_array, Q, target_temperature
    ):
        """
        实现 Nose-Hoover 恒温器算法。!!!没有去除质心运动

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
            xi_array,
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

    def parrinello_rahman_hoover(
        self,
        dt: float,
        num_atoms: int,
        masses: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        lattice_vectors: np.ndarray,
        xi: np.ndarray,
        Q: np.ndarray,
        total_stress: np.ndarray,  # 9 components
        target_pressure: np.ndarray,  # 9 components
        W: float,
    ):
        # 检查输入数组的形状
        if masses.shape != (num_atoms,):
            raise ValueError(
                f"Expected masses shape {(num_atoms,)}, got {masses.shape}"
            )
        if velocities.shape != (num_atoms * 3,):
            raise ValueError(
                f"Expected velocities shape {(num_atoms * 3,)}, got {velocities.shape}"
            )
        if forces.shape != (num_atoms * 3,):
            raise ValueError(
                f"Expected forces shape {(num_atoms * 3,)}, got {forces.shape}"
            )
        if lattice_vectors.shape != (9,):
            raise ValueError(
                f"Expected lattice_vectors shape {(9,)}, got {lattice_vectors.shape}"
            )
        if xi.shape != (9,):
            raise ValueError(f"Expected xi shape {(9,)}, got {xi.shape}")
        if Q.shape != (9,):
            raise ValueError(f"Expected Q shape {(9,)}, got {Q.shape}")
        if total_stress.shape != (9,):
            raise ValueError(
                f"Expected total_stress shape {(9,)}, got {total_stress.shape}"
            )
        if target_pressure.shape != (9,):
            raise ValueError(
                f"Expected target_pressure shape {(9,)}, got {target_pressure.shape}"
            )

        # 确保所有数组是连续的并且是浮点类型
        masses = np.ascontiguousarray(masses, dtype=np.float64)
        velocities = np.ascontiguousarray(velocities, dtype=np.float64)
        forces = np.ascontiguousarray(forces, dtype=np.float64)
        lattice_vectors = np.ascontiguousarray(lattice_vectors, dtype=np.float64)
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        total_stress = np.ascontiguousarray(total_stress, dtype=np.float64)
        target_pressure = np.ascontiguousarray(target_pressure, dtype=np.float64)

        try:
            self.lib.parrinello_rahman_hoover(
                dt,
                num_atoms,
                masses,
                velocities,
                forces,
                lattice_vectors,
                xi,
                Q,
                total_stress,
                target_pressure,
                W,
            )
        except Exception as e:
            raise RuntimeError(f"Error in C++ parrinello_rahman_hoover: {e}")
