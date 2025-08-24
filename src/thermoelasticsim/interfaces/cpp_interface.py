# 文件名: cpp_interface.py
# 作者: Gilbert Young
# 修改日期: 2025-08-24
# 文件描述: 用于在 Python 中调用 C++ 实现的接口类。

"""
接口模块。

该模块定义了 `CppInterface` 类，用于通过 pybind11 调用外部 C++ 函数库。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    # 导入 pybind11 扩展模块
    import thermoelasticsim._cpp_core as _cpp_core  # type: ignore
except ImportError:
    logger.error(
        "Failed to import pybind11 module _cpp_core. Please build the C++ extensions."
    )
    raise ImportError(
        "pybind11 module _cpp_core is not available. Run 'uv pip install -e .' to build."
    ) from None


class CppInterface:
    """用于调用 C++ 实现的函数的接口类

    @class CppInterface
    @brief 用于调用 C++ 实现的函数的接口类

    Parameters
    ----------
    lib_name : str
        库的名称，用于确定可用的函数集合。
    """

    def __init__(self, lib_name):
        self._lib_name = lib_name
        self._cpp = _cpp_core

        # 检查当前lib_name在pybind11中是否有对应函数
        if lib_name == "lennard_jones":
            if not (
                hasattr(_cpp_core, "calculate_lj_energy")
                and hasattr(_cpp_core, "calculate_lj_forces")
            ):
                raise RuntimeError(
                    "Lennard-Jones functions not available in pybind11 module"
                )
            logger.debug("Using pybind11 backend for Lennard-Jones")
        elif lib_name == "eam_al1":
            if not (
                hasattr(_cpp_core, "calculate_eam_al1_energy")
                and hasattr(_cpp_core, "calculate_eam_al1_forces")
            ):
                raise RuntimeError("EAM Al1 functions not available in pybind11 module")
            logger.debug("Using pybind11 backend for EAM Al1")
        elif lib_name == "eam_cu1":
            if not (
                hasattr(_cpp_core, "calculate_eam_cu1_energy")
                and hasattr(_cpp_core, "calculate_eam_cu1_forces")
            ):
                raise RuntimeError("EAM Cu1 functions not available in pybind11 module")
            logger.debug("Using pybind11 backend for EAM Cu1")
        elif lib_name == "stress_calculator":
            if not hasattr(_cpp_core, "compute_stress"):
                raise RuntimeError(
                    "Stress calculator functions not available in pybind11 module"
                )
            logger.debug("Using pybind11 backend for Stress Calculator")
        elif lib_name == "nose_hoover":
            if not hasattr(_cpp_core, "nose_hoover"):
                raise RuntimeError(
                    "Nose-Hoover functions not available in pybind11 module"
                )
            logger.debug("Using pybind11 backend for Nose-Hoover")
        elif lib_name == "nose_hoover_chain":
            if not hasattr(_cpp_core, "nose_hoover_chain"):
                raise RuntimeError(
                    "Nose-Hoover Chain functions not available in pybind11 module"
                )
            logger.debug("Using pybind11 backend for Nose-Hoover Chain")
        elif lib_name == "parrinello_rahman_hoover":
            if not hasattr(_cpp_core, "parrinello_rahman_hoover"):
                raise RuntimeError(
                    "Parrinello-Rahman-Hoover functions not available in pybind11 module"
                )
            logger.debug("Using pybind11 backend for Parrinello-Rahman-Hoover")
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

        # 使用 pybind11 路径
        self._cpp.compute_stress(
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
        # 直接调用 pybind11 模块
        self._cpp.calculate_lj_forces(
            num_atoms,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
            float(epsilon),
            float(sigma),
            float(cutoff),
            np.ascontiguousarray(box_lengths, dtype=np.float64),
            np.ascontiguousarray(neighbor_pairs, dtype=np.int32),
            int(num_pairs),
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
        return float(
            self._cpp.calculate_lj_energy(
                num_atoms,
                np.ascontiguousarray(positions, dtype=np.float64),
                float(epsilon),
                float(sigma),
                float(cutoff),
                np.ascontiguousarray(box_lengths, dtype=np.float64),
                np.ascontiguousarray(neighbor_pairs, dtype=np.int32),
                int(num_pairs),
            )
        )

    def calculate_eam_al1_forces(
        self,
        num_atoms: int,
        positions: np.ndarray,
        lattice_vectors: np.ndarray,
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
        lattice_vectors : numpy.ndarray
            晶格向量数组，形状为(3, 3)或(9,)
        forces : numpy.ndarray
            输出的力数组，形状为(num_atoms, 3)，将被更新

        Returns
        -------
        None
        """
        # pybind11路径
        self._cpp.calculate_eam_al1_forces(
            num_atoms,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(lattice_vectors, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
        )

    def calculate_eam_al1_energy(
        self, num_atoms: int, positions: np.ndarray, lattice_vectors: np.ndarray
    ) -> float:
        """
        计算EAM Al1势的总能量。

        Parameters
        ----------
        num_atoms : int
            原子数量
        positions : numpy.ndarray
            原子位置数组，形状为(num_atoms, 3)
        lattice_vectors : numpy.ndarray
            晶格向量数组，形状为(3, 3)或(9,)

        Returns
        -------
        float
            系统的总能量，单位为eV
        """
        # pybind11路径
        return float(
            self._cpp.calculate_eam_al1_energy(
                num_atoms,
                np.ascontiguousarray(positions, dtype=np.float64),
                np.ascontiguousarray(lattice_vectors, dtype=np.float64),
            )
        )

    def calculate_eam_al1_virial(
        self, num_atoms: int, positions: np.ndarray, lattice_vectors: np.ndarray
    ) -> np.ndarray:
        """计算EAM Al1的维里张量（未除以体积）。返回形状(3,3)。"""
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if pos.ndim == 2 and pos.shape[1] == 3:
            pos = pos.reshape(-1)
        lat = np.ascontiguousarray(lattice_vectors, dtype=np.float64)
        if lat.shape != (9,):
            lat = lat.reshape(9)

        if hasattr(self._cpp, "calculate_eam_al1_virial"):
            vir = self._cpp.calculate_eam_al1_virial(num_atoms, pos, lat)
            vir = np.ascontiguousarray(vir, dtype=np.float64)
        else:
            raise RuntimeError("C++ backend for EAM Al1 virial not available")
        return vir.reshape(3, 3)

    def calculate_eam_cu1_forces(
        self,
        num_atoms: int,
        positions: np.ndarray,
        lattice_vectors: np.ndarray,
        forces: np.ndarray,
    ) -> None:
        """
        计算EAM Cu1势的原子力。

        Parameters
        ----------
        num_atoms : int
            原子数量
        positions : numpy.ndarray
            原子位置数组，形状为(num_atoms, 3)
        lattice_vectors : numpy.ndarray
            晶格向量数组，形状为(3, 3)或(9,)
        forces : numpy.ndarray
            输出的力数组，形状为(num_atoms, 3)，将被更新

        Returns
        -------
        None
        """
        # pybind11路径
        self._cpp.calculate_eam_cu1_forces(
            num_atoms,
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(lattice_vectors, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
        )

    def calculate_eam_cu1_energy(
        self, num_atoms: int, positions: np.ndarray, lattice_vectors: np.ndarray
    ) -> float:
        """
        计算EAM Cu1势的总能量。

        Parameters
        ----------
        num_atoms : int
            原子数量
        positions : numpy.ndarray
            原子位置数组，形状为(num_atoms, 3)
        lattice_vectors : numpy.ndarray
            晶格向量数组，形状为(3, 3)或(9,)

        Returns
        -------
        float
            系统的总能量，单位为eV
        """
        # pybind11路径
        return float(
            self._cpp.calculate_eam_cu1_energy(
                num_atoms,
                np.ascontiguousarray(positions, dtype=np.float64),
                np.ascontiguousarray(lattice_vectors, dtype=np.float64),
            )
        )

    def calculate_eam_cu1_virial(
        self, num_atoms: int, positions: np.ndarray, lattice_vectors: np.ndarray
    ) -> np.ndarray:
        """计算EAM Cu1的维里张量（未除以体积）。返回形状(3,3)。"""
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if pos.ndim == 2 and pos.shape[1] == 3:
            pos = pos.reshape(-1)
        lat = np.ascontiguousarray(lattice_vectors, dtype=np.float64)
        if lat.shape != (9,):
            lat = lat.reshape(9)

        if hasattr(self._cpp, "calculate_eam_cu1_virial"):
            vir = self._cpp.calculate_eam_cu1_virial(num_atoms, pos, lat)
            vir = np.ascontiguousarray(vir, dtype=np.float64)
        else:
            raise RuntimeError("C++ backend for EAM Cu1 virial not available")
        return vir.reshape(3, 3)

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

        # pybind11路径
        self._cpp.nose_hoover(
            float(dt),
            int(num_atoms),
            np.ascontiguousarray(masses, dtype=np.float64),
            np.ascontiguousarray(velocities, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
            np.ascontiguousarray(xi_array, dtype=np.float64),
            float(Q),
            float(target_temperature),
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
        # pybind11路径
        self._cpp.nose_hoover_chain(
            float(dt),
            int(num_atoms),
            np.ascontiguousarray(masses, dtype=np.float64),
            np.ascontiguousarray(velocities, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
            np.ascontiguousarray(xi_chain, dtype=np.float64),
            np.ascontiguousarray(Q, dtype=np.float64),
            int(chain_length),
            float(target_temperature),
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
        """执行Parrinello-Rahman-Hoover恒压器积分步骤"""
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
            # pybind11路径
            self._cpp.parrinello_rahman_hoover(
                float(dt),
                int(num_atoms),
                masses,
                velocities,
                forces,
                lattice_vectors,
                xi,
                Q,
                total_stress,
                target_pressure,
                float(W),
            )
        except Exception as e:
            raise RuntimeError(f"Error in C++ parrinello_rahman_hoover: {e}") from e
