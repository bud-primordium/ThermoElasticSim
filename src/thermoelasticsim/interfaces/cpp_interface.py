# 文件名: cpp_interface.py
# 作者: Gilbert Young
# 修改日期: 2025-08-12
# 文件描述: 用于在 Python 中调用 C++ 实现的接口类。

"""
接口模块。

该模块定义了 `CppInterface` 类，用于通过 ctypes 调用外部 C++ 函数库。
"""

import ctypes
import logging
import os
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)

try:
    # 优先尝试导入 pybind11 扩展模块
    import thermoelasticsim._cpp_core as _cpp_core  # type: ignore
except Exception:  # pragma: no cover - 环境未构建扩展时忽略
    # 尝试自动构建扩展模块
    import subprocess
    import sys
    from pathlib import Path

    # 查找项目根目录
    module_dir = Path(__file__).parent.parent
    project_root = module_dir.parent.parent

    # 检查是否存在CMakeLists.txt
    if (project_root / "CMakeLists.txt").exists():
        try:
            # 创建构建目录
            build_dir = project_root / "build"
            build_dir.mkdir(exist_ok=True)

            # 尝试自动构建
            logger.info("自动构建pybind11扩展模块...")

            # 获取pybind11路径
            import pybind11

            pybind11_dir = pybind11.get_cmake_dir()

            # 配置CMake
            subprocess.check_call(
                [
                    "cmake",
                    str(project_root),
                    f"-Dpybind11_DIR={pybind11_dir}",
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                ],
                cwd=build_dir,
            )

            # 构建
            subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

            # 查找并复制扩展模块
            import glob

            for so_file in glob.glob(str(build_dir / "*_cpp_core*.so")):
                import shutil

                target = module_dir / Path(so_file).name
                shutil.copy2(so_file, target)
                logger.info(f"已复制扩展模块到 {target}")

            # 再次尝试导入
            import thermoelasticsim._cpp_core as _cpp_core

            logger.info("pybind11扩展模块构建成功")
        except Exception as e:
            logger.debug(f"自动构建失败: {e}")
            _cpp_core = None
    else:
        _cpp_core = None


class CppInterface:
    """用于调用 C++ 实现的函数的接口类

    @class CppInterface
    @brief 用于调用 C++ 实现的函数的接口类

    Parameters
    ----------
    lib_name : str
        库的名称，不包括前缀和扩展名。
    """

    def __init__(self, lib_name):
        self._use_pybind = False
        self._lib_name = lib_name
        self._cpp = None

        # 优先尝试 pybind11 路径（如果可用）
        if _cpp_core is not None:
            self._cpp = _cpp_core
            # 检查当前lib_name在pybind11中是否有对应函数
            if lib_name == "lennard_jones":
                if hasattr(_cpp_core, "calculate_lj_energy") and hasattr(
                    _cpp_core, "calculate_lj_forces"
                ):
                    self._use_pybind = True
                    logger.debug("Using pybind11 backend for Lennard-Jones")
                    return
            elif lib_name == "eam_al1":
                if hasattr(_cpp_core, "calculate_eam_al1_energy") and hasattr(
                    _cpp_core, "calculate_eam_al1_forces"
                ):
                    self._use_pybind = True
                    logger.debug("Using pybind11 backend for EAM Al1")
                    return
            elif lib_name == "stress_calculator":
                if hasattr(_cpp_core, "compute_stress"):
                    self._use_pybind = True
                    logger.debug("Using pybind11 backend for Stress Calculator")
                    return
            elif lib_name == "nose_hoover":
                if hasattr(_cpp_core, "nose_hoover"):
                    self._use_pybind = True
                    logger.debug("Using pybind11 backend for Nose-Hoover")
                    return
            elif lib_name == "nose_hoover_chain":
                if hasattr(_cpp_core, "nose_hoover_chain"):
                    self._use_pybind = True
                    logger.debug("Using pybind11 backend for Nose-Hoover Chain")
                    return
            elif lib_name == "parrinello_rahman_hoover" and hasattr(
                _cpp_core, "parrinello_rahman_hoover"
            ):
                self._use_pybind = True
                logger.debug("Using pybind11 backend for Parrinello-Rahman-Hoover")
                return

        # Fallback到ctypes实现
        logger.debug(f"Using ctypes backend for {lib_name}")
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
        while current_dir != os.path.dirname(current_dir):  # 循环直到文件系统的根目录
            if "pyproject.toml" in os.listdir(current_dir):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)

        if project_root is None:
            raise FileNotFoundError("无法定位项目根目录 (未找到 pyproject.toml)。")

        # 从项目根目录构建库文件的绝对路径
        lib_path = os.path.join(
            project_root,
            "src",
            "thermoelasticsim",
            "lib",
            lib_prefix + lib_name + lib_extension,
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
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # lattice_vectors (9, row-major)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # forces
            ]
            self.lib.calculate_eam_al1_forces.restype = None

            self.lib.calculate_eam_al1_energy.argtypes = [
                ctypes.c_int,  # num_atoms
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # positions
                ndpointer(
                    ctypes.c_double, flags="C_CONTIGUOUS"
                ),  # lattice_vectors (9, row-major)
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # energy (output)
            ]
            self.lib.calculate_eam_al1_energy.restype = None

            # 可选：维里计算（如果提供C接口）
            if hasattr(self.lib, "calculate_eam_al1_virial"):
                self.lib.calculate_eam_al1_virial.argtypes = [
                    ctypes.c_int,
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ]
                self.lib.calculate_eam_al1_virial.restype = None

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

        if self._use_pybind:
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
        else:
            # 使用 ctypes 路径
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
        if self._use_pybind:
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
        else:
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
        if self._use_pybind:
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
        else:
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
        box_lengths : numpy.ndarray
            模拟盒子的长度，形状为(3,)
        forces : numpy.ndarray
            输出的力数组，形状为(num_atoms, 3)，将被更新

        Returns
        -------
        None
        """
        if self._use_pybind:
            # pybind11路径
            self._cpp.calculate_eam_al1_forces(
                num_atoms,
                np.ascontiguousarray(positions, dtype=np.float64),
                np.ascontiguousarray(lattice_vectors, dtype=np.float64),
                np.ascontiguousarray(forces, dtype=np.float64),
            )
        else:
            # ctypes路径
            self.lib.calculate_eam_al1_forces(
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
        box_lengths : numpy.ndarray
            模拟盒子的长度，形状为(3,)

        Returns
        -------
        float
            系统的总能量，单位为eV
        """
        if self._use_pybind:
            # pybind11路径
            return float(
                self._cpp.calculate_eam_al1_energy(
                    num_atoms,
                    np.ascontiguousarray(positions, dtype=np.float64),
                    np.ascontiguousarray(lattice_vectors, dtype=np.float64),
                )
            )
        else:
            # ctypes路径
            energy = np.zeros(1, dtype=np.float64)
            self.lib.calculate_eam_al1_energy(
                num_atoms,
                np.ascontiguousarray(positions, dtype=np.float64),
                np.ascontiguousarray(lattice_vectors, dtype=np.float64),
                energy,
            )
            return energy[0]

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

        if self._use_pybind and hasattr(self._cpp, "calculate_eam_al1_virial"):
            vir = self._cpp.calculate_eam_al1_virial(num_atoms, pos, lat)
            vir = np.ascontiguousarray(vir, dtype=np.float64)
        else:
            if not hasattr(self, "lib") or not hasattr(
                self.lib, "calculate_eam_al1_virial"
            ):
                raise RuntimeError("C++ backend for EAM virial not available")
            vir = np.zeros(9, dtype=np.float64)
            self.lib.calculate_eam_al1_virial(num_atoms, pos, lat, vir)
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

        if self._use_pybind:
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
        else:
            # ctypes路径
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
        if self._use_pybind:
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
        else:
            # ctypes路径
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
            if self._use_pybind:
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
            else:
                # ctypes路径
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
            raise RuntimeError(f"Error in C++ parrinello_rahman_hoover: {e}") from e
