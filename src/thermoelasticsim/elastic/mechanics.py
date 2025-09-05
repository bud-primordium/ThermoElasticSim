# 文件名: mechanics.py
# 作者: Gilbert Young
# 修改日期: 2025-09-05
# 文件描述: 实现应力和应变计算器，包括基于 Lennard-Jones 势和EAM势的应力计算器。

import contextlib
import logging

import matplotlib as mpl
import numpy as np

from thermoelasticsim.interfaces.cpp_interface import CppInterface
from thermoelasticsim.utils.utils import TensorConverter

# 设置matplotlib的日志级别为WARNING，屏蔽字体调试信息
mpl.set_loglevel("WARNING")

# 配置我们自己的日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StressCalculator:
    r"""应力张量计算器

    总应力由两部分组成（体积 :math:`V`）：

    动能项（动量通量）
        .. math::
            \sigma^{\text{kin}}_{\alpha\beta}
            = -\,\frac{1}{V} \sum_i m_i\, v_{i\alpha} v_{i\beta}

    维里项（力-位矢项）
        .. math::
            \sigma^{\text{vir}}_{\alpha\beta}
            = -\,\frac{1}{V} \sum_{i<j} r_{ij,\alpha}\, F_{ij,\beta}

    Notes
    -----
    - 有限差分/晶格导数法（如 :math:`-\partial U/\partial \varepsilon` 或
      :math:`-V^{-1}\,\partial U/\partial \mathbf{h}\, \mathbf{h}^T`）是另一种等价
      的应力计算方式，用于数值校验；它并非额外“第三项”，不与维里项相加。
    - 本实现采用 :math:`\sigma = \sigma^{\text{kin}} + \sigma^{\text{vir}}`。
    - 对 EAM 势，使用经特殊处理的多体维里解析形式；如解析不可用，可用有限差分做校验。
    - 正负号及指标约定与项目其它模块保持一致。
    """

    def __init__(self):
        self.cpp_interface = CppInterface("stress_calculator")
        logger.debug("StressCalculator initialized")

    def calculate_kinetic_stress(self, cell) -> np.ndarray:
        r"""计算动能应力张量

        使用的约定：

        .. math::
           \sigma^{\text{kin}}_{\alpha\beta}
           = -\,\frac{1}{V} \sum_i m_i\, v_{i\alpha} v_{i\beta}.

        说明：在静止构型或 :math:`T\to 0` 极限，速度趋近于零，因此该项趋近于 0。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Returns
        -------
        numpy.ndarray
            动能应力张量 (3, 3)，单位 eV/Å³
        """
        try:
            velocities = cell.get_velocities()  # shape: (N, 3)
            masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
            volume = cell.volume

            # 向量化实现：-1/V * (V^T @ (diag(m) @ V))
            mv = velocities * masses[:, None]  # (N,3)
            kinetic_stress = -(velocities.T @ mv) / volume  # (3,3)

            return kinetic_stress

        except Exception as e:
            logger.error(f"Error in kinetic stress calculation: {e}")
            raise

    def calculate_virial_stress(self, cell, potential) -> np.ndarray:
        r"""计算维里应力张量（相互作用力贡献）

        公式：

        .. math::
            \sigma^{\text{vir}}_{\alpha\beta}
            = -\,\frac{1}{V} \sum_{i<j} r_{ij,\alpha}\, F_{ij,\beta}

        其中：

        :math:`\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j`
            从原子 :math:`j` 指向原子 :math:`i` 的位移向量

        :math:`\mathbf{F}_{ij}`
            原子 :math:`j` 作用于 :math:`i` 的力

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        np.ndarray
            维里应力张量 (3, 3)，单位 eV/Å³
        """
        try:
            volume = cell.volume
            # 优先使用C++的EAM维里实现（若可用）
            virial_tensor = None
            try:
                if hasattr(potential, "cpp_interface"):
                    libname = getattr(potential.cpp_interface, "_lib_name", None)
                    num_atoms = len(cell.atoms)
                    positions = np.ascontiguousarray(
                        cell.get_positions(), dtype=np.float64
                    )
                    lattice = np.ascontiguousarray(
                        cell.lattice_vectors, dtype=np.float64
                    )
                    if libname == "eam_al1" and hasattr(
                        potential.cpp_interface, "calculate_eam_al1_virial"
                    ):
                        virial_tensor = (
                            potential.cpp_interface.calculate_eam_al1_virial(
                                num_atoms, positions, lattice.flatten()
                            )
                        )
                    elif libname == "eam_cu1" and hasattr(
                        potential.cpp_interface, "calculate_eam_cu1_virial"
                    ):
                        virial_tensor = (
                            potential.cpp_interface.calculate_eam_cu1_virial(
                                num_atoms, positions, lattice.flatten()
                            )
                        )
                    elif libname == "tersoff_c1988" and hasattr(
                        potential, "_calculate_virial_tensor"
                    ):
                        virial_tensor = potential._calculate_virial_tensor(cell)
            except Exception as e_cpp:
                logger.debug(
                    f"C++ EAM virial not available, fallback to Python: {e_cpp}"
                )

            if virial_tensor is None:
                # 去除 Python 通用回退，强制要求势提供 C++/专用实现
                raise RuntimeError(
                    "Virial tensor calculation is not available for this potential. "
                    "Please build C++ backend or provide a specialized virial implementation."
                )

            virial_stress = virial_tensor / volume
            return virial_stress
        except Exception as e:
            logger.error(f"Error in virial stress calculation: {e}")
            raise

    def calculate_total_stress(self, cell, potential) -> np.ndarray:
        r"""计算总应力张量（动能项 + 维里项）

        .. math::
            \sigma_{\alpha\beta}
            = -\,\frac{1}{V} \left( \sum_i m_i v_{i\alpha} v_{i\beta}
            + \sum_{i<j} r_{ij,\alpha} F_{ij,\beta} \right)

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        np.ndarray
            总应力张量 (3, 3)，单位 eV/Å³
        """
        try:
            # 优先：若势提供专用维里实现（C++/Python），使用“动能(向量化) + 专用维里”以保持物理一致性
            try:
                kinetic_stress = self.calculate_kinetic_stress(cell)
                virial_stress = self.calculate_virial_stress(cell, potential)
                total_stress = kinetic_stress + virial_stress
                logger.debug(
                    f"Total stress magnitude: {np.linalg.norm(total_stress):.2e}"
                )
                return total_stress
            except Exception:
                # 若无专用维里实现，则使用 C++ 一步式总应力（避免 Python 通用回退）
                with contextlib.suppress(Exception):
                    potential.calculate_forces(cell)

                num_atoms = len(cell.atoms)
                positions = cell.get_positions()
                velocities = cell.get_velocities()
                forces = cell.get_forces()
                masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
                volume = cell.volume
                box_lengths = np.linalg.norm(cell.lattice_vectors, axis=1)
                stress_tensor = np.zeros((3, 3), dtype=np.float64)

                self.cpp_interface.compute_stress(
                    num_atoms,
                    positions,
                    velocities,
                    forces,
                    masses,
                    volume,
                    box_lengths,
                    stress_tensor,
                )
                return stress_tensor

        except Exception as e:
            logger.error(f"Error in total stress calculation: {e}")
            raise

    def calculate_finite_difference_stress(
        self, cell, potential, dr=1e-6
    ) -> np.ndarray:
        """已移除：有限差分应力路径不再提供，避免误用。"""
        raise RuntimeError("Finite-difference stress path has been removed.")

    # 已移除能量有限差分路径，避免误用。

    def get_all_stress_components(self, cell, potential) -> dict[str, np.ndarray]:
        """
        计算应力张量的所有分量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        Dict[str, np.ndarray]
            包含应力张量分量的字典，键包括：
            - "kinetic": 动能应力张量
            - "virial": 维里应力张量
            - "total": 总应力张量（kinetic + virial）

        """
        try:
            components = {}

            # 计算各个分量
            kinetic_stress = self.calculate_kinetic_stress(cell)
            virial_stress = self.calculate_virial_stress(cell, potential)
            total_stress = self.calculate_total_stress(cell, potential)

            # 标准键名
            components["kinetic"] = kinetic_stress
            components["virial"] = virial_stress
            components["total"] = total_stress

            return components

        except Exception as e:
            logger.error(f"Error calculating stress tensors: {e}")
            raise

    def compute_stress(self, cell, potential):
        """
        计算应力张量（使用总应力作为主要方法）

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        np.ndarray
            3x3 应力张量矩阵（总应力 = 动能 + 维里）
        """
        # 使用总应力作为主要计算方法
        return self.calculate_total_stress(cell, potential)

    # 向后兼容的别名方法
    def calculate_virial_kinetic_stress(self, cell, potential) -> np.ndarray:
        """向后兼容别名 - 指向总应力方法"""
        return self.calculate_total_stress(cell, potential)

    def calculate_stress_basic(self, cell, potential) -> np.ndarray:
        """向后兼容别名 - 指向总应力方法"""
        return self.calculate_total_stress(cell, potential)

    def validate_tensor_symmetry(
        self, tensor: np.ndarray, tolerance: float = 1e-10
    ) -> bool:
        """
        验证应力张量是否对称

        Parameters
        ----------
        tensor : np.ndarray
            应力张量 (3x3)
        tolerance : float, optional
            对称性的容差, 默认值为1e-10

        Returns
        -------
        bool
            如果对称则为True, 否则为False
        """
        if tensor.shape != (3, 3):
            logger.error(f"Tensor shape is {tensor.shape}, expected (3, 3).")
            return False

        is_symmetric = np.allclose(tensor, tensor.T, atol=tolerance)
        if not is_symmetric:
            logger.warning("Stress tensor is not symmetric.")

        return is_symmetric


class StrainCalculator:
    """
    应变计算器类

    Parameters
    ----------
    F : numpy.ndarray
        3x3 变形矩阵
    """

    def compute_strain(self, F):
        """
        计算应变张量并返回 Voigt 表示法

        Parameters
        ----------
        F : numpy.ndarray
            3x3 变形矩阵

        Returns
        -------
        numpy.ndarray
            应变向量，形状为 (6,)
        """
        strain_tensor = 0.5 * (F + F.T) - np.identity(3)  # 线性应变张量
        # 转换为 Voigt 表示法
        strain_voigt = TensorConverter.to_voigt(strain_tensor)
        # 对剪切分量乘以 2
        strain_voigt[3:] *= 2
        return strain_voigt
