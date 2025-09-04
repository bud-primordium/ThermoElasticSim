# 文件名: mechanics.py
# 作者: Gilbert Young
# 修改日期: 2025-08-12
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
            num_atoms = len(cell.atoms)
            velocities = cell.get_velocities()  # shape: (num_atoms, 3)
            masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
            volume = cell.volume

            # 初始化动能应力张量 (3,3)
            kinetic_stress = np.zeros((3, 3), dtype=np.float64)

            # 动能应力项：-1/V * Σᵢ m_i v_iα v_iβ
            for i in range(num_atoms):
                for α in range(3):
                    for β in range(3):
                        kinetic_stress[α, β] -= (
                            masses[i] * velocities[i, α] * velocities[i, β]
                        )

            kinetic_stress /= volume

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
                # 通用回退：-Σ_i ri ⊗ Fi（无需成对分解，适用于多体势）
                # 先确保力为最新
                with contextlib.suppress(Exception):
                    potential.calculate_forces(cell)
                vir = np.zeros((3, 3), dtype=np.float64)
                pos = cell.get_positions()
                frc = cell.get_forces()
                for i in range(pos.shape[0]):
                    vir -= np.outer(pos[i], frc[i])
                virial_tensor = vir

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
            kinetic_stress = self.calculate_kinetic_stress(cell)
            virial_stress = self.calculate_virial_stress(cell, potential)
            total_stress = kinetic_stress + virial_stress

            logger.debug(f"Total stress magnitude: {np.linalg.norm(total_stress):.2e}")
            return total_stress

        except Exception as e:
            logger.error(f"Error in total stress calculation: {e}")
            raise

    def _calculate_eam_virial_contribution(self, cell, potential) -> np.ndarray:
        r"""计算 EAM 势的维里贡献（多体解析形式）

        仍采用维里框架：

        .. math::
            -\frac{1}{V} \sum_{i<j} r_{ij,\alpha} F_{ij,\beta}

        其中：

        :math:`\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j`
            原子间最小镜像位移向量（适用于三斜晶胞）

        :math:`\mathbf{F}_{ij}`
            EAM 多体势对应的相互作用力（包含嵌入能项的贡献）

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            EAM 势能对象

        Returns
        -------
        np.ndarray
            维里贡献张量 (3, 3)，单位 eV/Å³
        """
        try:
            num_atoms = len(cell.atoms)
            positions = cell.get_positions()
            lattice = cell.lattice_vectors
            LT = lattice.T
            invLT = np.linalg.inv(LT)

            # 初始化维里张量
            virial_tensor = np.zeros((3, 3), dtype=np.float64)

            # 首先计算电子密度
            electron_density = np.zeros(num_atoms, dtype=np.float64)

            # 计算每个原子的电子密度
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i == j:
                        continue

                    # 计算原子间最小镜像距离向量（通用于三斜晶胞）
                    rij_cart = positions[i] - positions[j]
                    s = invLT @ rij_cart
                    s -= np.round(s)
                    rij = LT @ s
                    r = np.linalg.norm(rij)

                    if r <= 6.5:  # EAM Al1的截断半径
                        electron_density[i] += self._psi(r)

            # 计算原子间相互作用力和维里贡献
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):  # j > i 避免重复计算
                    # 计算原子间最小镜像距离向量（通用于三斜晶胞）
                    rij_cart = positions[i] - positions[j]
                    s = invLT @ rij_cart
                    s -= np.round(s)
                    rij = LT @ s
                    r = np.linalg.norm(rij)

                    if r > 1e-6 and r <= 6.5:  # 在截断半径内
                        # 计算EAM力的各个分量
                        d_phi = self._phi_grad(r)
                        d_psi = self._psi_grad(r)
                        d_F_i = self._F_grad(electron_density[i])
                        d_F_j = self._F_grad(electron_density[j])

                        # 计算力的大小
                        force_magnitude = -(d_phi + (d_F_i + d_F_j) * d_psi)

                        # 计算力向量 F_ij（原子j对原子i的力）
                        # 由于r_ij = r_i - r_j，力的方向是沿着-∇_i方向
                        # F_ij = force_magnitude * r_ij/|r_ij|
                        force_ij = force_magnitude * (rij / r)

                        # 计算维里贡献: -1/V * Σ(i<j) r_ij^α * F_ij^β
                        # 由于只计算i<j，已经包含了求和的1/2因子
                        for α in range(3):
                            for β in range(3):
                                virial_tensor[α, β] -= rij[α] * force_ij[β]

            return virial_tensor

        except Exception as e:
            logger.error(f"Error in EAM virial contribution calculation: {e}")
            raise

    def _phi(self, r):
        """EAM Al1对势函数"""
        phi_val = 0.0

        # 定义常数
        a0 = 0.65196946237834
        a1 = 7.6046051582736
        a2 = -5.8187505542843
        a3 = 1.0326940511805

        b = [
            13.695567100510,
            -44.514029786506,
            95.853674731436,
            -83.744769235189,
            29.906639687889,
        ]
        c = [
            -2.3612121457801,
            2.5279092055084,
            -3.3656803584012,
            0.94831589893263,
            -0.20965407907747,
        ]
        d = [
            0.24809459274509,
            -0.54072248340384,
            0.46579408228733,
            -0.18481649031556,
            0.028257788274378,
        ]

        # 区域1: [1.5, 2.3]
        if 1.5 <= r <= 2.3:
            phi_val += np.exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r)

        # 区域2: (2.3, 3.2]
        if 2.3 < r <= 3.2:
            dr = 3.2 - r
            for n in range(5):
                phi_val += b[n] * (dr ** (n + 4))

        # 区域3: (2.3, 4.8]
        if 2.3 < r <= 4.8:
            dr = 4.8 - r
            for n in range(5):
                phi_val += c[n] * (dr ** (n + 4))

        # 区域4: (2.3, 6.5]
        if 2.3 < r <= 6.5:
            dr = 6.5 - r
            for n in range(5):
                phi_val += d[n] * (dr ** (n + 4))

        return phi_val

    def _phi_grad(self, r):
        """EAM Al1对势函数的导数"""
        dphi = 0.0

        # 定义常数
        a0 = 0.65196946237834
        a1 = 7.6046051582736
        a2 = -5.8187505542843
        a3 = 1.0326940511805

        b = [
            13.695567100510,
            -44.514029786506,
            95.853674731436,
            -83.744769235189,
            29.906639687889,
        ]
        c = [
            -2.3612121457801,
            2.5279092055084,
            -3.3656803584012,
            0.94831589893263,
            -0.20965407907747,
        ]
        d = [
            0.24809459274509,
            -0.54072248340384,
            0.46579408228733,
            -0.18481649031556,
            0.028257788274378,
        ]

        if r < 1.5:
            return -1e10

        if 1.5 <= r <= 2.3:
            exp_term = np.exp(a0 + a1 * r + a2 * r * r + a3 * r * r * r)
            dphi += (a1 + 2.0 * a2 * r + 3.0 * a3 * r * r) * exp_term

        if 2.3 < r <= 3.2:
            dr = 3.2 - r
            for n in range(5):
                dphi += -(n + 4) * b[n] * (dr ** (n + 3))

        if 2.3 < r <= 4.8:
            dr = 4.8 - r
            for n in range(5):
                dphi += -(n + 4) * c[n] * (dr ** (n + 3))

        if 2.3 < r <= 6.5:
            dr = 6.5 - r
            for n in range(5):
                dphi += -(n + 4) * d[n] * (dr ** (n + 3))

        return dphi

    def _psi(self, r):
        """EAM Al1电子密度贡献函数"""
        psi_val = 0.0

        c_k = [
            0.00019850823042883,
            0.10046665347629,
            0.10054338881951,
            0.099104582963213,
            0.090086286376778,
            0.0073022698419468,
            0.014583614223199,
            -0.0010327381407070,
            0.0073219994475288,
            0.0095726042919017,
        ]
        r_k = [2.5, 2.6, 2.7, 2.8, 3.0, 3.4, 4.2, 4.8, 5.6, 6.5]

        for i in range(10):
            if r <= r_k[i]:
                psi_val += c_k[i] * ((r_k[i] - r) ** 4)

        return psi_val

    def _psi_grad(self, r):
        """EAM Al1电子密度贡献函数的导数"""
        dpsi = 0.0

        c_k = [
            0.00019850823042883,
            0.10046665347629,
            0.10054338881951,
            0.099104582963213,
            0.090086286376778,
            0.0073022698419468,
            0.014583614223199,
            -0.0010327381407070,
            0.0073219994475288,
            0.0095726042919017,
        ]
        r_k = [2.5, 2.6, 2.7, 2.8, 3.0, 3.4, 4.2, 4.8, 5.6, 6.5]

        for i in range(10):
            if r <= r_k[i]:
                dpsi += -4.0 * c_k[i] * ((r_k[i] - r) ** 3)

        return dpsi

    def _F(self, rho):
        """EAM Al1嵌入能函数"""
        F_val = -np.sqrt(rho)  # 对所有ρ的基本项

        if rho >= 16.0:
            dr = rho - 16.0
            D_n = [
                -6.1596236428225e-5,
                1.4856817073764e-5,
                -1.4585661621587e-6,
                7.2242013524147e-8,
                -1.7925388537626e-9,
                1.7720686711226e-11,
            ]

            for n in range(6):
                F_val += D_n[n] * (dr ** (n + 4))

        return F_val

    def _F_grad(self, rho):
        """EAM Al1嵌入能函数的导数"""
        dF = -0.5 / np.sqrt(rho)  # 对所有ρ的基本项

        if rho >= 16.0:
            dr = rho - 16.0
            D_n = [
                -6.1596236428225e-5,
                1.4856817073764e-5,
                -1.4585661621587e-6,
                7.2242013524147e-8,
                -1.7925388537626e-9,
                1.7720686711226e-11,
            ]

            for n in range(6):
                dF += (n + 4) * D_n[n] * (dr ** (n + 3))

        return dF

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


# 废弃的子类
# class StressCalculatorLJ(StressCalculator):
#     """
#     基于 Lennard-Jones 势的应力计算器

#     Parameters
#     ----------
#     cell : Cell
#         包含原子的晶胞对象
#     potential : Potential
#         Lennard-Jones 势能对象
#     """

#     def __init__(self):
#         self.cpp_interface = CppInterface("stress_calculator")

#     def compute_stress(self, cell, potential):
#         """
#         计算 Lennard-Jones 势的应力张量

#         Parameters
#         ----------
#         cell : Cell
#             包含原子的晶胞对象
#         potential : Potential
#             Lennard-Jones 势能对象

#         Returns
#         -------
#         numpy.ndarray
#             3x3 应力张量矩阵
#         """
#         # 计算并更新原子力
#         potential.calculate_forces(cell)

#         # 获取相关物理量
#         volume = cell.calculate_volume()
#         atoms = cell.atoms
#         num_atoms = len(atoms)
#         positions = np.array(
#             [atom.position for atom in atoms], dtype=np.float64
#         )  # (num_atoms, 3)
#         velocities = np.array(
#             [atom.velocity for atom in atoms], dtype=np.float64
#         )  # (num_atoms, 3)
#         forces = cell.get_forces()  # cell.get_forces() 返回 (num_atoms, 3)
#         masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
#         box_lengths = cell.get_box_lengths()  # (3,)

#         # 初始化应力张量数组
#         stress_tensor = np.zeros((3, 3), dtype=np.float64)

#         # 调用 C++ 接口计算应力张量
#         self.cpp_interface.compute_stress(
#             num_atoms,
#             positions,
#             velocities,
#             forces,
#             masses,
#             volume,
#             box_lengths,
#             stress_tensor,
#         )

#         # 这里因为stress_tensor已经是(3,3)，无需再次reshape
#         return stress_tensor


# class StressCalculatorEAM(StressCalculator):
#     """
#     基于 EAM 势的应力计算器

#     计算EAM势下的应力张量，包括：
#     1. 对势项的贡献
#     2. 电子密度的贡献
#     3. 嵌入能的贡献

#     Parameters
#     ----------
#     None
#     """

#     def __init__(self):
#         """初始化EAM应力计算器"""
#         self.cpp_interface = CppInterface("stress_calculator")

#     def compute_stress(self, cell, potential):
#         """
#         计算 EAM 势的应力张量

#         Parameters
#         ----------
#         cell : Cell
#             包含原子的晶胞对象
#         potential : EAMAl1Potential
#             EAM 势能对象

#         Returns
#         -------
#         numpy.ndarray
#             3x3 应力张量矩阵，单位为 eV/Å³

#         Notes
#         -----
#         EAM势的应力张量计算包括：
#         1. 对势部分的应力贡献
#         2. 由电子密度梯度产生的应力贡献
#         3. 嵌入能导致的应力贡献
#         """
#         # 计算并更新原子力
#         potential.calculate_forces(cell)

#         # 获取相关物理量
#         volume = cell.calculate_volume()
#         atoms = cell.atoms
#         num_atoms = len(atoms)
#         positions = np.array([atom.position for atom in atoms], dtype=np.float64)
#         velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64)
#         forces = cell.get_forces()  # 从cell获取更新后的力
#         masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
#         box_lengths = cell.get_box_lengths()

#         # 初始化应力张量数组
#         stress_tensor = np.zeros((3, 3), dtype=np.float64)

#         # 调用C++接口计算EAM应力张量
#         self.cpp_interface.compute_stress(
#             num_atoms,
#             positions,
#             velocities,
#             forces,
#             masses,
#             volume,
#             box_lengths,
#             stress_tensor,
#         )

#         return stress_tensor


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
