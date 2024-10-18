import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from .mechanics import StressCalculatorLJ, StrainCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer
from .utils import TensorConverter, EV_TO_GPA  # 导入单位转换因子

# 配置日志记录
logger = logging.getLogger(__name__)


class ElasticConstantsSolver:
    """
    @class ElasticConstantsSolver
    @brief 计算弹性常数的求解器类。
    """

    def solve(self, strains, stresses):
        """
        @brief 通过最小二乘法求解弹性常数矩阵。

        @param strains 应变数据列表，形状为 (N, 6)
        @param stresses 应力数据列表，形状为 (N, 6)
        @return 弹性常数矩阵，形状为 (6, 6)
        """
        strains = np.array(strains)
        stresses = np.array(stresses)

        # 检查输入数据维度
        if strains.ndim != 2 or stresses.ndim != 2:
            raise ValueError("Strains and stresses must be 2D arrays.")
        if strains.shape[0] != stresses.shape[0]:
            raise ValueError("Number of strain and stress samples must be equal.")
        if strains.shape[1] != 6 or stresses.shape[1] != 6:
            raise ValueError("Strains and stresses must have 6 components each.")

        logger.debug("Solving elastic constants using least squares.")
        C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
        logger.debug(f"Elastic constants matrix (before conversion):\n{C}")
        return C


class ElasticConstantsCalculator:
    """
    @class ElasticConstantsCalculator
    @brief 用于计算弹性常数的类。
    """

    def __init__(self, cell, potential, delta=1e-3, optimizer_type="GD"):
        """
        @param cell 晶胞对象
        @param potential 势能对象
        @param delta 变形大小
        @param optimizer_type 优化器类型，支持 'GD'（梯度下降）和 'BFGS'
        """
        self.cell = cell
        self.potential = potential
        self.delta = delta
        self.deformer = Deformer(delta)
        self.stress_calculator = StressCalculatorLJ()
        self.strain_calculator = StrainCalculator()

        if optimizer_type == "GD":
            self.optimizer = GradientDescentOptimizer(
                max_steps=10000,
                tol=1e-5,
                step_size=1e-3,
                energy_tol=1e-5,  # 根据要求调整步长
            )
        elif optimizer_type == "BFGS":
            self.optimizer = BFGSOptimizer(tol=1e-6, maxiter=10000)  # 增加 maxiter
        else:
            raise ValueError("Unsupported optimizer type. Choose 'GD' or 'BFGS'.")

    def calculate_stress_strain(self, F):
        """
        @brief 对单个应变矩阵施加变形、优化结构，并计算应力和应变。

        @param F 变形矩阵
        @return 应变和应力张量 (Voigt 表示法)
        """
        # 复制初始晶胞
        deformed_cell = self.cell.copy()

        # 施加变形
        deformed_cell.apply_deformation(F)
        logger.debug("Applied deformation to cell.")

        # 检查原子之间的最小距离，确保不会过近
        min_distance = np.inf
        num_atoms = len(deformed_cell.atoms)
        min_pair = (-1, -1)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                rij = deformed_cell.atoms[j].position - deformed_cell.atoms[i].position
                for dim in range(3):
                    rij[dim] -= (
                        round(rij[dim] / deformed_cell.lattice_vectors[dim, dim])
                        * deformed_cell.lattice_vectors[dim, dim]
                    )
                r = np.linalg.norm(rij)
                if r < min_distance:
                    min_distance = r
                    min_pair = (deformed_cell.atoms[i].id, deformed_cell.atoms[j].id)

        assert (
            min_distance > 0.8 * self.potential.sigma
        ), f"原子之间的最小距离过近：{min_distance} Å (atoms {min_pair[0]} and {min_pair[1]})"

        # 优化结构
        self.optimizer.optimize(deformed_cell, self.potential)
        logger.debug("Optimization completed.")

        # 计算应力张量
        stress_tensor = self.stress_calculator.compute_stress(
            deformed_cell, self.potential
        )
        logger.debug(f"Computed stress tensor:\n{stress_tensor}")

        # 计算应变张量
        strain_voigt = self.strain_calculator.compute_strain(F)
        logger.debug(f"Computed strain (Voigt): {strain_voigt}")

        # 转换应力为 Voigt 表示法
        stress_voigt = TensorConverter.to_voigt(stress_tensor)
        logger.debug(f"Converted stress to Voigt notation: {stress_voigt}")

        return strain_voigt, stress_voigt

    def calculate_elastic_constants(self):
        """
        @brief 计算弹性常数矩阵。

        @return 弹性常数矩阵，形状为 (6, 6)，单位为 GPa。
        """
        logger.debug("Starting elastic constants calculation.")

        # 生成六个变形矩阵
        F_list = self.deformer.generate_deformation_matrices()
        strains = []
        stresses = []

        # 使用线程池并行化每个应变的处理
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.calculate_stress_strain, F_list)

        for strain, stress in results:
            strains.append(strain)
            stresses.append(stress)

        # 使用求解器求解弹性常数
        logger.debug("Solving for elastic constants.")
        elastic_solver = ElasticConstantsSolver()
        C = elastic_solver.solve(strains, stresses)
        logger.debug(f"Elastic constants matrix (eV/Å^3 / strain):\n{C}")

        # 单位转换为 GPa
        C_in_GPa = C * EV_TO_GPA  # 使用 utils 中的单位转换因子
        logger.debug(f"Elastic constants matrix (GPa):\n{C_in_GPa}")

        return C_in_GPa
