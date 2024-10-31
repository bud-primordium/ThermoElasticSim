# 文件名: zeroelasticity.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现用于计算弹性常数的求解器和计算类（零温条件下）。

"""
弹性常数模块（零温）

包含 ZeroKElasticConstantsSolver 和 ZeroKElasticConstantsCalculator 类，
用于通过应力应变数据计算材料的弹性常数
"""

import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from .mechanics import StressCalculatorLJ, StrainCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer
from .utils import TensorConverter, EV_TO_GPA  # 导入单位转换因子
from .visualization import Visualizer
import os
import matplotlib.pyplot as plt  # 确保导入了 plt
import csv  # 导入 csv 模块
from sklearn.linear_model import Ridge  # 导入 Ridge 模型用于正则化

# 配置日志记录，移除 threadName
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ZeroKElasticConstantsSolver:
    """
    计算弹性常数的求解器类（零温）
    """

    def solve(self, strains, stresses, regularization=False, alpha=1e-5):
        """
        通过最小二乘法求解弹性常数矩阵

        Parameters
        ----------
        strains : array_like
            应变数据，形状为 (N, 6)
        stresses : array_like
            应力数据，形状为 (N, 6)
        regularization : bool, optional
            是否使用正则化，默认为 False
        alpha : float, optional
            正则化参数，默认为 1e-5

        Returns
        -------
        numpy.ndarray
            弹性常数矩阵，形状为 (6, 6)
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

        if regularization:
            logger.debug(
                "Solving elastic constants using Ridge regression with regularization."
            )
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(strains, stresses)
            C = model.coef_.T  # Ridge 回归的 coef_ 属性形状为 (6, 6)
            logger.debug(f"Elastic constants matrix (Ridge):\n{C}")
        else:
            logger.debug("Solving elastic constants using least squares.")
            C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
            logger.debug(f"Elastic constants matrix (before conversion):\n{C}")

        return C.T  # 返回转置后的矩阵


class ZeroKElasticConstantsCalculator:
    """
    计算弹性常数的类（零温）

    Parameters
    ----------
    cell : Cell
        晶胞对象
    potential : Potential
        势能对象
    delta : float, optional
        变形大小，默认为 1e-1
    num_steps : int, optional
        每个应变分量的步数，默认为 10
    optimizer_type : str, optional
        优化器类型，支持 'GD'（梯度下降）和 'BFGS'，默认为 'BFGS'
    optimizer_params : dict, optional
        优化器的参数，默认为 None。若提供，将覆盖默认参数。
    save_path : str, optional
        保存文件的路径，默认为 './output'
    """

    def __init__(
        self,
        cell,
        potential,
        delta=1e-1,
        num_steps=10,
        optimizer_type="BFGS",
        optimizer_params=None,
        save_path="./output",
    ):
        self.cell = cell
        self.potential = potential
        self.delta = delta  # 指的是形变
        self.num_steps = num_steps  # 每个应变分量的步数
        self.deformer = Deformer(delta, num_steps)
        self.stress_calculator = StressCalculatorLJ()
        self.strain_calculator = StrainCalculator()
        self.visualizer = Visualizer()  # 初始化 Visualizer 实例
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # 设置优化器参数，允许外部传入
        if optimizer_type == "GD":
            self.optimizer_params = {
                "max_steps": 200000,
                "tol": 1e-5,
                "step_size": 1e-4,
                "energy_tol": 1e-6,
            }
            if optimizer_params:
                self.optimizer_params.update(optimizer_params)
            self.optimizer_type = "GD"
        elif optimizer_type == "BFGS":
            self.optimizer_params = {"tol": 1e-6, "maxiter": 500000}
            if optimizer_params:
                self.optimizer_params.update(optimizer_params)
            self.optimizer_type = "BFGS"
        else:
            raise ValueError("Unsupported optimizer type. Choose 'GD' or 'BFGS'.")

    def calculate_initial_stress(self):
        """
        计算初始结构的应力，在优化之前验证应力计算是否正确

        Returns
        -------
        numpy.ndarray
            初始应力张量
        """
        logger.debug("Calculating initial stress before optimization.")
        initial_stress = self.stress_calculator.compute_stress(
            self.cell, self.potential
        )
        logger.debug(f"Initial stress tensor before optimization:\n{initial_stress}")

        # 保存初始应力张量
        self.initial_stress_tensor = initial_stress

        return initial_stress

    def optimize_initial_structure(self):
        """在施加变形前对结构进行一次优化，并保存优化过程的动画。"""
        logger.debug("Starting initial structure optimization.")
        if self.optimizer_type == "GD":
            optimizer = GradientDescentOptimizer(**self.optimizer_params)
        else:
            optimizer = BFGSOptimizer(**self.optimizer_params)

        converged, trajectory = optimizer.optimize(self.cell, self.potential)
        logger.debug("Initial structure optimization completed.")

        # 保存优化过程的动画
        animation_filename = os.path.join(self.save_path, "initial_optimization.gif")
        self.visualizer.create_optimization_animation(
            trajectory,
            filename=animation_filename,  # 明确传入文件路径
            title="Initial Structure Optimization",
            pbc=self.cell.pbc_enabled,
            show=False,
        )

        # 日志记录
        if converged:
            logger.info(f"Saved initial optimization animation to {animation_filename}")
        else:
            logger.warning("Initial structure optimization did not converge.")

    def calculate_stress_strain(self, F, deformation_index):
        """
        对指定的变形矩阵计算应力和应变

        Parameters
        ----------
        F : numpy.ndarray
            变形矩阵
        deformation_index : int
            变形矩阵的编号，便于日志追踪

        Returns
        -------
        tuple
            包含应变（Voigt 表示）、应力（Voigt 表示）、优化轨迹、变形后的晶胞对象的元组
        """
        logger.info(f"Processing deformation matrix #{deformation_index}")
        logger.debug(f"Deformation matrix F:\n{F}")

        # 复制初始晶胞
        deformed_cell = self.cell.copy()

        # 施加变形
        deformed_cell.apply_deformation(F)
        logger.debug(f"Applied deformation to cell #{deformation_index}.")

        # 为每个任务创建独立的优化器实例
        if self.optimizer_type == "GD":
            optimizer = GradientDescentOptimizer(**self.optimizer_params)
        else:
            optimizer = BFGSOptimizer(**self.optimizer_params)

        # 对变形后的结构进行优化并生成轨迹
        converged, trajectory = optimizer.optimize(deformed_cell, self.potential)
        logger.debug(f"Optimized deformed structure #{deformation_index}.")

        if not converged:
            logger.warning(
                f"Optimization did not converge for deformation #{deformation_index}"
            )

        # 计算变形后的应力张量
        stress_tensor = self.stress_calculator.compute_stress(
            deformed_cell, self.potential
        )

        # **减去初始应力张量**
        stress_tensor -= self.initial_stress_tensor
        # # 计算应变张量，使用 Green-Lagrange 应变定义
        # strain_tensor = 0.5 * (F.T @ F - np.eye(3))
        # 还使用小形变理论
        strain_tensor = 0.5 * (F + F.T) - np.eye(3)

        # 使用正确的 Voigt 转换
        strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")
        stress_voigt = TensorConverter.to_voigt(stress_tensor, tensor_type="stress")

        # 日志记录
        logger.info(
            f"Computed stress tensor for deformation #{deformation_index}:\n{stress_tensor}"
        )
        logger.info(
            f"Computed strain (Voigt) for deformation #{deformation_index}: {strain_voigt}"
        )
        logger.info(
            f"Converted stress to Voigt notation for deformation #{deformation_index}: {stress_voigt}"
        )

        # 返回需要的数据，稍后在主线程中处理可视化和文件保存
        return strain_voigt, stress_voigt, trajectory, deformed_cell, deformation_index

    def save_stress_strain_data(self, strain_data, stress_data):
        """
        将应力和应变数据合并保存到一个 CSV 文件中

        Parameters
        ----------
        strain_data : numpy.ndarray
            应变数据，形状为 (N, 6)
        stress_data : numpy.ndarray
            应力数据，形状为 (N, 6)
        """
        combined_data = np.hstack((strain_data, stress_data))
        headers = [
            "strain_11",
            "strain_22",
            "strain_33",
            "strain_23",
            "strain_13",
            "strain_12",
            "stress_11",
            "stress_22",
            "stress_33",
            "stress_23",
            "stress_13",
            "stress_12",
        ]
        filename = os.path.join(self.save_path, "stress_strain_data.csv")
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in combined_data:
                # 格式化每个数值，保留 6 位有效数字
                formatted_row = [f"{value:.6e}" for value in row]
                writer.writerow(formatted_row)
        logger.info(f"Stress-strain data saved to {filename}")

    def calculate_elastic_constants(self):
        """
        计算弹性常数矩阵

        Returns
        -------
        numpy.ndarray
            弹性常数矩阵，形状为 (6, 6)，单位为 GPa
        """
        logger.info("Starting elastic constants calculation.")

        # 计算初始应力
        self.calculate_initial_stress()

        # 优化初始结构并保存
        self.optimize_initial_structure()

        # 生成变形矩阵
        F_list = self.deformer.generate_deformation_matrices()
        strains, stresses = [], []

        # 用于存储每个变形的结果
        results = []

        # 并行计算应力和应变
        with ThreadPoolExecutor() as executor:
            futures = []
            for deformation_index, F in enumerate(F_list):
                futures.append(
                    executor.submit(self.calculate_stress_strain, F, deformation_index)
                )

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error occurred during stress-strain calculation: {e}"
                    )

        # 在主线程中处理可视化和文件保存
        for (
            strain_voigt,
            stress_voigt,
            trajectory,
            deformed_cell,
            deformation_index,
        ) in results:
            # 保存优化过程的动画
            deformation_dir = os.path.join(
                self.save_path, f"deformation_{deformation_index}"
            )
            os.makedirs(deformation_dir, exist_ok=True)

            animation_filename = os.path.join(
                deformation_dir, f"deformation_{deformation_index}_optimization.gif"
            )
            self.visualizer.create_optimization_animation(
                trajectory,
                animation_filename,
                title=f"Deformation #{deformation_index} Optimization",
                pbc=deformed_cell.pbc_enabled,
                show=False,
            )
            logger.info(
                f"Saved deformation optimization animation to {animation_filename}"
            )

            # 保存变形后的晶胞结构图
            plot_filename = os.path.join(
                deformation_dir, f"deformed_cell_{deformation_index}.png"
            )
            fig, ax = self.visualizer.plot_cell_structure(deformed_cell, show=False)
            fig.savefig(plot_filename)
            plt.close(fig)  # 关闭图形，释放内存
            logger.info(f"Saved deformed cell structure plot to {plot_filename}")

            # 保存应力和应变数据
            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        # 将应力和应变数据合并保存到一个 CSV 文件中
        strain_data = np.array(strains)
        stress_data = np.array(stresses)
        self.save_stress_strain_data(strain_data, stress_data)

        logger.info(f"Total strain-stress pairs: {len(strains)}")
        logger.info(f"Strain range: [{strain_data.min():.6e}, {strain_data.max():.6e}]")
        logger.info(f"Stress range: [{stress_data.min():.6e}, {stress_data.max():.6e}]")

        # 计算弹性常数矩阵
        logger.debug("Solving for elastic constants.")
        solver = ZeroKElasticConstantsSolver()
        # 可选：使用正则化来提高求解的稳定性
        C = solver.solve(strain_data, stress_data, regularization=False, alpha=1e-5)
        C_in_GPa = C * EV_TO_GPA
        logger.info(f"Elastic constants matrix (GPa):\n{C_in_GPa}")

        # 保存弹性常数矩阵到文件
        np.savetxt(
            os.path.join(self.save_path, "elastic_constants_GPa.txt"),
            C_in_GPa,
            fmt="%.6e",
        )

        # 保存应力-应变关系图
        fig, ax = self.visualizer.plot_stress_strain(
            strain_data, stress_data, show=False
        )
        fig.savefig(os.path.join(self.save_path, "stress_strain_relationship.png"))
        plt.close(fig)  # 关闭图形，释放内存

        # 可选：创建应力-应变关系的动画
        # self.visualizer.create_stress_strain_animation(
        #     strain_data,
        #     stress_data,
        #     os.path.join(self.save_path, "stress_strain_relationship.gif"),
        #     show=False,
        # )

        return C_in_GPa
