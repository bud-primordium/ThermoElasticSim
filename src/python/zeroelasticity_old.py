# 文件名: zeroelasticity.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现用于计算弹性常数的求解器和计算类（零温条件下）。

"""
弹性常数模块（零温）

包含 ZeroKElasticConstantsSolver 和 ZeroKElasticConstantsCalculator 类，
用于通过应力应变数据计算材料的弹性常数
"""

import os
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from .mechanics import StrainCalculator, StressCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer, LBFGSOptimizer
from .utils import TensorConverter, EV_TO_GPA
from .visualization import Visualizer

# 配置日志记录
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
        通过最小二乘法求解弹性常数矩阵，并计算整体拟合优度

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
        tuple
            (elastic_constants, r2_score)
            - elastic_constants: 弹性常数矩阵，形状为 (6, 6)
            - r2_score: 整体拟合的 R^2 值
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
            C = model.coef_.T
            predictions = strains @ C.T
        else:
            logger.debug("Solving elastic constants using least squares.")
            C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
            C = C.T
            predictions = strains @ C

        # 使用sklearn的r2_score计算拟合优度
        r2_value = r2_score(stresses.flatten(), predictions.flatten())
        logger.debug(f"Overall R^2 score for elastic constants fitting: {r2_value:.6f}")

        return C, r2_value

    def perform_individual_fits(self, strains, stresses, save_path):
        """
        对每一对应变和应力分量进行线性拟合，并保存拟合结果。

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据，形状为 (N, 6)
        stresses : numpy.ndarray
            应力数据，形状为 (N, 6)
        save_path : str
            保存拟合结果的文件路径

        Returns
        -------
        None
        """
        logger.info(
            "Performing individual linear fits for each stress-strain component."
        )

        components = ["11", "22", "33", "23", "13", "12"]
        results = []

        with open(save_path, "w") as f:
            # 写入表头
            f.write(
                "Deformation_Type\tDependent_Variable\tIndependent_Variable\tSlope\tIntercept\tR^2\n"
            )
            logger.debug("Written header to the fit results file.")

            for i, strain_comp in enumerate(components):
                for j, stress_comp in enumerate(components):
                    # 应用与绘图相同的筛选逻辑
                    mask = np.abs(strains[:, i]) > 1e-10
                    x = strains[mask, i]
                    y = stresses[mask, j]

                    logger.debug(
                        f"Fitting Stress {stress_comp} (dependent) vs Strain {strain_comp} (independent)."
                    )
                    logger.debug(f"Strain {strain_comp} data (x): {x}")
                    logger.debug(f"Stress {stress_comp} data (y): {y}")

                    if len(x) < 2:
                        logger.warning(
                            f"Not enough data points for fit: Strain {strain_comp} vs Stress {stress_comp}"
                        )
                        slope, intercept, r2 = np.nan, np.nan, np.nan
                    else:
                        try:
                            # 进行线性拟合
                            coeffs = np.polyfit(x, y, 1)
                            slope, intercept = coeffs
                            fit_line = np.poly1d(coeffs)
                            y_pred = fit_line(x)
                            r2 = r2_score(y, y_pred)

                            logger.debug(
                                f"Fitted {strain_comp} vs {stress_comp}: "
                                f"Slope={slope:.6f}, Intercept={intercept:.6f}, R^2={r2:.6f}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error fitting {strain_comp} vs {stress_comp}: {e}"
                            )
                            slope, intercept, r2 = np.nan, np.nan, np.nan

                    # 写入结果，保留6位小数
                    f.write(
                        f"{strain_comp}\t{stress_comp}\t{strain_comp}\t{slope:.6f}\t{intercept:.6f}\t{r2:.6f}\n"
                    )
                    logger.debug(
                        f"Written fit to file: {strain_comp} vs {stress_comp}, "
                        f"Slope={slope:.6f}, Intercept={intercept:.6f}, R^2={r2:.6f}"
                    )

        logger.info(f"Individual fit results saved to {save_path}")


class ZeroKElasticConstantsCalculator:
    """
    计算弹性常数的模拟类（零温）

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
        优化器类型，支持 'GD'（梯度下降）、'BFGS'、'LBFGS'，默认为 'LBFGS'
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
        optimizer_type="LBFGS",
        optimizer_params=None,
        save_path="./output",
    ):
        self.cell = cell
        self.potential = potential
        self.delta = delta  # 指的是形变
        self.num_steps = num_steps  # 每个应变分量的步数
        self.deformer = Deformer(delta, num_steps)
        self.stress_calculator = StressCalculator()
        self.strain_calculator = StrainCalculator()
        self.visualizer = Visualizer()  # 初始化 Visualizer 实例
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.optimizer_type = optimizer_type
        self.optimizer_params = self._get_optimizer_params(
            optimizer_type, optimizer_params
        )

    def _get_optimizer_params(self, optimizer_type, optimizer_params):
        """
        获取优化器参数，允许外部传入并覆盖默认参数。

        Parameters
        ----------
        optimizer_type : str
            优化器类型
        optimizer_params : dict or None
            外部传入的优化器参数

        Returns
        -------
        dict
            优化器参数
        """
        default_params = {
            "GD": {
                "max_steps": 200000,
                "tol": 1e-5,
                "step_size": 1e-4,
                "energy_tol": 1e-6,
            },
            "BFGS": {
                "tol": 1e-8,
                "maxiter": 1000000,
            },
            "LBFGS": {
                "ftol": 1e-8,
                "gtol": 1e-5,
                "maxls": 100,
                "maxiter": 10000,
            },
        }

        if optimizer_type not in default_params:
            raise ValueError(
                "Unsupported optimizer type. Choose 'GD', 'BFGS', or 'LBFGS'."
            )

        params = default_params[optimizer_type].copy()
        if optimizer_params:
            params.update(optimizer_params)

        logger.debug(f"Optimizer parameters for {optimizer_type}: {params}")
        return params

    def _get_optimizer_instance(self):
        """
        根据优化器类型返回相应的优化器实例。

        Returns
        -------
        Optimizer
            优化器实例
        """
        optimizer_mapping = {
            "GD": GradientDescentOptimizer,
            "BFGS": BFGSOptimizer,
            "LBFGS": LBFGSOptimizer,
        }

        optimizer_class = optimizer_mapping.get(self.optimizer_type)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        optimizer = optimizer_class(**self.optimizer_params)
        logger.debug(
            f"Initialized optimizer: {self.optimizer_type} with params: {self.optimizer_params}"
        )
        return optimizer

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
        """
        在施加变形前对结构进行一次优化，并保存优化过程的动画。
        """
        logger.debug("Starting initial structure optimization.")
        optimizer = self._get_optimizer_instance()

        converged, trajectory = optimizer.optimize(self.cell, self.potential)
        logger.debug("Initial structure optimization completed.")

        # 保存优化过程的动画
        animation_filename = os.path.join(self.save_path, "initial_optimization.gif")
        self.visualizer.create_optimization_animation(
            trajectory,
            filename=animation_filename,
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

        # 获取优化器实例
        optimizer = self._get_optimizer_instance()

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

        # 减去初始应力张量
        stress_tensor -= self.initial_stress_tensor
        # 对称化应力张量
        stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

        # 计算应变张量，使用小形变理论
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

    def process_deformation_visualization(self, strain_data, stress_data):
        """
        处理所有变形模式的可视化，为每种变形生成单独的PNG

        Parameters
        ----------
        strain_data : numpy.ndarray
            应变数据，形状为 (N, 6)
        stress_data : numpy.ndarray
            应力数据，形状为 (N, 6)
        """
        logger.info("Processing deformation visualizations.")
        # 创建可视化保存目录
        vis_dir = os.path.join(self.save_path, "deformation_analysis")
        os.makedirs(vis_dir, exist_ok=True)

        # 对每种变形模式分别处理
        num_modes = 6
        for i in range(num_modes):
            # 找出对应于这种变形模式的数据点
            # 通过检查该模式的应变是否为主导应变来筛选
            # 使用一个小阈值来避免数值误差
            mask = np.abs(strain_data[:, i]) > 1e-10
            mode_strains = strain_data[mask]
            mode_stresses = stress_data[mask]

            # 确保找到了有效的数据点
            if len(mode_strains) > 0:
                self.visualizer.plot_deformation_stress_strain(
                    mode_strains, mode_stresses, i, vis_dir, show=False
                )
                logger.debug(f"Plotted deformation stress-strain for mode {i}.")

    def _save_deformation_results(self, result):
        """
        保存单个变形的优化动画和晶胞结构图。

        Parameters
        ----------
        result : tuple
            包含应变、应力、优化轨迹、变形后的晶胞和变形索引的元组

        Returns
        -------
        tuple
            应变和应力的 Voigt 表示
        """
        (
            strain_voigt,
            stress_voigt,
            trajectory,
            deformed_cell,
            deformation_index,
        ) = result

        deformation_dir = os.path.join(
            self.save_path, f"deformation_{deformation_index}"
        )
        os.makedirs(deformation_dir, exist_ok=True)

        # 保存优化轨迹动画
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
        logger.info(f"Saved deformation optimization animation to {animation_filename}")

        # 保存变形后的晶胞结构图
        plot_filename = os.path.join(
            deformation_dir, f"deformed_cell_{deformation_index}.png"
        )
        fig, ax = self.visualizer.plot_cell_structure(deformed_cell, show=False)
        fig.savefig(plot_filename)
        plt.close(fig)
        logger.info(f"Saved deformed cell structure plot to {plot_filename}")

        return strain_voigt, stress_voigt

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
        results = [None] * len(F_list)  # 预分配结果列表，确保顺序

        # 并行计算应力和应变
        with ThreadPoolExecutor() as executor:
            # 提交所有任务并立即获取future对象
            future_to_idx = {
                executor.submit(self.calculate_stress_strain, F, i): i
                for i, F in enumerate(F_list)
            }

            # 按完成顺序处理结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    logger.debug(f"Completed deformation {idx}")
                except Exception as e:
                    logger.error(f"Error in deformation {idx}: {e}")
                    raise RuntimeError(f"Deformation {idx} failed") from e

        # 检查是否所有计算都完成
        if any(r is None for r in results):
            raise RuntimeError("Some deformations failed to complete")

        strains, stresses = [], []

        # 处理结果并保存相关文件
        for result in results:
            strain_voigt, stress_voigt = self._save_deformation_results(result)
            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        strain_data = np.array(strains)
        stress_data = np.array(stresses)

        # 生成每种变形模式的PNG
        self.process_deformation_visualization(strain_data, stress_data)

        # 保存应力应变数据
        self.save_stress_strain_data(strain_data, stress_data)

        logger.info(f"Total strain-stress pairs: {len(strains)}")
        logger.info(f"Strain range: [{strain_data.min():.6e}, {strain_data.max():.6e}]")
        logger.info(f"Stress range: [{stress_data.min():.6e}, {stress_data.max():.6e}]")

        # 计算弹性常数矩阵和R^2
        logger.debug("Solving for elastic constants.")
        solver = ZeroKElasticConstantsSolver()
        C, r2 = solver.solve(strain_data, stress_data, regularization=False, alpha=1e-5)
        C_in_GPa = C * EV_TO_GPA

        # 记录R^2值和弹性常数矩阵
        logger.info(f"Elastic constants matrix (GPa):\n{C_in_GPa}")
        logger.info(f"Overall R^2 score: {r2:.6f}")

        # 保存弹性常数矩阵和R^2到文件
        results_file = os.path.join(self.save_path, "elastic_constants_GPa.txt")
        with open(results_file, "w") as f:
            f.write("Elastic Constants Matrix (GPa):\n")
            np.savetxt(f, C_in_GPa, fmt="%.6e")
            f.write(f"\nOverall R^2 score: {r2:.6f}")
            f.write("\nNote: R^2 closer to 1 indicates better fit quality")
        logger.info(f"Elastic constants and R^2 saved to {results_file}")

        # 创建并保存拟合结果到统一的txt文件
        fit_results_file = os.path.join(self.save_path, "individual_fits.txt")
        solver.perform_individual_fits(strain_data, stress_data, fit_results_file)

        # 保存应力-应变关系图
        fig, ax = self.visualizer.plot_stress_strain_multiple(
            strain_data, stress_data, show=False
        )
        stress_strain_plot = os.path.join(
            self.save_path, "stress_strain_relationship.png"
        )
        fig.savefig(stress_strain_plot)
        plt.close(fig)
        logger.info(f"Saved stress-strain relationship plot to {stress_strain_plot}")

        # 保留应力-应变关系动画的创建
        animation_filename = os.path.join(
            self.save_path, "stress_strain_relationship.gif"
        )
        self.visualizer.create_stress_strain_animation(
            strain_data,
            stress_data,
            animation_filename,
            show=False,
        )
        logger.info(
            f"Saved stress-strain relationship animation to {animation_filename}"
        )

        return C_in_GPa
