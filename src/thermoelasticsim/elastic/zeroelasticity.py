# 文件名: zeroelasticity.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 实现用于计算弹性常数的求解器和管理工作流（零温条件下）。

"""
弹性常数模块（零温）

包含 ElasticConstantsSolver 和 ElasticConstantsWorkflow 类，
用于通过应力应变数据计算材料的弹性常数并管理整个计算流程

Classes:
    ElasticConstantsSolver: 根据应力应变数据拟合弹性常数矩阵的求解器
    ElasticConstantsWorkflow: 管理从初始优化到最终弹性常数拟合的全流程
"""

import os
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from thermoelasticsim.elastic.mechanics import StrainCalculator, StressCalculator
from thermoelasticsim.elastic.deformation import Deformer
from thermoelasticsim.utils.optimizers import GradientDescentOptimizer, BFGSOptimizer, LBFGSOptimizer
from thermoelasticsim.utils.utils import TensorConverter, EV_TO_GPA
from thermoelasticsim.utils.visualization import Visualizer
from datetime import datetime

# 获取当前时间字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ElasticConstantsSolver:
    """根据应力应变数据拟合弹性常数矩阵的求解器（零温）

    Methods
    -------
    solve(strains, stresses, regularization=False, alpha=1e-5)
        通过最小二乘或Ridge回归求解弹性常数矩阵
    perform_individual_fits(strains, stresses, save_path)
        对每对应变和应力分量进行线性拟合并保存结果
    """

    def solve(
        self,
        strains: np.ndarray,
        stresses: np.ndarray,
        regularization: bool = False,
        alpha: float = 1e-5,
    ) -> tuple[np.ndarray, float]:
        """通过最小二乘或Ridge回归求解弹性常数矩阵，并计算整体拟合优度

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据，形状为 (N, 6)
        stresses : numpy.ndarray
            应力数据，形状为 (N, 6)
        regularization : bool, optional
            是否使用正则化，默认为 False
        alpha : float, optional
            正则化参数，默认为 1e-5

        Returns
        -------
        tuple
            (C, r2_score)
            C为6x6弹性常数矩阵，r2_score为全局拟合的R²值

        Raises
        ------
        ValueError
            如果输入数组形状不匹配或不是2D数组

        Notes
        -----
        使用numpy.linalg.lstsq进行最小二乘拟合
        使用sklearn.linear_model.Ridge进行正则化拟合
        """
        strains = np.array(strains)
        stresses = np.array(stresses)

        # 基本检查
        if strains.ndim != 2 or stresses.ndim != 2:
            raise ValueError("Strains and stresses must be 2D arrays.")
        if strains.shape[0] != stresses.shape[0]:
            raise ValueError("Number of strain and stress samples must be equal.")
        if strains.shape[1] != 6 or stresses.shape[1] != 6:
            raise ValueError("Strains and stresses must have 6 components each.")

        if regularization:
            logger.debug("Solving elastic constants using Ridge regression.")
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(strains, stresses)
            C = model.coef_.T
            predictions = strains @ C.T
        else:
            logger.debug("Solving elastic constants using least squares.")
            C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
            C = C.T
            predictions = strains @ C

        r2_value = r2_score(stresses.flatten(), predictions.flatten())
        logger.debug(f"Overall R^2 score for elastic constants fitting: {r2_value:.6f}")
        return C, r2_value

    def perform_individual_fits(
        self, strains: np.ndarray, stresses: np.ndarray, save_path: str
    ) -> None:
        """对每一对应变和应力分量进行线性拟合，并保存拟合结果

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

        Notes
        -----
        结果文件包含以下列：
        Deformation_Type, Dependent_Variable, Independent_Variable,
        Slope, Intercept, R^2
        """
        logger.info(
            "Performing individual linear fits for each stress-strain component."
        )
        components = ["11", "22", "33", "23", "13", "12"]

        with open(save_path, "w") as f:
            # 写入表头
            f.write(
                "Deformation_Type\tDependent_Variable\tIndependent_Variable\tSlope\tIntercept\tR^2\n"
            )
            logger.debug("Written header to the fit results file.")

            for i, strain_comp in enumerate(components):
                for j, stress_comp in enumerate(components):
                    mask = np.abs(strains[:, i]) > 1e-10
                    x = strains[mask, i]
                    y = stresses[mask, j]

                    logger.debug(
                        f"Fitting Stress {stress_comp} vs Strain {strain_comp}."
                    )
                    logger.debug(f"x (strain): {x}")
                    logger.debug(f"y (stress): {y}")

                    if len(x) < 2:
                        logger.warning(
                            f"Not enough data points for fit: {strain_comp} vs {stress_comp}"
                        )
                        slope, intercept, r2 = np.nan, np.nan, np.nan
                    else:
                        try:
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

                    f.write(
                        f"{strain_comp}\t{stress_comp}\t{strain_comp}\t{slope:.6f}\t{intercept:.6f}\t{r2:.6f}\n"
                    )
                    logger.debug(
                        f"Written fit result to file: {strain_comp} vs {stress_comp}"
                    )


class ElasticConstantsWorkflow:
    """管理从初始优化到最终弹性常数拟合的全流程类

    Attributes
    ----------
    cell : Cell
        晶胞对象
    potential : Potential
        势能对象
    delta : float
        变形大小
    num_steps : int
        每个应变分量的步数
    deformer : Deformer
        变形矩阵生成器
    stress_calculator : StressCalculator
        应力计算器
    strain_calculator : StrainCalculator
        应变计算器
    visualizer : Visualizer
        可视化工具
    save_path : str
        结果保存路径
    optimizer_type : str
        优化器类型
    optimizer_params : dict
        优化器参数
    initial_stress_tensor : numpy.ndarray
        初始应力张量

    Methods
    -------
    run()
        执行完整工作流程
    calculate_initial_stress()
        计算初始应力
    perform_initial_optimization()
        执行初始结构优化
    apply_and_optimize_deformations()
        应用变形并优化结构
    save_stress_strain_data(strains, stresses)
        保存应力应变数据
    process_deformation_visualization(strains, stresses)
        处理变形可视化
    fit_elastic_constants_and_save(strains, stresses)
        拟合弹性常数并保存结果
    """

    def __init__(
        self,
        cell,
        potential,
        delta=1e-1,
        num_steps=10,
        optimizer_type="LBFGS",
        optimizer_params=None,
        save_path=f"../output_new/test_fcc_{current_time}",
    ):
        self.cell = cell
        self.potential = potential
        self.delta = delta
        self.num_steps = num_steps
        self.deformer = Deformer(delta, num_steps)
        self.stress_calculator = StressCalculator()
        self.strain_calculator = StrainCalculator()
        self.visualizer = Visualizer()
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.optimizer_type = optimizer_type
        self.optimizer_params = self._get_optimizer_params(
            optimizer_type, optimizer_params
        )

        self.initial_stress_tensor = None

    def run(self):
        """
        主流程：执行所有步骤并返回最终的C矩阵和R²值。
        """
        self.calculate_initial_stress()
        self.perform_initial_optimization()
        strain_data, stress_data = self.apply_and_optimize_deformations()

        self.save_stress_strain_data(strain_data, stress_data)
        self.process_deformation_visualization(strain_data, stress_data)
        C_in_GPa, r2 = self.fit_elastic_constants_and_save(strain_data, stress_data)
        return C_in_GPa, r2

    def _get_optimizer_params(self, optimizer_type, optimizer_params):
        default_params = {
            "GD": {
                "max_steps": 200000,
                "tol": 1e-5,
                "step_size": 1e-4,
                "energy_tol": 1e-6,
            },
            "BFGS": {"tol": 1e-8, "maxiter": 1000000},
            "LBFGS": {"ftol": 1e-8, "gtol": 1e-5, "maxls": 100, "maxiter": 10000},
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
        logger.debug("Calculating initial stress before optimization.")
        initial_stress = self.stress_calculator.compute_stress(
            self.cell, self.potential
        )
        # 对称化
        initial_stress = 0.5 * (initial_stress + initial_stress.T)
        logger.debug(f"Initial stress tensor:\n{initial_stress}")
        self.initial_stress_tensor = initial_stress

    def perform_initial_optimization(self):
        logger.debug("Starting initial structure optimization.")
        optimizer = self._get_optimizer_instance()
        converged, trajectory = optimizer.optimize(self.cell, self.potential)
        logger.debug("Initial structure optimization completed.")

        animation_filename = os.path.join(self.save_path, "initial_optimization.gif")
        self.visualizer.create_optimization_animation(
            trajectory,
            filename=animation_filename,
            title="Initial Structure Optimization",
            pbc=self.cell.pbc_enabled,
            show=False,
        )

        if converged:
            logger.info(f"Saved initial optimization animation to {animation_filename}")
        else:
            logger.warning("Initial structure optimization did not converge.")

    def apply_and_optimize_deformations(self):
        F_list = self.deformer.generate_deformation_matrices()
        results = self._compute_stress_strain_for_all(F_list)

        strains, stresses = [], []
        for result in results:
            strain_voigt, stress_voigt = self._save_deformation_results(result)
            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        strain_data = np.array(strains)
        stress_data = np.array(stresses)
        return strain_data, stress_data

    def _compute_stress_strain_for_all(self, F_list):
        results = [None] * len(F_list)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(self.calculate_stress_strain, F, i): i
                for i, F in enumerate(F_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    logger.debug(f"Completed deformation {idx}")
                except Exception as e:
                    logger.error(f"Error in deformation {idx}: {e}")
                    raise RuntimeError(f"Deformation {idx} failed") from e

        if any(r is None for r in results):
            raise RuntimeError("Some deformations failed to complete")

        return results

    def calculate_stress_strain(self, F, deformation_index):
        logger.info(f"Processing deformation matrix #{deformation_index}")
        logger.debug(f"Deformation matrix F:\n{F}")

        deformed_cell = self.cell.copy()
        deformed_cell.apply_deformation(F)
        logger.debug(f"Applied deformation to cell #{deformation_index}.")

        optimizer = self._get_optimizer_instance()
        converged, trajectory = optimizer.optimize(deformed_cell, self.potential)
        logger.debug(f"Optimized deformed structure #{deformation_index}.")

        if not converged:
            logger.warning(
                f"Optimization did not converge for deformation #{deformation_index}"
            )

        stress_tensor = self.stress_calculator.compute_stress(
            deformed_cell, self.potential
        )

        # 对称化应力张量，消除数值微小非对称性
        stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)
        stress_tensor -= self.initial_stress_tensor

        strain_tensor = 0.5 * (F + F.T) - np.eye(3)
        strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")
        stress_voigt = TensorConverter.to_voigt(stress_tensor, tensor_type="stress")

        logger.info(
            f"Computed stress tensor for deformation #{deformation_index}:\n{stress_tensor}"
        )
        logger.info(
            f"Computed strain (Voigt) for deformation #{deformation_index}: {strain_voigt}"
        )
        logger.info(
            f"Converted stress to Voigt for deformation #{deformation_index}: {stress_voigt}"
        )

        return strain_voigt, stress_voigt, trajectory, deformed_cell, deformation_index

    def _save_deformation_results(self, result):
        (strain_voigt, stress_voigt, trajectory, deformed_cell, deformation_index) = (
            result
        )

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

    def save_stress_strain_data(self, strain_data, stress_data):
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
                formatted_row = [f"{value:.6e}" for value in row]
                writer.writerow(formatted_row)
        logger.info(f"Stress-strain data saved to {filename}")

    def process_deformation_visualization(self, strain_data, stress_data):
        logger.info("Processing deformation visualizations.")
        vis_dir = os.path.join(self.save_path, "deformation_analysis")
        os.makedirs(vis_dir, exist_ok=True)

        num_modes = 6
        for i in range(num_modes):
            mask = np.abs(strain_data[:, i]) > 1e-10
            mode_strains = strain_data[mask]
            mode_stresses = stress_data[mask]

            if len(mode_strains) > 0:
                self.visualizer.plot_deformation_stress_strain(
                    mode_strains, mode_stresses, i, vis_dir, show=False
                )
                logger.debug(f"Plotted deformation stress-strain for mode {i}.")

    def fit_elastic_constants_and_save(self, strain_data, stress_data):
        logger.debug("Solving for elastic constants.")
        solver = ElasticConstantsSolver()
        C, r2 = solver.solve(strain_data, stress_data, regularization=False, alpha=1e-5)
        C_in_GPa = C * EV_TO_GPA

        logger.info(f"Elastic constants matrix (GPa):\n{C_in_GPa}")
        logger.info(f"Overall R^2 score: {r2:.6f}")

        # 保存弹性常数矩阵和R²
        results_file = os.path.join(self.save_path, "elastic_constants_GPa.txt")
        with open(results_file, "w") as f:
            f.write("Elastic Constants Matrix (GPa):\n")
            np.savetxt(f, C_in_GPa, fmt="%.6e")
            f.write(f"\nOverall R^2 score: {r2:.6f}")
            f.write("\nNote: R^2 closer to 1 indicates better fit quality")
        logger.info(f"Elastic constants and R^2 saved to {results_file}")

        # 个别线性拟合结果
        fit_results_file = os.path.join(self.save_path, "individual_fits.txt")
        solver.perform_individual_fits(strain_data, stress_data, fit_results_file)

        # 绘制应力-应变关系图
        fig, ax = self.visualizer.plot_stress_strain_multiple(
            strain_data, stress_data, show=False
        )
        stress_strain_plot = os.path.join(
            self.save_path, "stress_strain_relationship.png"
        )
        fig.savefig(stress_strain_plot)
        plt.close(fig)
        logger.info(f"Saved stress-strain relationship plot to {stress_strain_plot}")

        # 创建应力-应变关系动画
        animation_filename = os.path.join(
            self.save_path, "stress_strain_relationship.gif"
        )
        self.visualizer.create_stress_strain_animation(
            strain_data, stress_data, animation_filename, show=False
        )
        logger.info(
            f"Saved stress-strain relationship animation to {animation_filename}"
        )

        return C_in_GPa, r2
