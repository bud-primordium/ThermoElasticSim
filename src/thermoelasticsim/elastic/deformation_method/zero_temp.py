#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零温显式形变法弹性常数计算模块

该模块实现零温条件下通过显式形变法计算弹性常数。采用三层架构：
底层结构弛豫 → 中层工作流管理 → 高层数据求解

基本原理：
1. 制备无应力基态
2. 施加微小形变
3. 优化原子位置
4. 测量应力响应
5. 线性拟合求解弹性常数

数学基础：
胡克定律的张量形式：:math:`\\sigma = C : \\varepsilon`

在Voigt记号下：:math:`\\sigma = C \\cdot \\varepsilon`

其中：
- :math:`\\sigma`：应力向量 (6×1)
- :math:`C`：弹性常数矩阵 (6×6)
- :math:`\\varepsilon`：应变向量 (6×1)

.. moduleauthor:: Gilbert Young
.. created:: 2024-10-20
.. modified:: 2025-07-11
.. version:: 4.0.0
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.potentials import Potential
from thermoelasticsim.utils.optimizers import (
    LBFGSOptimizer,
    BFGSOptimizer,
    CGOptimizer,
)
from thermoelasticsim.utils.utils import TensorConverter, EV_TO_GPA

logger = logging.getLogger(__name__)


@dataclass
class DeformationResult:
    """
    单次形变计算结果的数据容器

    Attributes
    ----------
    strain_voigt : numpy.ndarray
        Voigt记号应变向量，形状 (6,)
    stress_voigt : numpy.ndarray
        Voigt记号应力向量，形状 (6,)
    converged : bool
        结构优化是否收敛
    deformation_matrix : numpy.ndarray
        形变矩阵F，形状 (3,3)
    """

    strain_voigt: np.ndarray
    stress_voigt: np.ndarray
    converged: bool
    deformation_matrix: np.ndarray


class StructureRelaxer:
    """
    结构弛豫计算引擎

    负责执行结构优化计算，包括完全弛豫和内部弛豫两种模式。

    Parameters
    ----------
    optimizer_type : str, optional
        优化器类型，默认 'L-BFGS'
    optimizer_params : dict, optional
        优化器参数字典

    Attributes
    ----------
    optimizer_type : str
        使用的优化器类型
    optimizer_params : dict
        优化器配置参数

    Notes
    -----
    两种弛豫模式的区别：

    - **完全弛豫**：同时优化原子位置和晶胞参数，用于制备无应力基态
    - **内部弛豫**：仅优化原子位置，保持晶胞形状固定，用于形变后的平衡

    Examples
    --------
    >>> relaxer = StructureRelaxer()
    >>> # 完全弛豫制备基态
    >>> converged = relaxer.full_relax(cell, potential)
    >>> # 形变后内部弛豫
    >>> converged = relaxer.internal_relax(deformed_cell, potential)
    """

    def __init__(
        self,
        optimizer_type: str = "L-BFGS",
        optimizer_params: Optional[Dict[str, Any]] = None,
        supercell_dims: Optional[Tuple[int, int, int]] = None,
    ):
        """
        初始化结构弛豫器

        Parameters
        ----------
        optimizer_type : str, optional
            优化器类型，支持 'L-BFGS', 'BFGS', 'GD'，默认 'L-BFGS'
        optimizer_params : dict, optional
            优化器参数，如果为None则使用默认参数
        supercell_dims : tuple, optional
            超胞维度(nx, ny, nz)，用于正确显示等效单胞参数

        Raises
        ------
        ValueError
            如果指定了不支持的优化器类型
        """
        # 验证优化器类型
        if optimizer_type not in ["L-BFGS", "BFGS", "GD"]:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        self.optimizer_type = optimizer_type
        self.supercell_dims = supercell_dims

        # 设置默认参数
        default_params = {
            "ftol": 1e-9,  # 函数收敛阈值
            "gtol": 1e-7,  # 梯度收敛阈值
            "maxiter": 200000,  # 最大迭代次数（增加以应对剪切变形）
        }

        if optimizer_params is None:
            self.optimizer_params = default_params
        else:
            # 合并用户参数和默认参数，但排除optimizer_type等不是LBFGSOptimizer参数的键
            filtered_params = {
                k: v
                for k, v in optimizer_params.items()
                if k not in ["optimizer_type", "optimizer_params"]
            }
            self.optimizer_params = {**default_params, **filtered_params}

        logger.debug(f"初始化StructureRelaxer，优化器: {optimizer_type}")
        logger.debug(f"优化器参数: {self.optimizer_params}")

    def full_relax(self, cell: Cell, potential: Potential) -> bool:
        """
        执行完全弛豫：同时优化原子位置和晶胞参数

        完全弛豫用于制备无应力基态，是零温弹性常数计算的第一步。
        通过同时优化原子位置和晶胞参数，消除系统内应力。

        Parameters
        ----------
        cell : Cell
            待优化的晶胞对象，会被直接修改
        potential : Potential
            势能函数对象

        Returns
        -------
        bool
            优化是否成功收敛

        Notes
        -----
        数学表述：

        .. math::
            \\min_{r,h} E(r,h) \\quad \\text{s.t.} \\quad \\sigma = 0

        其中：
        - :math:`r`：原子位置
        - :math:`h`：晶胞参数
        - :math:`E`：总能量
        - :math:`\\sigma`：应力张量

        Examples
        --------
        >>> relaxer = StructureRelaxer()
        >>> converged = relaxer.full_relax(cell, potential)
        >>> if converged:
        ...     print("成功制备无应力基态")
        """
        logger.info("开始完全弛豫：优化原子位置和晶胞参数")

        # 记录初始能量用于监控
        initial_energy = potential.calculate_energy(cell)
        logger.debug(f"初始总能量: {initial_energy:.6f} eV")

        # 完全弛豫需要同时优化晶胞参数，当前仅 LBFGS 支持变胞。
        # 若用户传入 BFGS/GD，这里忽略并使用 LBFGS（记录信息）。
        if self.optimizer_type != "L-BFGS":
            logger.info(
                "完全弛豫需要变胞，已自动使用 L-BFGS（忽略 optimizer_type=%s）",
                self.optimizer_type,
            )

        optimizer = LBFGSOptimizer(
            supercell_dims=self.supercell_dims, **self.optimizer_params
        )

        # 执行完全弛豫（relax_cell=True 表示同时优化晶胞）
        converged, _ = optimizer.optimize(cell, potential, relax_cell=True)

        # 失败时尝试一次更宽松/更耐心的回退（提高迭代数）
        if not converged:
            fallback_params = {**self.optimizer_params}
            fallback_params["maxiter"] = max(
                2 * self.optimizer_params.get("maxiter", 1000), 10000
            )
            logger.warning(
                "完全弛豫未收敛，尝试回退设置: maxiter=%d", fallback_params["maxiter"]
            )
            optimizer = LBFGSOptimizer(
                supercell_dims=self.supercell_dims, **fallback_params
            )
            converged, _ = optimizer.optimize(cell, potential, relax_cell=True)

        # 记录优化结果
        final_energy = potential.calculate_energy(cell)
        energy_change = final_energy - initial_energy

        logger.info(f"完全弛豫{'成功' if converged else '失败'}")
        logger.debug(f"最终总能量: {final_energy:.6f} eV")
        logger.debug(f"能量变化: {energy_change:.6f} eV")

        # 收敛性警告
        if not converged:
            logger.warning("完全弛豫未收敛，可能影响后续计算精度")

        return converged

    def internal_relax(self, cell: Cell, potential: Potential) -> bool:
        """
        执行内部弛豫：仅优化原子位置，保持晶胞形状固定

        内部弛豫用于形变后的结构优化，在保持宏观形变的前提下，
        寻找原子的最优位置配置。

        Parameters
        ----------
        cell : Cell
            待优化的晶胞对象，会被直接修改
        potential : Potential
            势能函数对象

        Returns
        -------
        bool
            优化是否成功收敛

        Notes
        -----
        数学表述：

        .. math::
            \\min_{r} E(r,h_{\\text{fixed}}) \\quad \\text{s.t.} \\quad h = \\text{const}

        其中：
        - :math:`r`：原子位置（优化变量）
        - :math:`h_{\\text{fixed}}`：固定的晶胞参数
        - :math:`E`：总能量

        Examples
        --------
        >>> relaxer = StructureRelaxer()
        >>> # 先施加形变
        >>> cell.apply_deformation(deformation_matrix)
        >>> # 再内部弛豫
        >>> converged = relaxer.internal_relax(cell, potential)
        """
        logger.debug("开始内部弛豫：仅优化原子位置")

        # 记录初始能量
        initial_energy = potential.calculate_energy(cell)
        logger.debug(f"形变后初始能量: {initial_energy:.6f} eV")

        # 内部弛豫（固定晶胞）优先采用用户指定的优化器，推荐 BFGS。

        def _build_optimizer(opt_type: str):
            params = self.optimizer_params
            if opt_type == "BFGS":
                tol = params.get(
                    "tol",
                    min(params.get("ftol", 1e-9), params.get("gtol", 1e-7)),
                )
                maxiter = params.get("maxiter", 10000)
                return BFGSOptimizer(tol=tol, maxiter=maxiter)
            if opt_type == "L-BFGS":
                return LBFGSOptimizer(
                    supercell_dims=self.supercell_dims,
                    ftol=params.get("ftol", 1e-9),
                    gtol=params.get("gtol", 1e-7),
                    maxiter=params.get("maxiter", 10000),
                )
            raise ValueError(f"未知优化器类型: {opt_type}")

        # 构建回退序列
        primary = self.optimizer_type
        if primary == "BFGS":
            sequence = ["L-BFGS", "BFGS", "CG", "BFGS"]
        elif primary == "L-BFGS":
            sequence = ["L-BFGS", "BFGS", "CG", "BFGS"]
        else:
            sequence = ["L-BFGS", "BFGS", "CG"]

        converged = False
        for idx, opt_type in enumerate(sequence):
            optimizer = (
                _build_optimizer(opt_type)
                if opt_type != "CG"
                else CGOptimizer(
                    tol=self.optimizer_params.get("gtol", 1e-6),
                    maxiter=self.optimizer_params.get("maxiter", 10000),
                )
            )
            logger.info(
                "内部弛豫尝试优化器: %s (%d/%d)", opt_type, idx + 1, len(sequence)
            )
            if opt_type == "L-BFGS":
                converged, _ = optimizer.optimize(cell, potential, relax_cell=False)
            else:
                converged, _ = optimizer.optimize(cell, potential)
            if converged:
                break
            else:
                logger.warning("内部弛豫使用 %s 未收敛，进入回退。", opt_type)

        # 记录优化结果
        final_energy = potential.calculate_energy(cell)
        energy_change = final_energy - initial_energy

        logger.debug(f"内部弛豫{'成功' if converged else '失败'}")
        logger.debug(f"最终能量: {final_energy:.6f} eV")
        logger.debug(f"能量变化: {energy_change:.6f} eV")

        # 收敛性警告
        if not converged:
            logger.warning("内部弛豫未收敛，可能影响应力计算精度")

        return converged


class ZeroTempDeformationCalculator:
    """
    零温显式形变法计算器

    管理从基态制备到弹性常数求解的完整计算流程。
    实现标准的显式形变法：制备基态→施加形变→内部弛豫→测量应力→线性拟合。

    Parameters
    ----------
    cell : Cell
        待计算的晶胞对象
    potential : Potential
        势能函数对象
    delta : float, optional
        应变幅度，默认0.005 (0.5%)
    num_steps : int, optional
        每个应变分量的步数，默认5（教学模式）
    relaxer_params : dict, optional
        结构弛豫器参数

    Attributes
    ----------
    cell : Cell
        原始晶胞对象
    potential : Potential
        势能函数对象
    delta : float
        应变幅度
    num_steps : int
        形变步数
    relaxer : StructureRelaxer
        结构弛豫器实例
    reference_stress : numpy.ndarray
        基态参考应力张量

    Notes
    -----
    计算流程包含5个步骤：

    1. **基态制备**：完全弛豫获得无应力基态
    2. **形变生成**：生成6个Voigt分量的形变矩阵序列
    3. **应力计算**：对每个形变施加内部弛豫并测量应力
    4. **数据收集**：收集所有应力-应变数据对
    5. **线性拟合**：通过最小二乘法求解弹性常数矩阵

    应变生成策略：
    - 对称应变：:math:`\\varepsilon_{11}, \\varepsilon_{22}, \\varepsilon_{33}`
    - 剪切应变：:math:`\\varepsilon_{23}, \\varepsilon_{13}, \\varepsilon_{12}`
    - 应变范围：:math:`[-\\delta, +\\delta]`，均匀分布

    Examples
    --------
    >>> calculator = ZeroTempDeformationCalculator(cell, potential, delta=0.005)
    >>> elastic_matrix, r2_score = calculator.calculate()
    >>> print(f"弹性常数矩阵 (GPa):\\n{elastic_matrix}")
    >>> print(f"拟合优度 R²: {r2_score:.6f}")
    """

    def __init__(
        self,
        cell: Cell,
        potential: Potential,
        delta: float = 0.005,
        num_steps: int = 5,
        relaxer_params: Optional[Dict[str, Any]] = None,
        supercell_dims: Optional[Tuple[int, int, int]] = None,
    ):
        """
        初始化零温形变计算器

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能函数对象
        delta : float, optional
            应变幅度，建议范围0.001-0.01，默认0.005
        num_steps : int, optional
            每个应变分量的步数，1为生产模式，>1为教学模式，默认5
        relaxer_params : dict, optional
            传递给StructureRelaxer的参数
        supercell_dims : tuple, optional
            超胞维度(nx, ny, nz)，用于正确显示等效单胞参数

        Raises
        ------
        ValueError
            如果delta或num_steps参数不合理
        """
        # 参数有效性检查
        if delta <= 0 or delta > 0.1:
            raise ValueError(f"应变幅度delta={delta}不合理，建议范围0.001-0.01")
        if num_steps <= 0:
            raise ValueError(f"步数num_steps={num_steps}必须为正整数")

        # 保存核心对象
        self.cell = cell
        self.potential = potential
        self.delta = delta
        self.num_steps = num_steps
        self.supercell_dims = supercell_dims

        # 创建结构弛豫器
        if relaxer_params:
            # 提取optimizer_type和optimizer_params
            optimizer_type = relaxer_params.get("optimizer_type", "L-BFGS")
            optimizer_params = relaxer_params.get("optimizer_params", None)
            self.relaxer = StructureRelaxer(
                optimizer_type=optimizer_type,
                optimizer_params=optimizer_params,
                supercell_dims=self.supercell_dims,
            )
        else:
            self.relaxer = StructureRelaxer(supercell_dims=self.supercell_dims)

        # 初始化状态变量
        self.reference_stress = None  # 基态参考应力

        # 记录初始化信息
        logger.info(f"初始化ZeroTempDeformationCalculator")
        logger.info(f"应变幅度: {delta:.2e} ({delta*100:.4f}%)")
        logger.info(f"形变步数: {num_steps}")

        # 物理合理性提醒
        if delta > 0.01:
            logger.warning(f"应变幅度{delta:.3f}较大，可能超出线性弹性范围")

    def _get_equivalent_unit_cell_parameter(self, lattice_vectors: np.ndarray) -> float:
        """
        计算等效单胞晶格参数a

        对于超胞，将晶格矢量长度除以对应的超胞维度
        """
        if self.supercell_dims is None:
            # 如果没有提供超胞信息，直接返回第一个晶格矢量的长度
            return np.linalg.norm(lattice_vectors[0])
        else:
            # 返回等效单胞参数
            return np.linalg.norm(lattice_vectors[0]) / self.supercell_dims[0]

    def calculate(self) -> Tuple[np.ndarray, float]:
        """
        执行完整的零温弹性常数计算

        Returns
        -------
        tuple[numpy.ndarray, float]
            弹性常数矩阵(GPa)和拟合优度R²的元组

        Notes
        -----
        该方法执行5步计算流程：

        1. 制备无应力基态
        2. 生成形变矩阵序列
        3. 计算每个形变的应力响应
        4. 收集应力-应变数据
        5. 线性拟合求解弹性常数

        拟合质量评估：
        - R² > 0.999：优秀
        - R² > 0.99：良好
        - R² < 0.99：需改进

        Examples
        --------
        >>> calculator = ZeroTempDeformationCalculator(cell, potential)
        >>> C_matrix, r2 = calculator.calculate()
        >>>
        >>> # 检查拟合质量
        >>> if r2 > 0.999:
        ...     print("拟合质量优秀")
        >>> else:
        ...     print(f"拟合质量R²={r2:.6f}，可能需要调整参数")
        """
        logger.info("开始零温弹性常数计算")

        # 步骤1：制备无应力基态
        logger.info("步骤1/5：制备无应力基态")
        self._prepare_reference_state()

        # 步骤2：生成形变矩阵序列
        logger.info("步骤2/5：生成形变矩阵")
        deformation_matrices = self._generate_deformation_matrices()
        total_deformations = len(deformation_matrices)
        logger.info(f"生成{total_deformations}个形变矩阵")

        # 步骤3：计算所有形变的应力响应
        logger.info("步骤3/5：计算应力响应")
        results = []
        for i, F in enumerate(deformation_matrices):
            logger.info(f"计算形变 {i+1}/{total_deformations}")
            result = self._compute_single_deformation(F)
            results.append(result)

            # 收敛性监控
            if result.converged:
                logger.debug(f"形变{i+1}计算成功")
            else:
                logger.warning(f"形变{i+1}优化未收敛")

        # 步骤4：提取应力应变数据
        logger.info("步骤4/5：收集应力应变数据")
        strains = np.array([result.strain_voigt for result in results])
        stresses = np.array([result.stress_voigt for result in results])

        logger.info(f"收集到{len(strains)}组应力应变数据")
        logger.debug(f"应变数据形状: {strains.shape}")
        logger.debug(f"应力数据形状: {stresses.shape}")

        # 步骤5：线性拟合求解弹性常数
        logger.info("步骤5/5：拟合弹性常数")
        solver = ElasticConstantsSolver()
        elastic_matrix_eV, r2_score = solver.solve(strains, stresses)

        # 单位转换：eV/Å³ → GPa
        elastic_matrix_GPa = elastic_matrix_eV * EV_TO_GPA

        # 结果总结
        logger.info("零温弹性常数计算完成")
        logger.info(f"拟合优度R²: {r2_score:.6f}")

        # 拟合质量评估
        if r2_score > 0.999:
            logger.info("拟合质量：优秀")
        elif r2_score > 0.99:
            logger.info("拟合质量：良好")
        else:
            logger.warning("拟合质量：需要改进，建议检查参数设置")

        return elastic_matrix_GPa, r2_score

    def _prepare_reference_state(self) -> None:
        """
        制备无应力基态

        通过完全弛豫获得无应力基态，并计算基态应力张量作为后续计算的参考。
        所有形变计算的应力都将相对于此基态应力。

        Notes
        -----
        基态制备的重要性：

        1. 消除初始应力：确保所有形变从相同的无应力状态开始
        2. 提供参考点：基态应力作为后续相对应力计算的零点
        3. 保证线性：在无应力基态附近，应力-应变关系最接近线性
        """
        logger.debug("制备无应力基态")

        # 记录初始状态
        initial_energy = self.potential.calculate_energy(self.cell)
        initial_lattice = self.cell.lattice_vectors.copy()
        initial_a = self._get_equivalent_unit_cell_parameter(initial_lattice)
        logger.info(f"弛豫前状态:")
        logger.info(f"  初始总能量: {initial_energy:.8f} eV")
        logger.info(f"  每原子能量: {initial_energy/self.cell.num_atoms:.8f} eV/atom")
        logger.info(f"  初始等效单胞晶格常数: a = {initial_a:.6f} Å")
        logger.info(f"  初始体积: {self.cell.volume:.6f} Å³")

        # 执行完全弛豫
        logger.info("开始完全弛豫...")
        converged = self.relaxer.full_relax(self.cell, self.potential)

        # 检查弛豫结果
        final_energy = self.potential.calculate_energy(self.cell)
        final_lattice = self.cell.lattice_vectors.copy()
        final_a = self._get_equivalent_unit_cell_parameter(final_lattice)
        energy_change = final_energy - initial_energy

        logger.info(f"弛豫后状态:")
        logger.info(f"  最终总能量: {final_energy:.8f} eV")
        logger.info(f"  每原子能量: {final_energy/self.cell.num_atoms:.8f} eV/atom")
        logger.info(f"  能量变化: {energy_change:.8f} eV")
        logger.info(f"  弛豫后等效单胞晶格常数: a = {final_a:.6f} Å")
        logger.info(
            f"  晶格常数变化: Δa = {final_a - initial_a:.6f} Å ({(final_a - initial_a)/initial_a*100:.3f}%)"
        )
        logger.info(f"  最终体积: {self.cell.volume:.6f} Å³")

        # 与EAM Al1文献值比较
        literature_a = 4.045
        a_error = abs(final_a - literature_a) / literature_a * 100
        logger.info(f"  与EAM Al1文献值(4.045 Å)比较: 误差 = {a_error:.2f}%")

        if not converged:
            logger.warning("基态弛豫未完全收敛，可能影响计算精度")
        else:
            logger.info("✓ 完全弛豫成功收敛")

        # 计算并保存基态应力张量
        self.reference_stress = self._calculate_stress_tensor(self.cell)

        # 将应力转换为GPa单位并详细输出
        stress_gpa = self.reference_stress * 160.2176  # eV/Å³ to GPa conversion
        logger.info("基态应力张量 (GPa):")
        for i in range(3):
            row_str = "  ".join(f"{stress_gpa[i,j]:10.6f}" for j in range(3))
            logger.info(f"  [{row_str}]")

        stress_magnitude = np.linalg.norm(stress_gpa)
        logger.info(f"基态应力大小: {stress_magnitude:.6f} GPa")

        # 检查基态应力大小
        max_stress = np.max(np.abs(self.reference_stress))
        if max_stress > 0.01:  # 0.01 eV/Å³ ≈ 16 GPa
            logger.warning(
                f"基态应力较大({max_stress:.6f} eV/Å³ = {max_stress*160.2176:.2f} GPa)，可能未完全弛豫"
            )
        else:
            logger.info("✓ 基态应力较小，弛豫质量良好")

        logger.info("无应力基态制备完成")

    def _generate_deformation_matrices(self) -> List[np.ndarray]:
        """
        生成形变矩阵序列

        为6个独立的Voigt应变分量生成形变矩阵。每个分量按指定步数
        在±delta范围内均匀分布。

        Returns
        -------
        List[numpy.ndarray]
            形变矩阵列表，每个矩阵形状为(3,3)

        Notes
        -----
        Voigt应变分量对应关系：

        - ε₁ = ε₁₁：x方向正应变
        - ε₂ = ε₂₂：y方向正应变
        - ε₃ = ε₃₃：z方向正应变
        - ε₄ = 2ε₂₃：yz平面剪切应变
        - ε₅ = 2ε₁₃：xz平面剪切应变
        - ε₆ = 2ε₁₂：xy平面剪切应变

        形变矩阵构造：

        .. math::
            F = I + \\varepsilon

        其中I是单位矩阵，ε是应变张量。
        """
        logger.debug("生成形变矩阵序列")

        deformation_matrices = []

        # 确定应变幅度序列
        if self.num_steps == 1:
            # 生产模式：仅使用正应变
            strain_amplitudes = [self.delta]
            logger.debug("生产模式：使用单一正应变")
        else:
            # 教学模式：使用正负应变的对称分布
            strain_amplitudes = np.linspace(-self.delta, self.delta, self.num_steps)
            # 移除零应变点（避免无意义计算）
            strain_amplitudes = strain_amplitudes[strain_amplitudes != 0]
            logger.debug(f"教学模式：使用{len(strain_amplitudes)}个应变幅度")

        # Voigt应变分量到张量指标的映射
        voigt_to_tensor = {
            0: (0, 0),  # ε₁₁
            1: (1, 1),  # ε₂₂
            2: (2, 2),  # ε₃₃
            3: (1, 2),  # ε₂₃ (和ε₃₂)
            4: (0, 2),  # ε₁₃ (和ε₃₁)
            5: (0, 1),  # ε₁₂ (和ε₂₁)
        }

        # 为每个Voigt分量生成形变矩阵
        for voigt_idx in range(6):
            i, j = voigt_to_tensor[voigt_idx]

            for amplitude in strain_amplitudes:
                # 创建应变张量
                strain_tensor = np.zeros((3, 3))

                # 剪切应变需要保证对称性，幅度要减半以保持工程剪切应变γ = amplitude
                if i != j:
                    # 对于剪切应变，工程剪切应变 γ = 2ε，所以 ε = γ/2 = amplitude/2
                    strain_tensor[i, j] = amplitude / 2
                    strain_tensor[j, i] = amplitude / 2
                else:
                    # 对于正应变，直接使用amplitude
                    strain_tensor[i, j] = amplitude

                # 构造形变矩阵：F = I + ε
                F = np.eye(3) + strain_tensor
                deformation_matrices.append(F)

        logger.debug(f"生成{len(deformation_matrices)}个形变矩阵")

        return deformation_matrices

    def _compute_single_deformation(
        self, deformation_matrix: np.ndarray
    ) -> DeformationResult:
        """
        计算单个形变的应力响应

        对给定的形变矩阵，执行形变→内部弛豫→应力计算的完整流程。

        Parameters
        ----------
        deformation_matrix : numpy.ndarray
            形变矩阵F，形状(3,3)

        Returns
        -------
        DeformationResult
            包含应力、应变和收敛信息的结果对象

        Notes
        -----
        计算步骤：

        1. **施加形变**：cell.apply_deformation(F)
        2. **内部弛豫**：优化原子位置，保持晶胞形状
        3. **计算应力**：使用virial公式计算应力张量
        4. **计算应变**：从形变矩阵提取应变张量
        5. **Voigt转换**：将张量转换为Voigt向量

        应变张量计算：

        .. math::
            \\varepsilon = \\frac{1}{2}(F + F^T) - I

        其中F是形变矩阵，I是单位矩阵。
        """
        logger.debug("计算单个形变的应力响应")

        # 复制原始晶胞（避免修改原始数据）
        deformed_cell = self.cell.copy()

        # 记录形变前状态
        initial_energy_def = self.potential.calculate_energy(deformed_cell)
        initial_lattice_def = deformed_cell.lattice_vectors.copy()
        initial_a_def = self._get_equivalent_unit_cell_parameter(initial_lattice_def)

        # 施加形变
        deformed_cell.apply_deformation(deformation_matrix)
        logger.debug("已施加形变到晶胞")

        # 记录形变后状态（首次形变）
        after_deform_energy = self.potential.calculate_energy(deformed_cell)
        after_deform_lattice = deformed_cell.lattice_vectors.copy()
        after_deform_a = self._get_equivalent_unit_cell_parameter(after_deform_lattice)

        # 计算应变张量和应变幅度（首次形变）
        strain_tensor = 0.5 * (deformation_matrix + deformation_matrix.T) - np.eye(3)
        strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")
        strain_magnitude = np.linalg.norm(strain_voigt)

        logger.info(f"    形变详情:")
        logger.info(
            f"      应变Voigt向量: [{strain_voigt[0]:8.6f}, {strain_voigt[1]:8.6f}, {strain_voigt[2]:8.6f}, {strain_voigt[3]:8.6f}, {strain_voigt[4]:8.6f}, {strain_voigt[5]:8.6f}]"
        )
        logger.info(f"      应变幅度: {strain_magnitude:.6f}")
        logger.info(
            f"      等效单胞晶格常数变化: {initial_a_def:.6f} → {after_deform_a:.6f} Å"
        )
        logger.info(
            f"      能量变化: {after_deform_energy - initial_energy_def:.8f} eV"
        )

        # 剪切检测（仅对剪切分量应用更强的收敛策略与降幅重试）
        shear_pairs = [(1, 2), (0, 2), (0, 1)]
        shear_pair_used = None
        for ii, jj in shear_pairs:
            if abs(strain_tensor[ii, jj]) > 0:
                shear_pair_used = (ii, jj)
                break

        # 内部弛豫（仅优化原子位置）
        logger.info(f"    开始内部弛豫 (固定晶胞，优化原子位置)...")

        # 对剪切分量，按更大的迭代数尝试一次
        if shear_pair_used is not None:
            logger.info("    剪切分量检测到，使用剪切专用收敛参数 (maxiter↑)")
            from typing import Any

            shear_params: dict[str, Any] = (
                getattr(self.relaxer, "optimizer_params", {}) or {}
            )
            shear_params = {**shear_params}
            shear_params["maxiter"] = max(
                int(shear_params.get("maxiter", 10000)), 20000
            )
            shear_relaxer = StructureRelaxer(
                optimizer_type=getattr(self.relaxer, "optimizer_type", "L-BFGS"),
                optimizer_params=shear_params,
                supercell_dims=self.supercell_dims,
            )
            converged = shear_relaxer.internal_relax(deformed_cell, self.potential)
        else:
            converged = self.relaxer.internal_relax(deformed_cell, self.potential)

        # 若剪切仍未收敛：将剪切应变幅度减半，重试一次
        if (shear_pair_used is not None) and (not converged):
            ii, jj = shear_pair_used
            logger.warning("    剪切未收敛，降幅重试：γ -> γ/2")
            # 从基态重新复制、应用半幅剪切
            deformed_cell_half = self.cell.copy()
            # 原 off-diagonal 应变分量 ε = γ/2，在 F 中体现在 F[ii,jj] 与 F[jj,ii]
            eps_old = strain_tensor[ii, jj]
            eps_new = 0.5 * eps_old
            F_half = np.eye(3)
            F_half[ii, jj] += eps_new
            F_half[jj, ii] += eps_new
            deformed_cell_half.apply_deformation(F_half)

            # 重新弛豫
            converged_half = shear_relaxer.internal_relax(
                deformed_cell_half, self.potential
            )
            # 若半幅成功，则用半幅的形变/应变数据替换
            if converged_half:
                deformed_cell = deformed_cell_half
                deformation_matrix = F_half
                strain_tensor = 0.5 * (
                    deformation_matrix + deformation_matrix.T
                ) - np.eye(3)
                strain_voigt = TensorConverter.to_voigt(
                    strain_tensor, tensor_type="strain"
                )
                strain_magnitude = np.linalg.norm(strain_voigt)
                logger.info("    降幅重试成功，采用半幅剪切结果")
                converged = True

        # 记录弛豫后状态
        after_relax_energy = self.potential.calculate_energy(deformed_cell)
        logger.info(f"    内部弛豫完成:")
        logger.info(f"      收敛状态: {'✓ 成功' if converged else '✗ 未收敛'}")
        logger.info(
            f"      弛豫能量变化: {after_relax_energy - after_deform_energy:.8f} eV"
        )
        logger.info(
            f"      总能量变化: {after_relax_energy - initial_energy_def:.8f} eV"
        )

        # 计算应力张量
        stress_tensor = self._calculate_stress_tensor(deformed_cell)

        # 应力相对于基态（消除基态应力的影响）
        stress_tensor_relative = stress_tensor - self.reference_stress

        # 转换为Voigt记号
        stress_voigt = TensorConverter.to_voigt(
            stress_tensor_relative, tensor_type="stress"
        )

        # 详细输出应力信息
        stress_gpa = stress_voigt * 160.2176  # 转换为GPa
        logger.info(f"    应力计算结果:")
        logger.info(
            f"      相对应力Voigt向量 (GPa): [{stress_gpa[0]:8.3f}, {stress_gpa[1]:8.3f}, {stress_gpa[2]:8.3f}, {stress_gpa[3]:8.3f}, {stress_gpa[4]:8.3f}, {stress_gpa[5]:8.3f}]"
        )

        # 计算应力应变比 (粗略的弹性模量指示)
        if strain_magnitude > 1e-8:
            stress_strain_ratio = np.linalg.norm(stress_gpa) / strain_magnitude
            logger.info(
                f"      应力应变比: {stress_strain_ratio:.2f} GPa (粗略弹性模量指示)"
            )

        logger.debug(f"应变(Voigt): {strain_voigt}")
        logger.debug(f"应力(Voigt): {stress_voigt}")

        return DeformationResult(
            strain_voigt=strain_voigt,
            stress_voigt=stress_voigt,
            converged=converged,
            deformation_matrix=deformation_matrix,
        )

    def _calculate_stress_tensor(self, cell: Cell) -> np.ndarray:
        """
        计算应力张量

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        numpy.ndarray
            应力张量，形状(3,3)

        Notes
        -----
        应力张量通过virial公式计算：

        .. math::
            \\sigma_{\\alpha\\beta} = \\frac{1}{V} \\sum_i r_{i\\alpha} f_{i\\beta}

        其中：
        - :math:`V`：晶胞体积
        - :math:`r_{i\\alpha}`：第i个原子在α方向的位置
        - :math:`f_{i\\beta}`：第i个原子在β方向受到的力

        应力张量必须是对称的，任何非对称性都被平均处理。
        """
        # 使用Cell类的应力计算方法
        stress_tensor = cell.calculate_stress_tensor(self.potential)

        # 确保对称性（消除数值误差导致的微小非对称性）
        stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

        return stress_tensor


class ElasticConstantsSolver:
    """
    弹性常数求解器

    从应力应变数据通过线性回归求解弹性常数矩阵。
    支持最小二乘法和岭回归两种方法，并提供拟合质量评估。

    Methods
    -------
    solve(strains, stresses, method='least_squares')
        求解弹性常数矩阵

    Notes
    -----
    数学原理：

    根据胡克定律：:math:`\\sigma = C \\cdot \\varepsilon`

    其中：
    - :math:`\\sigma`：应力向量 (N×6)
    - :math:`C`：弹性常数矩阵 (6×6)
    - :math:`\\varepsilon`：应变向量 (N×6)

    最小二乘求解：

    .. math::
        \\min_C ||\\sigma - C \\cdot \\varepsilon||^2

    解析解：

    .. math::
        C = (\\varepsilon^T \\varepsilon)^{-1} \\varepsilon^T \\sigma

    Examples
    --------
    >>> solver = ElasticConstantsSolver()
    >>> C_matrix, r2_score = solver.solve(strains, stresses)
    >>> print(f"弹性常数矩阵:\\n{C_matrix}")
    >>> print(f"拟合优度: {r2_score:.6f}")
    """

    def __init__(self):
        """初始化弹性常数求解器"""
        logger.debug("初始化ElasticConstantsSolver")

    def solve(
        self,
        strains: np.ndarray,
        stresses: np.ndarray,
        method: str = "least_squares",
        alpha: float = 1e-5,
    ) -> Tuple[np.ndarray, float]:
        """
        求解弹性常数矩阵

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据，形状(N, 6)，N为数据点数
        stresses : numpy.ndarray
            应力数据，形状(N, 6)
        method : str, optional
            求解方法，支持'least_squares'和'ridge'，默认'least_squares'
        alpha : float, optional
            岭回归正则化参数，仅在method='ridge'时使用，默认1e-5

        Returns
        -------
        tuple[numpy.ndarray, float]
            弹性常数矩阵(6,6)和拟合优度R²

        Raises
        ------
        ValueError
            如果输入数据格式不正确或求解方法不支持

        Notes
        -----
        支持的求解方法：

        1. **最小二乘法** ('least_squares')：
           标准线性回归，适用于大多数情况

        2. **岭回归** ('ridge')：
           加入L2正则化，适用于病态矩阵情况

        拟合优度R²计算：

        .. math::
            R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

        其中：
        - :math:`SS_{res} = \\sum(y_i - \\hat{y}_i)^2`：残差平方和
        - :math:`SS_{tot} = \\sum(y_i - \\bar{y})^2`：总平方和

        Examples
        --------
        >>> solver = ElasticConstantsSolver()
        >>> # 使用最小二乘法
        >>> C, r2 = solver.solve(strains, stresses)
        >>> # 使用岭回归（适用于病态情况）
        >>> C, r2 = solver.solve(strains, stresses, method='ridge', alpha=1e-3)
        """
        logger.info(f"开始求解弹性常数，方法: {method}")

        # 数据有效性验证
        self._validate_data(strains, stresses)

        # 根据方法选择求解器
        if method == "least_squares":
            C, r2_score = self._least_squares_solve(strains, stresses)
        elif method == "ridge":
            C, r2_score = self._ridge_regression_solve(strains, stresses, alpha)
        else:
            raise ValueError(f"不支持的求解方法: {method}")

        # 若全局拟合质量差或矩阵明显不物理，使用“按应变分量逐列稳健回归”的保险方案
        needs_rescue = (
            not np.allclose(C, C.T, rtol=1e-6, atol=1e-6)
            or np.any(np.isnan(C))
            or np.isinf(np.linalg.cond(strains))
            or r2_score < 0.95
        )
        if needs_rescue:
            C_alt, r2_alt = self._per_mode_columnwise_solve(strains, stresses)
            # 对称化，提升物理一致性
            C_alt = 0.5 * (C_alt + C_alt.T)
            # 若保险方案更好，则采用
            if r2_alt >= r2_score or r2_score < 0.9:
                C, r2_score = C_alt, r2_alt

        # 如仍不理想，再尝试立方晶系约束拟合（三参数：C11, C12, C44）
        if r2_score < 0.95:
            C_cubic, r2_cubic = self._cubic_constrained_fit(strains, stresses)
            if r2_cubic >= r2_score:
                C, r2_score = C_cubic, r2_cubic

        logger.info(f"弹性常数求解完成，R²: {r2_score:.6f}")

        # 物理合理性检查
        self._validate_elastic_matrix(C)

        return C, r2_score

    def _per_mode_columnwise_solve(
        self, strains: np.ndarray, stresses: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        逐列稳健求解 C：
        - 对第 k 个 Voigt 分量，仅用该分量非零且其余分量近零的样本行，拟合 C[:, k]
        - 适用于我们“单分量逐一施加形变”的数据结构，可显著降低全局拟合的病态和串扰
        """
        num_modes = 6
        C = np.zeros((num_modes, num_modes), dtype=np.float64)

        # 判定“其余分量近零”的阈值（相对量级）
        other_tol = 1e-12

        for k in range(num_modes):
            eps_k = strains[:, k]
            others = np.delete(strains, k, axis=1)
            mask = (np.abs(eps_k) > other_tol) & (
                np.max(np.abs(others), axis=1) < other_tol
            )

            if not np.any(mask):
                # 回退：如果严格筛选无样本，放宽对其它分量的阈值
                mask = np.abs(eps_k) > other_tol
            X = eps_k[mask][:, None]  # (M,1)
            Y = stresses[mask]  # (M,6)

            if X.shape[0] == 0:
                # 无法求解该列，保持零，后续对称化会缓解
                continue

            # 一维最小二乘：X @ col = Y  →  col = (X^T X)^-1 X^T Y
            denom = float(X.T @ X)
            if denom < 1e-20:
                continue
            col_k = (X.T @ Y) / denom  # 形状 (1,6)
            C[:, k] = col_k.ravel()

        # 拟合优度：用组装出的 C 复原应力
        Y_pred = strains @ C.T
        ss_res = np.sum((stresses - Y_pred) ** 2)
        ss_tot = np.sum((stresses - np.mean(stresses)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
        return C, r2

    def _cubic_constrained_fit(
        self, strains: np.ndarray, stresses: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        在立方晶系假设下拟合：仅 C11, C12, C44。
        使用“单分量形变”的行，分别以稳健统计（中位数斜率）估计：
        - C11: 使用 ε11 行的 σ11/ε11、ε22 行的 σ22/ε22、ε33 行的 σ33/ε33 的中位数
        - C12: 使用 ε11 行的 σ22/ε11、σ33/ε11 等跨分量比值的中位数（再与其它轴对换求中位数）
        - C44: 使用 γ23 行的 σ23/γ23、γ13 行的 σ13/γ13、γ12 行的 σ12/γ12 的中位数
        """

        def median_slope(x: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
            mask = np.abs(x) > eps
            if not np.any(mask):
                return 0.0
            ratios = y[mask] / x[mask]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size == 0:
                return 0.0
            return float(np.median(ratios))

        # 收集按分量的行索引
        idx_eps = [np.where(np.abs(strains[:, k]) > 1e-12)[0] for k in range(6)]

        # C11: 取各主轴的自响应斜率的中位数
        c11_candidates = []
        for axis in range(3):  # 0→σ11/ε11, 1→σ22/ε22, 2→σ33/ε33
            rows = idx_eps[axis]
            if rows.size:
                c11_candidates.append(
                    median_slope(strains[rows, axis], stresses[rows, axis])
                )
        C11 = float(np.median(c11_candidates)) if c11_candidates else 0.0

        # C12: 取“交叉主应力/主应变”的中位数（多个轴互换后再取中位数）
        c12_candidates = []
        # ε11 → σ22, σ33； ε22 → σ11, σ33； ε33 → σ11, σ22
        for eps_axis in range(3):
            rows = idx_eps[eps_axis]
            if rows.size:
                for sig_axis in range(3):
                    if sig_axis == eps_axis:
                        continue
                    c12_candidates.append(
                        median_slope(strains[rows, eps_axis], stresses[rows, sig_axis])
                    )
        C12 = float(np.median(c12_candidates)) if c12_candidates else 0.0

        # C44: 取剪切自响应的中位数（γ23→σ23 等）
        shear_pairs = [(3, 3), (4, 4), (5, 5)]  # (εk, σk) in Voigt ordering
        c44_candidates = []
        for eps_k, sig_k in shear_pairs:
            rows = idx_eps[eps_k]
            if rows.size:
                c44_candidates.append(
                    median_slope(strains[rows, eps_k], stresses[rows, sig_k])
                )
        C44 = float(np.median(c44_candidates)) if c44_candidates else 0.0

        # 组装立方晶系 6×6 矩阵
        C = np.zeros((6, 6), dtype=np.float64)
        # 主块
        C[0, 0] = C[1, 1] = C[2, 2] = C11
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
        # 剪切块
        C[3, 3] = C[4, 4] = C[5, 5] = C44

        # 用该 C 评估 R²
        Y_pred = strains @ C.T
        ss_res = np.sum((stresses - Y_pred) ** 2)
        ss_tot = np.sum((stresses - np.mean(stresses)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
        return C, r2

    def _validate_data(self, strains: np.ndarray, stresses: np.ndarray) -> None:
        """
        验证输入数据的有效性

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据
        stresses : numpy.ndarray
            应力数据

        Raises
        ------
        ValueError
            如果数据不符合要求
        """
        # 形状检查
        if strains.shape != stresses.shape:
            raise ValueError(
                f"应变和应力数据形状不匹配: {strains.shape} vs {stresses.shape}"
            )

        if strains.ndim != 2 or strains.shape[1] != 6:
            raise ValueError(f"应变数据必须是(N,6)形状，当前: {strains.shape}")

        if stresses.ndim != 2 or stresses.shape[1] != 6:
            raise ValueError(f"应力数据必须是(N,6)形状，当前: {stresses.shape}")

        # 数据点数量检查
        if strains.shape[0] < 6:
            raise ValueError(f"数据点数量{strains.shape[0]}少于未知数数量6")

        # 数值有效性检查
        if np.any(~np.isfinite(strains)):
            raise ValueError("应变数据包含无效值(NaN或Inf)")

        if np.any(~np.isfinite(stresses)):
            raise ValueError("应力数据包含无效值(NaN或Inf)")

        logger.debug(f"数据验证通过: {strains.shape[0]}个数据点")

    def _least_squares_solve(
        self, strains: np.ndarray, stresses: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        最小二乘法求解弹性常数

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据(N, 6)
        stresses : numpy.ndarray
            应力数据(N, 6)

        Returns
        -------
        tuple[numpy.ndarray, float]
            弹性常数矩阵(6,6)和拟合优度R²

        Notes
        -----
        求解线性方程组：

        .. math::
            \\varepsilon \\cdot C = \\sigma

        使用numpy.linalg.lstsq求解，该函数使用SVD分解处理病态矩阵。
        """
        logger.debug("使用最小二乘法求解")

        # 求解线性方程组：strains @ C = stresses
        # 注意：lstsq求解的是 X @ beta = y 形式，所以结果需要转置
        C, _, rank, _ = np.linalg.lstsq(strains, stresses, rcond=None)
        C = C.T  # 转置得到(6,6)弹性常数矩阵

        # 计算拟合优度R²
        predicted_stresses = strains @ C.T  # 预测应力
        ss_res = np.sum((stresses - predicted_stresses) ** 2)  # 残差平方和
        ss_tot = np.sum((stresses - np.mean(stresses)) ** 2)  # 总平方和

        # 处理零方差情况（所有应力都相同）
        if ss_tot == 0:
            if ss_res == 0:
                r2_score = 1.0  # 完美拟合：零数据零预测
            else:
                r2_score = 0.0  # 无法拟合：零数据非零预测
        else:
            r2_score = 1 - ss_res / ss_tot

        # 记录求解信息
        logger.debug(f"矩阵秩: {rank}/6")
        logger.debug(f"条件数: {np.linalg.cond(strains):.2e}")

        # 秩亏警告
        if rank < 6:
            logger.warning(f"应变矩阵秩亏({rank}/6)，可能导致解不唯一")

        # 病态矩阵警告
        if np.linalg.cond(strains) > 1e12:
            logger.warning("应变矩阵病态，建议使用岭回归")

        return C, r2_score

    def _ridge_regression_solve(
        self, strains: np.ndarray, stresses: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, float]:
        """
        岭回归求解弹性常数

        Parameters
        ----------
        strains : numpy.ndarray
            应变数据(N, 6)
        stresses : numpy.ndarray
            应力数据(N, 6)
        alpha : float
            正则化参数

        Returns
        -------
        tuple[numpy.ndarray, float]
            弹性常数矩阵(6,6)和拟合优度R²

        Notes
        -----
        岭回归目标函数：

        .. math::
            \\min_C ||\\sigma - C \\cdot \\varepsilon||^2 + \\alpha ||C||^2

        解析解：

        .. math::
            C = (\\varepsilon^T \\varepsilon + \\alpha I)^{-1} \\varepsilon^T \\sigma

        正则化参数α控制平滑程度：
        - α = 0：退化为最小二乘法
        - α较大：更平滑但可能欠拟合
        - α较小：接近最小二乘法但改善条件数
        """
        logger.debug(f"使用岭回归求解，正则化参数: {alpha}")

        # 岭回归求解
        XTX = strains.T @ strains  # ε^T ε
        XTX_reg = XTX + alpha * np.eye(XTX.shape[0])  # ε^T ε + αI
        XTy = strains.T @ stresses  # ε^T σ

        # 求解正则化方程组
        C = np.linalg.solve(XTX_reg, XTy).T

        # 计算拟合优度R²
        predicted_stresses = strains @ C.T
        ss_res = np.sum((stresses - predicted_stresses) ** 2)
        ss_tot = np.sum((stresses - np.mean(stresses)) ** 2)

        # 处理零方差情况（所有应力都相同）
        if ss_tot == 0:
            if ss_res == 0:
                r2_score = 1.0  # 完美拟合：零数据零预测
            else:
                r2_score = 0.0  # 无法拟合：零数据非零预测
        else:
            r2_score = 1 - ss_res / ss_tot

        # 记录求解信息
        logger.debug(f"正则化后条件数: {np.linalg.cond(XTX_reg):.2e}")

        return C, r2_score

    def _validate_elastic_matrix(self, C: np.ndarray) -> None:
        """
        验证弹性常数矩阵的物理合理性

        Parameters
        ----------
        C : numpy.ndarray
            弹性常数矩阵(6,6)

        Notes
        -----
        物理约束检查：

        1. **对角元素正定**：所有模量必须为正
        2. **矩阵对称**：热力学要求Cᵢⱼ = Cⱼᵢ
        3. **矩阵正定**：所有特征值必须为正
        4. **数值稳定**：条件数不能过大

        这些检查有助于发现计算错误或物理不合理的结果。
        """
        logger.debug("验证弹性常数矩阵的物理合理性")

        # 检查对角元素（各向模量必须为正）
        diagonal = np.diag(C)
        if np.any(diagonal <= 0):
            logger.warning("弹性常数对角元素存在非正值")
            logger.warning(f"对角元素: {diagonal}")

        # 检查对称性（热力学互易关系）
        if not np.allclose(C, C.T, rtol=1e-10):
            logger.warning("弹性常数矩阵不对称")
            asymmetry = np.max(np.abs(C - C.T))
            logger.warning(f"最大非对称度: {asymmetry:.2e}")

        # 检查正定性（稳定性要求）
        eigenvalues = np.linalg.eigvals(C)
        if np.any(eigenvalues <= 0):
            logger.warning("弹性常数矩阵不是正定的")
            logger.warning(f"特征值: {eigenvalues}")

        # 检查数值稳定性
        condition_number = np.linalg.cond(C)
        if condition_number > 1e12:
            logger.warning(f"弹性常数矩阵条件数过大: {condition_number:.2e}")

        logger.debug("弹性常数矩阵验证完成")


def calculate_zero_temp_elastic_constants(
    cell: Cell, potential: Potential, delta: float = 0.005, num_steps: int = 5, **kwargs
) -> Tuple[np.ndarray, float]:
    """
    零温弹性常数计算的便捷函数

    这是一个高级接口，封装了完整的零温弹性常数计算流程。
    适合快速计算和脚本使用。

    Parameters
    ----------
    cell : Cell
        晶胞对象
    potential : Potential
        势能函数对象
    delta : float, optional
        应变幅度，默认0.005 (0.5%)
    num_steps : int, optional
        每个应变分量的步数，默认5
    **kwargs
        其他参数传递给ZeroTempDeformationCalculator

    Returns
    -------
    tuple[numpy.ndarray, float]
        弹性常数矩阵(GPa)和拟合优度R²

    Examples
    --------
    >>> from thermoelasticsim.elastic.deformation_method.zero_temp import calculate_zero_temp_elastic_constants
    >>>
    >>> # 快速计算
    >>> C_matrix, r2 = calculate_zero_temp_elastic_constants(cell, potential)
    >>> print(f"弹性常数矩阵(GPa):\\n{C_matrix}")
    >>> print(f"拟合优度: {r2:.6f}")
    >>>
    >>> # 高精度计算
    >>> C_matrix, r2 = calculate_zero_temp_elastic_constants(
    ...     cell, potential, delta=0.001, num_steps=1
    ... )
    """
    # 创建计算器实例
    calculator = ZeroTempDeformationCalculator(
        cell, potential, delta, num_steps, **kwargs
    )

    # 执行计算
    return calculator.calculate()
