#!/usr/bin/env python3
"""
弹性常数轨迹记录器

专门用于记录弹性常数计算过程的轨迹数据，包括：
- 形变过程的原子位置演化
- 应力-应变数据点
- 优化收敛轨迹
- 弹性常数计算元数据

Author: Gilbert Young
Created: 2025-08-15
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from thermoelasticsim.utils.trajectory import TrajectoryWriter

logger = logging.getLogger(__name__)


class ElasticTrajectoryRecorder:
    """
    弹性常数计算轨迹记录器
    
    专门记录弹性常数计算过程中的所有相关数据：
    - 每个形变步骤的系统状态
    - 优化过程的收敛轨迹
    - 应力-应变关系数据
    - 弹性常数计算的元数据
    
    Parameters
    ----------
    output_path : str
        输出H5文件路径
    elastic_type : str
        弹性常数类型 (C11, C12, C44等)
    calculation_method : str
        计算方法 (uniaxial, shear等)
    
    Examples
    --------
    >>> recorder = ElasticTrajectoryRecorder('c44_trajectory.h5', 'C44', 'shear')
    >>> recorder.initialize(cell, potential, strain_points)
    >>> 
    >>> for strain in strain_points:
    >>>     recorder.record_deformation_step(cell, strain, 'before_relax')
    >>>     # ... 优化过程 ...
    >>>     recorder.record_deformation_step(cell, strain, 'after_relax')
    >>> 
    >>> recorder.finalize()
    """

    def __init__(
        self,
        output_path: str,
        elastic_type: str,
        calculation_method: str = 'default',
        supercell_size: tuple[int, int, int] = (3, 3, 3)
    ):
        """
        初始化记录器
        
        Parameters
        ----------
        output_path : str
            输出文件路径
        elastic_type : str
            弹性常数类型
        calculation_method : str
            计算方法
        supercell_size : tuple
            超胞尺寸
        """
        self.output_path = Path(output_path)
        self.elastic_type = elastic_type
        self.calculation_method = calculation_method
        self.supercell_size = supercell_size

        # 创建输出目录
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 轨迹写入器
        self.trajectory_writer = None

        # 数据存储
        self.deformation_steps = []
        self.optimization_history = []
        self.stress_strain_data = []
        self.metadata = {
            'elastic_type': elastic_type,
            'calculation_method': calculation_method,
            'supercell_size': supercell_size,
            'created': datetime.now().isoformat(),
            'version': '1.0'
        }

        # 状态跟踪
        self.initialized = False
        self.current_strain = 0.0
        self.current_step = 0
        self.step_counter = 0

        logger.info(f"创建弹性轨迹记录器: {elastic_type}, 输出: {output_path}")

    def initialize(
        self,
        cell,  # Cell对象
        potential,  # Potential对象
        strain_points: list[float],
        additional_metadata: dict | None = None
    ):
        """
        初始化记录器和H5文件结构
        
        Parameters
        ----------
        cell : Cell
            初始晶胞
        potential : Potential
            势能函数
        strain_points : List[float]
            应变点列表
        additional_metadata : dict, optional
            额外的元数据
        """
        self.trajectory_writer = TrajectoryWriter(
            str(self.output_path),
            mode='w',
            compression='gzip',
            chunk_size=50  # 较小的块尺寸适合弹性常数计算
        )

        # 准备元数据
        n_atoms = cell.num_atoms
        atom_types = [atom.symbol for atom in cell.atoms]

        # 扩展元数据
        self.metadata.update({
            'n_atoms': n_atoms,
            'atom_types': list(set(atom_types)),
            'strain_points': strain_points,
            'strain_range': [min(strain_points), max(strain_points)],
            'n_strain_points': len(strain_points),
            'potential_type': type(potential).__name__,
            'initial_lattice': cell.lattice_vectors.tolist(),
            'initial_volume': float(cell.volume)
        })

        if additional_metadata:
            self.metadata.update(additional_metadata)

        # 初始化轨迹写入器
        self.trajectory_writer.initialize(
            n_atoms=n_atoms,
            n_frames_estimate=len(strain_points) * 50,  # 预估帧数
            atom_types=atom_types,
            metadata=self.metadata
        )

        # 创建弹性常数专用组
        self._setup_elastic_groups()

        self.initialized = True
        logger.info(f"弹性轨迹记录器初始化完成: {n_atoms}原子, {len(strain_points)}应变点")

    def _setup_elastic_groups(self):
        """设置弹性常数专用的HDF5组"""
        file = self.trajectory_writer.file

        # 创建弹性常数专用组
        elastic_group = file.create_group('elastic_constants')
        elastic_group.attrs['type'] = self.elastic_type
        elastic_group.attrs['method'] = self.calculation_method

        # 形变步骤组
        deformation_group = elastic_group.create_group('deformation_steps')

        # 优化轨迹组
        optimization_group = elastic_group.create_group('optimization')

        # 应力-应变数据组
        stress_strain_group = elastic_group.create_group('stress_strain')

        self.elastic_group = elastic_group
        self.deformation_group = deformation_group
        self.optimization_group = optimization_group
        self.stress_strain_group = stress_strain_group

    def record_deformation_step(
        self,
        cell,  # Cell对象
        strain: float,
        step_type: str,
        stress_tensor: np.ndarray | None = None,
        energy: float | None = None,
        converged: bool = True,
        additional_data: dict | None = None
    ):
        """
        记录单个形变步骤
        
        Parameters
        ----------
        cell : Cell
            当前晶胞状态
        strain : float
            应变值
        step_type : str
            步骤类型 (base_state, before_internal_relax, after_internal_relax等)
        stress_tensor : np.ndarray, optional
            应力张量 (3x3)
        energy : float, optional
            总能量
        converged : bool
            优化是否收敛
        additional_data : dict, optional
            额外数据
        """
        if not self.initialized:
            raise RuntimeError("记录器未初始化，请先调用initialize()")

        positions = cell.get_positions()
        lattice = cell.lattice_vectors
        volume = cell.volume

        # 计算晶格常数信息
        lattice_a = np.linalg.norm(lattice[0])
        lattice_b = np.linalg.norm(lattice[1])
        lattice_c = np.linalg.norm(lattice[2])

        # 计算晶格角度
        cos_alpha = np.dot(lattice[1], lattice[2]) / (lattice_b * lattice_c)
        cos_beta = np.dot(lattice[0], lattice[2]) / (lattice_a * lattice_c)
        cos_gamma = np.dot(lattice[0], lattice[1]) / (lattice_a * lattice_b)

        alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
        beta = np.degrees(np.arccos(np.clip(cos_beta, -1, 1)))
        gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1, 1)))

        # 增强的描述信息 - 根据步骤类型调整详细程度
        if step_type == "base_state":
            # 基态：简洁显示
            description = f"base_state | E={energy:.3f}eV | a={lattice_a:.3f}A | V={volume:.1f}A³ | cubic"
        else:
            # 形变态：显示完整信息
            converged_symbol = "✓" if converged else "✗"
            # 判断是否为立方（所有角度接近90度）
            is_cubic = abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1

            if is_cubic:
                # 接近立方，简化显示
                description = (
                    f"{step_type} | strain={strain:.4f} | E={energy:.3f}eV | "
                    f"a={lattice_a:.3f}A | V={volume:.1f}A³ | {converged_symbol}"
                )
            else:
                # 三斜晶胞，显示关键角度变化和完整晶格参数
                description = (
                    f"{step_type} | strain={strain:.4f} | E={energy:.3f}eV | "
                    f"a={lattice_a:.3f}A b={lattice_b:.3f}A c={lattice_c:.3f}A | "
                    f"α={alpha:.1f}deg β={beta:.1f}deg γ={gamma:.1f}deg | V={volume:.1f}A³ | {converged_symbol}"
                )

        # 记录到轨迹文件
        self.trajectory_writer.write_frame(
            positions=positions,
            box=lattice,
            time=self.step_counter * 0.001,  # 伪时间，以ps为单位
            step=self.step_counter,
            stress=stress_tensor,
            energy=energy,
            strain_value=strain,
            step_type=step_type,
            converged=converged,
            volume=volume,
            lattice_a=lattice_a,
            lattice_b=lattice_b,
            lattice_c=lattice_c,
            lattice_alpha=alpha,
            lattice_beta=beta,
            lattice_gamma=gamma,
            description=description
        )

        # 记录形变步骤数据
        step_data = {
            'step_id': self.step_counter,
            'strain': strain,
            'step_type': step_type,
            'positions': positions.copy(),
            'lattice': lattice.copy(),
            'volume': volume,
            'energy': energy,
            'converged': converged,
            'timestamp': datetime.now().isoformat()
        }

        if stress_tensor is not None:
            step_data['stress_tensor'] = stress_tensor.copy()

        if additional_data:
            step_data.update(additional_data)

        self.deformation_steps.append(step_data)

        # 如果是应力-应变数据点，记录到专用数组
        if step_type in ['after_relax', 'final'] and stress_tensor is not None:
            self._record_stress_strain_point(strain, stress_tensor, energy, converged)

        self.step_counter += 1
        logger.debug(f"记录形变步骤: 应变={strain:.4f}, 类型={step_type}, 收敛={converged}")

    def record_optimization_step(
        self,
        cell,  # Cell对象
        iteration: int,
        energy: float,
        forces: np.ndarray,
        gradient_norm: float,
        converged: bool = False
    ):
        """
        记录优化过程中的单步
        
        Parameters
        ----------
        cell : Cell
            当前系统状态
        iteration : int
            迭代次数
        energy : float
            当前能量
        forces : np.ndarray
            原子力
        gradient_norm : float
            梯度范数
        converged : bool
            是否已收敛
        """
        positions = cell.get_positions()

        # 记录到轨迹（以更高频率）
        self.trajectory_writer.write_frame(
            positions=positions,
            box=cell.lattice_vectors,
            time=self.step_counter * 0.001,
            step=self.step_counter,
            forces=forces,
            energy=energy,
            strain_value=self.current_strain,
            step_type='optimization',
            iteration=iteration,
            gradient_norm=gradient_norm,
            converged=converged
        )

        # 记录优化历史
        opt_data = {
            'iteration': iteration,
            'energy': energy,
            'gradient_norm': gradient_norm,
            'converged': converged,
            'strain': self.current_strain,
            'timestamp': datetime.now().isoformat()
        }

        self.optimization_history.append(opt_data)
        self.step_counter += 1

    def _record_stress_strain_point(
        self,
        strain: float,
        stress_tensor: np.ndarray,
        energy: float | None,
        converged: bool
    ):
        """记录应力-应变数据点"""
        # 根据弹性常数类型提取相关应力分量
        relevant_stress = self._extract_relevant_stress(stress_tensor)

        stress_strain_point = {
            'strain': strain,
            'stress_tensor': stress_tensor.copy(),
            'relevant_stress': relevant_stress,
            'energy': energy,
            'converged': converged,
            'elastic_type': self.elastic_type
        }

        self.stress_strain_data.append(stress_strain_point)

    def _extract_relevant_stress(self, stress_tensor: np.ndarray) -> float:
        """根据弹性常数类型提取相关应力分量"""
        if self.elastic_type == 'C11':
            return stress_tensor[0, 0]  # σxx
        elif self.elastic_type == 'C12':
            return stress_tensor[1, 1]  # σyy (对于xx应变)
        elif self.elastic_type == 'C44':
            return stress_tensor[1, 2]  # σyz
        elif self.elastic_type == 'C55':
            return stress_tensor[0, 2]  # σxz
        elif self.elastic_type == 'C66':
            return stress_tensor[0, 1]  # σxy
        else:
            # 默认返回最大应力分量
            return np.max(np.abs(stress_tensor))

    def set_current_strain(self, strain: float):
        """设置当前应变值"""
        self.current_strain = strain

    def save_analysis_data(self):
        """保存分析数据到HDF5文件"""
        if not self.stress_strain_data:
            logger.warning("无应力-应变数据可保存")
            return

        # 准备数组数据
        strains = np.array([point['strain'] for point in self.stress_strain_data])
        relevant_stresses = np.array([point['relevant_stress'] for point in self.stress_strain_data])
        energies = np.array([point['energy'] or 0.0 for point in self.stress_strain_data])
        converged_flags = np.array([point['converged'] for point in self.stress_strain_data])

        # 保存到HDF5
        self.stress_strain_group.create_dataset('strains', data=strains)
        self.stress_strain_group.create_dataset('stresses', data=relevant_stresses)
        self.stress_strain_group.create_dataset('energies', data=energies)
        self.stress_strain_group.create_dataset('converged', data=converged_flags)

        # 完整的应力张量
        if self.stress_strain_data:
            stress_tensors = np.array([point['stress_tensor'] for point in self.stress_strain_data])
            self.stress_strain_group.create_dataset('stress_tensors', data=stress_tensors)

        # 线性拟合
        converged_mask = converged_flags
        if np.sum(converged_mask) >= 2:
            conv_strains = strains[converged_mask]
            conv_stresses = relevant_stresses[converged_mask]

            # 线性拟合
            coeffs = np.polyfit(conv_strains, conv_stresses, 1)
            elastic_constant = coeffs[0]

            # 计算R²
            y_pred = np.polyval(coeffs, conv_strains)
            ss_res = np.sum((conv_stresses - y_pred) ** 2)
            ss_tot = np.sum((conv_stresses - np.mean(conv_stresses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 保存拟合结果
            fit_group = self.stress_strain_group.create_group('linear_fit')
            fit_group.attrs['elastic_constant'] = elastic_constant
            fit_group.attrs['r_squared'] = r_squared
            fit_group.attrs['slope'] = coeffs[0]
            fit_group.attrs['intercept'] = coeffs[1]
            fit_group.attrs['converged_points'] = np.sum(converged_mask)
            fit_group.attrs['total_points'] = len(strains)

            logger.info(f"保存拟合结果: {self.elastic_type} = {elastic_constant:.2f} GPa, R² = {r_squared:.4f}")

    def finalize(self) -> str:
        """
        完成记录并关闭文件
        
        Returns
        -------
        str
            输出文件路径
        """
        if not self.initialized:
            logger.warning("记录器未初始化，无法完成")
            return str(self.output_path)

        # 保存分析数据
        self.save_analysis_data()

        # 保存汇总元数据
        self.metadata.update({
            'finalized': datetime.now().isoformat(),
            'total_frames': self.step_counter,
            'deformation_steps': len(self.deformation_steps),
            'optimization_steps': len(self.optimization_history),
            'stress_strain_points': len(self.stress_strain_data)
        })

        self.trajectory_writer.write_metadata(self.metadata)

        # 关闭文件
        self.trajectory_writer.close()

        logger.info(f"弹性轨迹记录完成: {self.output_path}")
        logger.info(f"总帧数: {self.step_counter}, 形变步骤: {len(self.deformation_steps)}")

        return str(self.output_path)

    def get_summary(self) -> dict[str, Any]:
        """获取记录汇总信息"""
        return {
            'elastic_type': self.elastic_type,
            'calculation_method': self.calculation_method,
            'output_path': str(self.output_path),
            'total_frames': self.step_counter,
            'deformation_steps': len(self.deformation_steps),
            'optimization_steps': len(self.optimization_history),
            'stress_strain_points': len(self.stress_strain_data),
            'initialized': self.initialized
        }

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.initialized:
            self.finalize()


class ElasticTrajectoryManager:
    """
    弹性常数轨迹管理器
    
    管理多个弹性常数的轨迹记录，提供统一接口。
    
    Examples
    --------
    >>> manager = ElasticTrajectoryManager('elastic_trajectories/')
    >>> manager.add_recorder('C11', 'uniaxial', (3, 3, 3))
    >>> manager.add_recorder('C44', 'shear', (3, 3, 3))
    >>> 
    >>> # 在计算过程中记录
    >>> manager.record_all('C11', cell, strain, 'after_relax', stress_tensor)
    >>> 
    >>> manager.finalize_all()
    """

    def __init__(self, output_dir: str):
        """
        初始化管理器
        
        Parameters
        ----------
        output_dir : str
            输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recorders = {}  # {elastic_type: ElasticTrajectoryRecorder}

        logger.info(f"创建弹性轨迹管理器: {output_dir}")

    def add_recorder(
        self,
        elastic_type: str,
        calculation_method: str,
        supercell_size: tuple[int, int, int] = (3, 3, 3)
    ) -> ElasticTrajectoryRecorder:
        """
        添加轨迹记录器
        
        Parameters
        ----------
        elastic_type : str
            弹性常数类型
        calculation_method : str
            计算方法
        supercell_size : tuple
            超胞尺寸
            
        Returns
        -------
        ElasticTrajectoryRecorder
            记录器对象
        """
        output_path = self.output_dir / f"{elastic_type.lower()}_trajectory.h5"

        recorder = ElasticTrajectoryRecorder(
            str(output_path),
            elastic_type,
            calculation_method,
            supercell_size
        )

        self.recorders[elastic_type] = recorder

        logger.info(f"添加{elastic_type}轨迹记录器")
        return recorder

    def initialize_recorder(
        self,
        elastic_type: str,
        cell,  # Cell对象
        potential,  # Potential对象
        strain_points: list[float],
        additional_metadata: dict | None = None
    ):
        """初始化指定的记录器"""
        if elastic_type not in self.recorders:
            raise ValueError(f"未找到{elastic_type}记录器")

        self.recorders[elastic_type].initialize(
            cell, potential, strain_points, additional_metadata
        )

    def record_all(
        self,
        elastic_type: str,
        cell,  # Cell对象
        strain: float,
        step_type: str,
        stress_tensor: np.ndarray | None = None,
        **kwargs
    ):
        """记录到指定的记录器"""
        if elastic_type in self.recorders:
            self.recorders[elastic_type].record_deformation_step(
                cell, strain, step_type, stress_tensor, **kwargs
            )

    def finalize_all(self) -> dict[str, str]:
        """完成所有记录器"""
        results = {}

        for elastic_type, recorder in self.recorders.items():
            try:
                output_path = recorder.finalize()
                results[elastic_type] = output_path
                logger.info(f"{elastic_type}轨迹记录完成: {output_path}")
            except Exception as e:
                logger.error(f"{elastic_type}轨迹记录失败: {e}")
                results[elastic_type] = None

        return results

    def get_all_summaries(self) -> dict[str, dict]:
        """获取所有记录器的汇总信息"""
        return {
            elastic_type: recorder.get_summary()
            for elastic_type, recorder in self.recorders.items()
        }
