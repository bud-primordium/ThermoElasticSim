#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带轨迹记录的优化器扩展

为现有优化器添加轨迹记录功能，在优化过程中自动捕获原子位置、
晶格矢量、能量、应力等信息并保存到HDF5文件。

Author: Gilbert Young
Created: 2025-08-15
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time

from thermoelasticsim.utils.trajectory import TrajectoryWriter
from thermoelasticsim.core.structure import Cell
from thermoelasticsim.potentials import Potential
from thermoelasticsim.utils.optimizers import LBFGSOptimizer

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """
    轨迹记录器
    
    可以作为回调函数集成到优化器中，自动记录优化轨迹。
    
    Parameters
    ----------
    output_file : str
        输出轨迹文件名
    record_interval : int
        记录间隔（每N步记录一次）
    record_forces : bool
        是否记录力
    record_stress : bool
        是否记录应力
    record_energy : bool
        是否记录能量
    compression : str
        压缩算法
    
    Examples
    --------
    >>> recorder = TrajectoryRecorder('optimization.h5')
    >>> optimizer = LBFGSOptimizerWithTrajectory(recorder=recorder)
    >>> optimizer.optimize(cell, potential)
    """
    
    def __init__(
        self,
        output_file: str,
        record_interval: int = 1,
        record_forces: bool = True,
        record_stress: bool = True,
        record_energy: bool = True,
        compression: str = 'gzip',
        compression_opts: int = 4
    ):
        self.output_file = Path(output_file)
        self.record_interval = record_interval
        self.record_forces = record_forces
        self.record_stress = record_stress
        self.record_energy = record_energy
        
        self.writer = TrajectoryWriter(
            str(self.output_file),
            compression=compression,
            compression_opts=compression_opts
        )
        
        self.step_count = 0
        self.start_time = None
        self.initialized = False
        
        logger.info(f"创建轨迹记录器: {output_file}")
        
    def initialize(self, cell: Cell, metadata: Optional[Dict[str, Any]] = None):
        """
        初始化记录器
        
        Parameters
        ----------
        cell : Cell
            初始晶胞
        metadata : dict, optional
            元数据
        """
        n_atoms = len(cell.atoms)
        
        # 获取原子类型
        atom_types = [atom.symbol for atom in cell.atoms]
        
        # 准备元数据
        full_metadata = {
            'optimization_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_atoms': n_atoms,
            'record_interval': self.record_interval,
            'record_forces': self.record_forces,
            'record_stress': self.record_stress,
            'record_energy': self.record_energy,
        }
        
        if metadata:
            full_metadata.update(metadata)
            
        # 初始化HDF5文件
        self.writer.initialize(
            n_atoms=n_atoms,
            n_frames_estimate=1000,
            atom_types=atom_types,
            metadata=full_metadata
        )
        
        self.initialized = True
        self.start_time = time.time()
        
        logger.info(f"轨迹记录器初始化完成: {n_atoms}原子")
        
    def record(self, cell: Cell, potential: Potential, step: Optional[int] = None):
        """
        记录当前状态
        
        Parameters
        ----------
        cell : Cell
            当前晶胞
        potential : Potential
            势能函数
        step : int, optional
            优化步数
        """
        if not self.initialized:
            self.initialize(cell)
            
        # 检查记录间隔
        if self.step_count % self.record_interval != 0:
            self.step_count += 1
            return
            
        # 获取数据
        positions = cell.get_positions()
        box = cell.lattice_vectors
        
        # 计算时间（秒转换为皮秒）
        current_time = (time.time() - self.start_time) * 1e9  # 秒转皮秒
        
        # 准备额外数据
        kwargs = {}
        
        if self.record_forces:
            forces = cell.get_forces()
            kwargs['forces'] = forces
            
        if self.record_stress:
            stress = cell.calculate_stress_tensor(potential)
            kwargs['stress'] = stress
            
        if self.record_energy:
            energy = potential.calculate_energy(cell)
            kwargs['energy'] = energy
            kwargs['energy_per_atom'] = energy / len(cell.atoms)
            
        # 计算体积
        kwargs['volume'] = cell.calculate_volume()
        
        # 写入帧
        self.writer.write_frame(
            positions=positions,
            box=box,
            time=current_time,
            step=step if step is not None else self.step_count,
            **kwargs
        )
        
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            logger.debug(f"已记录{self.step_count}帧")
            
    def finalize(self):
        """完成记录并关闭文件"""
        if self.writer:
            # 写入最终元数据
            end_metadata = {
                'optimization_end': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_frames': self.writer.n_frames,
                'total_time': time.time() - self.start_time if self.start_time else 0,
            }
            
            self.writer.write_metadata(end_metadata)
            self.writer.close()
            
            logger.info(f"轨迹记录完成: {self.writer.n_frames}帧")
            

class LBFGSOptimizerWithTrajectory(LBFGSOptimizer):
    """
    带轨迹记录功能的L-BFGS优化器
    
    在原有L-BFGS优化器基础上添加轨迹记录功能。
    
    Parameters
    ----------
    recorder : TrajectoryRecorder, optional
        轨迹记录器
    trajectory_file : str, optional
        轨迹文件名（如果不提供recorder）
    record_interval : int
        记录间隔
    **kwargs
        其他L-BFGS参数
    """
    
    def __init__(
        self,
        recorder: Optional[TrajectoryRecorder] = None,
        trajectory_file: Optional[str] = None,
        record_interval: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 设置记录器
        if recorder is not None:
            self.recorder = recorder
        elif trajectory_file is not None:
            self.recorder = TrajectoryRecorder(
                trajectory_file,
                record_interval=record_interval
            )
        else:
            self.recorder = None
            
    def optimize(self, cell: Cell, potential: Potential, relax_cell: bool = False):
        """
        执行优化并记录轨迹
        
        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能函数
        relax_cell : bool
            是否优化晶格
            
        Returns
        -------
        tuple
            (是否收敛, 优化信息字典)
        """
        # 初始化记录器
        if self.recorder:
            metadata = {
                'optimizer': 'L-BFGS',
                'relax_cell': relax_cell,
                'ftol': self.ftol,
                'gtol': self.gtol,
                'maxiter': self.maxiter,
            }
            self.recorder.initialize(cell, metadata)
            
            # 记录初始状态
            self.recorder.record(cell, potential, step=0)
            
        # 创建回调函数
        original_callback = getattr(self, '_callback', None)
        
        def trajectory_callback(xk):
            """优化过程中的回调函数"""
            if self.recorder:
                # 更新晶胞状态（xk包含了当前的优化变量）
                # 这里需要根据relax_cell来解析xk
                self.recorder.record(cell, potential)
                
            if original_callback:
                return original_callback(xk)
                
        # 临时替换回调函数
        self._callback = trajectory_callback
        
        try:
            # 执行原始优化
            result = super().optimize(cell, potential, relax_cell)
            
            # 记录最终状态
            if self.recorder:
                self.recorder.record(cell, potential)
                self.recorder.finalize()
                
            return result
            
        finally:
            # 恢复原始回调函数
            self._callback = original_callback
            

def create_optimizer_with_trajectory(
    optimizer_type: str = 'L-BFGS',
    trajectory_file: Optional[str] = None,
    record_interval: int = 1,
    **optimizer_params
):
    """
    创建带轨迹记录的优化器
    
    Parameters
    ----------
    optimizer_type : str
        优化器类型
    trajectory_file : str, optional
        轨迹文件名
    record_interval : int
        记录间隔
    **optimizer_params
        优化器参数
        
    Returns
    -------
    optimizer
        带轨迹记录的优化器实例
    """
    if trajectory_file:
        recorder = TrajectoryRecorder(
            trajectory_file,
            record_interval=record_interval
        )
    else:
        recorder = None
        
    if optimizer_type == 'L-BFGS':
        return LBFGSOptimizerWithTrajectory(
            recorder=recorder,
            **optimizer_params
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        

# 用于形变计算的轨迹记录
class DeformationTrajectoryRecorder:
    """
    专门用于形变计算的轨迹记录器
    
    记录形变过程中的完整信息，包括应变、应力、能量等。
    """
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.writer = TrajectoryWriter(str(self.output_file))
        self.deformation_data = []
        self.initialized = False
        
    def initialize(self, base_cell: Cell, metadata: Optional[Dict[str, Any]] = None):
        """初始化"""
        n_atoms = len(base_cell.atoms)
        atom_types = [atom.symbol for atom in base_cell.atoms]
        
        full_metadata = {
            'calculation_type': 'deformation',
            'n_atoms': n_atoms,
        }
        
        if metadata:
            full_metadata.update(metadata)
            
        self.writer.initialize(
            n_atoms=n_atoms,
            n_frames_estimate=100,
            atom_types=atom_types,
            metadata=full_metadata
        )
        
        self.initialized = True
        
    def record_deformation(
        self,
        cell: Cell,
        potential: Potential,
        strain: np.ndarray,
        stress: np.ndarray,
        deformation_matrix: np.ndarray,
        mode: str,
        converged: bool
    ):
        """
        记录形变状态
        
        Parameters
        ----------
        cell : Cell
            形变后的晶胞
        potential : Potential
            势能函数
        strain : np.ndarray
            应变张量
        stress : np.ndarray
            应力张量
        deformation_matrix : np.ndarray
            形变矩阵
        mode : str
            形变模式
        converged : bool
            是否收敛
        """
        if not self.initialized:
            raise RuntimeError("必须先初始化记录器")
            
        positions = cell.get_positions()
        box = cell.lattice_vectors
        forces = cell.get_forces()
        energy = potential.calculate_energy(cell)
        
        # 记录形变特定数据
        self.writer.write_frame(
            positions=positions,
            box=box,
            time=len(self.deformation_data),  # 使用索引作为时间
            step=len(self.deformation_data),
            forces=forces,
            stress=stress,
            energy=energy,
            strain_tensor=strain,
            deformation_matrix=deformation_matrix,
            deformation_mode=mode,
            converged=converged,
            volume=cell.calculate_volume()
        )
        
        # 保存到内部列表
        self.deformation_data.append({
            'strain': strain.copy(),
            'stress': stress.copy(),
            'energy': energy,
            'converged': converged,
            'mode': mode,
        })
        
    def finalize(self):
        """完成记录"""
        if self.writer:
            # 写入汇总信息
            metadata = {
                'total_deformations': len(self.deformation_data),
                'converged_count': sum(1 for d in self.deformation_data if d['converged']),
            }
            
            self.writer.write_metadata(metadata)
            self.writer.close()
            
            logger.info(f"形变轨迹记录完成: {len(self.deformation_data)}个形变")