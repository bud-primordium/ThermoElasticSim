#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弹性常数统一可视化器

提供统一的弹性常数计算过程可视化接口，支持：
- 从CSV数据或H5轨迹文件加载数据
- 自动识别弹性常数类型(C11/C12/C44等)
- 生成应力-应变响应图表
- 创建交互式HTML仪表板
- 轨迹动画和收敛分析

设计理念：不再特殊化C44，统一处理所有弹性常数

Author: Gilbert Young
Created: 2025-08-15
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import json
from datetime import datetime

from .stress_strain_analyzer import StressStrainAnalyzer, ElasticDataProcessor
from .response_plotter import ResponsePlotter
from ...utils.trajectory import TrajectoryReader

logger = logging.getLogger(__name__)


class ElasticVisualizer:
    """
    弹性常数统一可视化器
    
    提供简洁的API接口，自动处理数据加载、分析和可视化。
    
    Parameters
    ----------
    output_dir : str, optional
        输出目录，默认为当前目录下的'visualization_output'
    dpi : int, optional
        图像分辨率，默认300
    
    Examples
    --------
    # 从CSV数据创建可视化
    >>> visualizer = ElasticVisualizer()
    >>> visualizer.load_csv_data('elastic_data.csv')
    >>> visualizer.generate_dashboard('dashboard.html')
    
    # 从H5轨迹文件创建可视化
    >>> visualizer.load_trajectory('trajectory.h5')
    >>> visualizer.generate_dashboard('dashboard.html')
    
    # 批量处理多个文件
    >>> visualizer.load_multiple_csv(['c11_data.csv', 'c44_data.csv'])
    >>> visualizer.generate_dashboard('combined_dashboard.html')
    """
    
    def __init__(
        self,
        output_dir: str = "visualization_output",
        dpi: int = 300,
        figsize_scale: float = 1.0
    ):
        """
        初始化可视化器
        
        Parameters
        ----------
        output_dir : str
            输出目录路径
        dpi : int
            图像分辨率
        figsize_scale : float
            图像尺寸缩放因子
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.analyzer = StressStrainAnalyzer()
        self.plotter = ResponsePlotter(dpi=dpi, figsize_scale=figsize_scale)
        
        # 数据存储
        self.raw_data = {}  # 原始数据，按弹性常数类型分组
        self.analysis_results = {}  # 分析结果
        self.trajectory_data = None  # 轨迹数据
        self.metadata = {}  # 元数据
        
        logger.info(f"ElasticVisualizer初始化完成，输出目录: {self.output_dir}")
    
    def load_csv_data(
        self, 
        filepath: str, 
        auto_group: bool = True
    ) -> "ElasticVisualizer":
        """
        从CSV文件加载弹性常数数据
        
        Parameters
        ----------
        filepath : str
            CSV文件路径
        auto_group : bool
            是否自动按弹性常数类型分组
            
        Returns
        -------
        ElasticVisualizer
            返回自身，支持链式调用
        """
        logger.info(f"加载CSV数据: {filepath}")
        
        data = ElasticDataProcessor.load_from_csv(filepath)
        
        if auto_group:
            grouped_data = ElasticDataProcessor.group_by_elastic_constant(data)
            self.raw_data.update(grouped_data)
        else:
            # 如果不自动分组，需要手动指定类型
            raise NotImplementedError("需要实现手动分组功能")
        
        # 更新元数据
        self.metadata.update({
            'source_file': filepath,
            'data_points': len(data),
            'elastic_constants': list(self.raw_data.keys()),
            'load_time': datetime.now().isoformat()
        })
        
        logger.info(f"数据加载完成，识别到弹性常数: {list(self.raw_data.keys())}")
        return self
    
    def load_multiple_csv(
        self, 
        filepaths: List[str]
    ) -> "ElasticVisualizer":
        """
        加载多个CSV文件
        
        Parameters
        ----------
        filepaths : List[str]
            CSV文件路径列表
            
        Returns
        -------
        ElasticVisualizer
            返回自身，支持链式调用
        """
        for filepath in filepaths:
            self.load_csv_data(filepath, auto_group=True)
        
        return self
    
    def load_trajectory(
        self, 
        filepath: str
    ) -> "ElasticVisualizer":
        """
        从H5轨迹文件加载数据
        
        Parameters
        ----------
        filepath : str
            H5轨迹文件路径
            
        Returns
        -------
        ElasticVisualizer
            返回自身，支持链式调用
        """
        logger.info(f"加载轨迹文件: {filepath}")
        
        try:
            reader = TrajectoryReader(filepath)
            self.trajectory_data = reader.read_all()
            
            # 从轨迹中提取弹性常数数据
            self._extract_elastic_data_from_trajectory()
            
            self.metadata.update({
                'trajectory_file': filepath,
                'has_trajectory': True,
                'load_time': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"加载轨迹文件失败: {e}")
            raise
        
        return self
    
    def analyze_data(self) -> "ElasticVisualizer":
        """
        分析加载的数据
        
        Returns
        -------
        ElasticVisualizer
            返回自身，支持链式调用
        """
        logger.info("开始数据分析")
        
        if not self.raw_data:
            raise ValueError("未加载任何数据，请先调用load_*方法")
        
        # 对每种弹性常数进行分析
        self.analysis_results = self.analyzer.analyze_multiple_constants(self.raw_data)
        
        # 生成汇总报告
        summary_df = self.analyzer.create_summary_report(self.analysis_results)
        summary_path = self.output_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"数据分析完成，汇总报告保存至: {summary_path}")
        return self
    
    def generate_plots(self) -> Dict[str, str]:
        """
        生成所有可视化图表
        
        Returns
        -------
        Dict[str, str]
            生成的图表文件路径字典
        """
        logger.info("生成可视化图表")
        
        if not self.analysis_results:
            self.analyze_data()
        
        plot_files = {}
        
        # 检查数据类型并生成相应图表
        c11_data = self.raw_data.get('C11', [])
        c12_data = self.raw_data.get('C12', [])
        
        # C11/C12联合图
        if c11_data and c12_data:
            supercell_size = self._infer_supercell_size()
            output_path = self.output_dir / "c11_c12_combined_response.png"
            
            plot_file = self.plotter.plot_c11_c12_combined_response(
                c11_data, c12_data, supercell_size, str(output_path)
            )
            plot_files['c11_c12_combined'] = str(output_path)
        
        # 剪切响应图 (C44/C55/C66)
        shear_constants = ['C44', 'C55', 'C66']
        shear_data = []
        
        for const in shear_constants:
            if const in self.raw_data:
                result = self.analysis_results[const]
                shear_data.append({
                    'direction': f"{const.lower()[:2]}({const})",
                    'strains': [row['applied_strain'] for row in self.raw_data[const]],
                    'stresses': [row['measured_stress_GPa'] for row in self.raw_data[const]],
                    'converged_states': [row['optimization_converged'] for row in self.raw_data[const]],
                    'elastic_constant': result.fit_result.elastic_constant
                })
        
        if shear_data:
            supercell_size = self._infer_supercell_size()
            output_path = self.output_dir / "shear_response.png"
            
            plot_file = self.plotter.plot_shear_response(
                shear_data, supercell_size, str(output_path)
            )
            plot_files['shear_response'] = str(output_path)
        
        logger.info(f"图表生成完成，共{len(plot_files)}个文件")
        return plot_files
    
    def generate_dashboard(
        self, 
        output_path: str,
        include_trajectory: bool = True
    ) -> str:
        """
        生成交互式HTML仪表板
        
        Parameters
        ----------
        output_path : str
            输出HTML文件路径
        include_trajectory : bool
            是否包含轨迹动画（如果有轨迹数据）
            
        Returns
        -------
        str
            生成的HTML文件路径
        """
        logger.info(f"生成HTML仪表板: {output_path}")
        
        # 确保有分析结果
        if not self.analysis_results:
            self.analyze_data()
        
        # 生成图表
        plot_files = self.generate_plots()
        
        # 创建仪表板数据
        dashboard_data = self._prepare_dashboard_data(plot_files)
        
        # 生成HTML
        html_content = self._generate_html_dashboard(dashboard_data, include_trajectory)
        
        # 保存文件
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML仪表板生成完成: {output_file}")
        return str(output_file)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取汇总统计信息
        
        Returns
        -------
        Dict[str, Any]
            统计信息字典
        """
        if not self.analysis_results:
            self.analyze_data()
        
        stats = {
            'elastic_constants': {},
            'overall': {
                'total_constants': len(self.analysis_results),
                'total_data_points': sum(len(data) for data in self.raw_data.values()),
                'average_convergence_rate': 0.0,
                'average_r_squared': 0.0
            }
        }
        
        convergence_rates = []
        r_squared_values = []
        
        for const_type, result in self.analysis_results.items():
            fit = result.fit_result
            
            stats['elastic_constants'][const_type] = {
                'value_GPa': fit.elastic_constant,
                'literature_GPa': result.literature_value,
                'error_percent': result.relative_error,
                'r_squared': fit.r_squared,
                'convergence_rate': fit.convergence_rate,
                'data_quality': result.data_quality
            }
            
            convergence_rates.append(fit.convergence_rate)
            r_squared_values.append(fit.r_squared)
        
        if convergence_rates:
            stats['overall']['average_convergence_rate'] = np.mean(convergence_rates)
            stats['overall']['average_r_squared'] = np.mean(r_squared_values)
        
        return stats
    
    def _extract_elastic_data_from_trajectory(self):
        """从轨迹数据中提取弹性常数计算数据"""
        # TODO: 实现轨迹数据提取逻辑
        # 这将在Phase 2中实现
        logger.warning("轨迹数据提取功能尚未实现")
        pass
    
    def _infer_supercell_size(self) -> Tuple[int, int, int]:
        """从数据中推断超胞尺寸"""
        # 尝试从元数据或数据中推断
        # 简单实现：使用默认值
        return (3, 3, 3)
    
    def _prepare_dashboard_data(self, plot_files: Dict[str, str]) -> Dict[str, Any]:
        """准备仪表板数据"""
        summary_stats = self.get_summary_statistics()
        
        # 准备可序列化的数据
        serializable_data = {
            'metadata': self.metadata,
            'summary_statistics': summary_stats,
            'plot_files': plot_files,
            'analysis_results': self._serialize_analysis_results(),
            'generation_time': datetime.now().isoformat()
        }
        
        return serializable_data
    
    def _serialize_analysis_results(self) -> Dict[str, Any]:
        """将分析结果序列化为JSON兼容格式"""
        serialized = {}
        
        for const_type, result in self.analysis_results.items():
            fit = result.fit_result
            serialized[const_type] = {
                'elastic_constant': fit.elastic_constant,
                'literature_value': result.literature_value,
                'relative_error': result.relative_error,
                'r_squared': fit.r_squared,
                'convergence_rate': fit.convergence_rate,
                'data_quality': result.data_quality,
                'converged_count': fit.converged_count,
                'total_count': fit.total_count
            }
        
        return serialized
    
    def _generate_html_dashboard(
        self, 
        dashboard_data: Dict[str, Any],
        include_trajectory: bool = True
    ) -> str:
        """生成HTML仪表板内容（使用Jinja2模板系统）"""
        from ..web.dashboard_generator import DashboardGenerator
        
        # 使用新的模板系统
        generator = DashboardGenerator()
        return generator.generate_elastic_dashboard(
            dashboard_data,
            template_name='elastic_dashboard.html',
            embed_images=True
        )
    
    # 旧的HTML生成方法已移除，现在使用Jinja2模板系统
    
    def export_data(self, format: str = "csv") -> str:
        """
        导出分析数据
        
        Parameters
        ----------
        format : str
            导出格式 ('csv', 'json')
            
        Returns
        -------
        str
            导出文件路径
        """
        if format == "csv":
            return self.analyzer.export_detailed_analysis(
                self.analysis_results, 
                str(self.output_dir / "detailed_analysis.csv")
            )
        elif format == "json":
            output_path = self.output_dir / "analysis_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._serialize_analysis_results(), f, indent=2, ensure_ascii=False)
            return str(output_path)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        loaded_constants = list(self.raw_data.keys()) if self.raw_data else []
        return f"ElasticVisualizer(constants={loaded_constants}, output_dir='{self.output_dir}')"