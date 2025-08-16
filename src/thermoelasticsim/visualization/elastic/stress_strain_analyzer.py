#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应力-应变数据分析器

提供弹性常数计算过程中的数据分析功能，包括：
- 应力-应变数据处理
- 线性拟合和统计分析
- 收敛性评估
- 误差分析

Author: Gilbert Young
Created: 2025-08-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitResult:
    """
    拟合结果数据容器
    
    Attributes
    ----------
    elastic_constant : float
        拟合得到的弹性常数值 (GPa)
    r_squared : float
        拟合的R²值
    slope : float
        拟合直线斜率
    intercept : float
        拟合直线截距
    std_error : float
        标准误差
    converged_count : int
        收敛点数量
    total_count : int
        总点数
    convergence_rate : float
        收敛率
    """
    elastic_constant: float
    r_squared: float
    slope: float
    intercept: float
    std_error: float
    converged_count: int
    total_count: int
    convergence_rate: float


@dataclass
class ElasticAnalysisResult:
    """
    弹性常数分析结果
    
    Attributes
    ----------
    constant_type : str
        弹性常数类型 (C11, C12, C44等)
    fit_result : FitResult
        拟合结果
    literature_value : float
        文献值 (GPa)
    relative_error : float
        相对误差 (%)
    data_quality : str
        数据质量评级 (Excellent/Good/Fair/Poor)
    raw_data : List[Dict]
        原始数据点
    """
    constant_type: str
    fit_result: FitResult
    literature_value: float
    relative_error: float
    data_quality: str
    raw_data: List[Dict]


class StressStrainAnalyzer:
    """
    应力-应变数据分析器
    
    提供统一的数据分析接口，支持所有类型的弹性常数计算。
    
    Parameters
    ----------
    literature_values : Dict[str, float], optional
        文献值字典，键为弹性常数名称，值为文献值(GPa)
    
    Examples
    --------
    >>> analyzer = StressStrainAnalyzer()
    >>> result = analyzer.analyze_elastic_constant('C11', c11_data)
    >>> print(f"C11 = {result.fit_result.elastic_constant:.1f} GPa")
    """
    
    # 默认文献值 (GPa)
    DEFAULT_LITERATURE_VALUES = {
        'C11': 110.0,
        'C12': 61.0,
        'C44': 33.0,
        'C55': 33.0,
        'C66': 33.0,
        'bulk_modulus': 77.3,  # (C11 + 2*C12)/3
        'shear_modulus': 33.0   # C44
    }
    
    def __init__(self, literature_values: Optional[Dict[str, float]] = None):
        """
        初始化分析器
        
        Parameters
        ----------
        literature_values : Dict[str, float], optional
            自定义文献值，如果不提供则使用默认值
        """
        self.literature_values = literature_values or self.DEFAULT_LITERATURE_VALUES
        
    def analyze_elastic_constant(
        self, 
        constant_type: str, 
        data: List[Dict],
        strain_key: str = "applied_strain",
        stress_key: str = "measured_stress_GPa", 
        converged_key: str = "optimization_converged"
    ) -> ElasticAnalysisResult:
        """
        分析单个弹性常数
        
        Parameters
        ----------
        constant_type : str
            弹性常数类型 (C11, C12, C44等)
        data : List[Dict]
            数据点列表，每个字典包含应变、应力、收敛状态等信息
        strain_key : str
            应变数据的键名
        stress_key : str
            应力数据的键名
        converged_key : str
            收敛状态的键名
            
        Returns
        -------
        ElasticAnalysisResult
            分析结果
        """
        logger.info(f"开始分析{constant_type}弹性常数，数据点数: {len(data)}")
        
        # 提取数据
        strains = [row[strain_key] for row in data]
        stresses = [row[stress_key] for row in data]
        converged_states = [row[converged_key] for row in data]
        
        # 过滤收敛点
        converged_strains = [s for s, c in zip(strains, converged_states) if c]
        converged_stresses = [st for st, c in zip(stresses, converged_states) if c]
        
        # 拟合分析
        fit_result = self._perform_linear_fit(
            converged_strains, converged_stresses, 
            len(converged_states), sum(converged_states)
        )
        
        # 获取文献值和计算误差
        literature_value = self.literature_values.get(constant_type, 0.0)
        relative_error = self._calculate_relative_error(
            fit_result.elastic_constant, literature_value
        )
        
        # 评估数据质量
        data_quality = self._assess_data_quality(fit_result, relative_error)
        
        result = ElasticAnalysisResult(
            constant_type=constant_type,
            fit_result=fit_result,
            literature_value=literature_value,
            relative_error=relative_error,
            data_quality=data_quality,
            raw_data=data
        )
        
        logger.info(
            f"{constant_type}分析完成: "
            f"值={fit_result.elastic_constant:.1f} GPa, "
            f"误差={relative_error:+.1f}%, "
            f"R²={fit_result.r_squared:.4f}, "
            f"收敛率={fit_result.convergence_rate:.1%}"
        )
        
        return result
    
    def analyze_multiple_constants(
        self, 
        data_dict: Dict[str, List[Dict]],
        **kwargs
    ) -> Dict[str, ElasticAnalysisResult]:
        """
        分析多个弹性常数
        
        Parameters
        ----------
        data_dict : Dict[str, List[Dict]]
            数据字典，键为弹性常数名称，值为数据点列表
        **kwargs
            传递给analyze_elastic_constant的额外参数
            
        Returns
        -------
        Dict[str, ElasticAnalysisResult]
            分析结果字典
        """
        results = {}
        
        for constant_type, data in data_dict.items():
            try:
                results[constant_type] = self.analyze_elastic_constant(
                    constant_type, data, **kwargs
                )
            except Exception as e:
                logger.error(f"分析{constant_type}时出错: {e}")
                
        return results
    
    def create_summary_report(
        self, 
        results: Dict[str, ElasticAnalysisResult]
    ) -> pd.DataFrame:
        """
        创建分析汇总报告
        
        Parameters
        ----------
        results : Dict[str, ElasticAnalysisResult]
            分析结果字典
            
        Returns
        -------
        pd.DataFrame
            汇总报告表格
        """
        summary_data = []
        
        for constant_type, result in results.items():
            fit = result.fit_result
            summary_data.append({
                'constant': constant_type,
                'value_GPa': fit.elastic_constant,
                'literature_GPa': result.literature_value,
                'error_percent': result.relative_error,
                'r_squared': fit.r_squared,
                'convergence_rate': fit.convergence_rate,
                'converged_points': f"{fit.converged_count}/{fit.total_count}",
                'data_quality': result.data_quality,
                'std_error': fit.std_error
            })
            
        return pd.DataFrame(summary_data)
    
    def _perform_linear_fit(
        self, 
        strains: List[float], 
        stresses: List[float],
        total_count: int,
        converged_count: int
    ) -> FitResult:
        """执行线性拟合"""
        if len(strains) < 2:
            logger.warning("收敛点不足，无法进行线性拟合")
            return FitResult(
                elastic_constant=0.0,
                r_squared=0.0,
                slope=0.0,
                intercept=0.0,
                std_error=np.inf,
                converged_count=converged_count,
                total_count=total_count,
                convergence_rate=converged_count / total_count if total_count > 0 else 0.0
            )
        
        # 线性拟合
        coeffs = np.polyfit(strains, stresses, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        # 计算R²
        y_pred = np.polyval(coeffs, strains)
        ss_res = np.sum((stresses - y_pred) ** 2)
        ss_tot = np.sum((stresses - np.mean(stresses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
        
        # 计算标准误差
        if len(strains) > 2:
            residuals = stresses - y_pred
            std_error = np.sqrt(np.sum(residuals**2) / (len(strains) - 2))
        else:
            std_error = 0.0
            
        return FitResult(
            elastic_constant=slope,  # 斜率即为弹性常数
            r_squared=r_squared,
            slope=slope,
            intercept=intercept,
            std_error=std_error,
            converged_count=converged_count,
            total_count=total_count,
            convergence_rate=converged_count / total_count if total_count > 0 else 0.0
        )
    
    def _calculate_relative_error(
        self, 
        calculated_value: float, 
        literature_value: float
    ) -> float:
        """计算相对误差"""
        if literature_value == 0:
            return float('inf')
        return (calculated_value - literature_value) / literature_value * 100
    
    def _assess_data_quality(
        self, 
        fit_result: FitResult, 
        relative_error: float
    ) -> str:
        """评估数据质量"""
        # 综合考虑多个因素
        r_squared_good = fit_result.r_squared >= 0.95
        convergence_good = fit_result.convergence_rate >= 0.8
        error_good = abs(relative_error) <= 10
        error_fair = abs(relative_error) <= 25
        
        if r_squared_good and convergence_good and error_good:
            return "Excellent"
        elif fit_result.r_squared >= 0.9 and convergence_good and error_fair:
            return "Good"
        elif fit_result.r_squared >= 0.8 and fit_result.convergence_rate >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def export_detailed_analysis(
        self, 
        results: Dict[str, ElasticAnalysisResult], 
        output_path: str
    ) -> str:
        """
        导出详细分析结果到CSV
        
        Parameters
        ----------
        results : Dict[str, ElasticAnalysisResult]
            分析结果字典
        output_path : str
            输出文件路径
            
        Returns
        -------
        str
            输出文件路径
        """
        all_data = []
        
        for constant_type, result in results.items():
            for row in result.raw_data:
                # 添加分析结果信息
                enhanced_row = row.copy()
                enhanced_row.update({
                    'elastic_constant_type': constant_type,
                    'fitted_elastic_constant_GPa': result.fit_result.elastic_constant,
                    'literature_value_GPa': result.literature_value,
                    'relative_error_percent': result.relative_error,
                    'r_squared': result.fit_result.r_squared,
                    'data_quality': result.data_quality,
                    'convergence_rate': result.fit_result.convergence_rate
                })
                all_data.append(enhanced_row)
        
        # 保存到CSV
        df = pd.DataFrame(all_data)
        filepath = Path(output_path)
        df.to_csv(filepath, index=False)
        
        logger.info(f"详细分析结果已导出: {filepath}")
        return str(filepath)


class ElasticDataProcessor:
    """
    弹性常数数据预处理器
    
    提供数据清洗、格式转换、异常值检测等功能。
    """
    
    @staticmethod
    def load_from_csv(filepath: str) -> List[Dict]:
        """
        从CSV文件加载数据
        
        Parameters
        ----------
        filepath : str
            CSV文件路径
            
        Returns
        -------
        List[Dict]
            数据点列表
        """
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    
    @staticmethod
    def group_by_elastic_constant(data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按弹性常数类型分组数据
        
        Parameters
        ----------
        data : List[Dict]
            数据点列表
            
        Returns
        -------
        Dict[str, List[Dict]]
            分组后的数据字典
        """
        groups = {}
        
        for row in data:
            # 根据计算方法或其他字段确定弹性常数类型
            constant_type = ElasticDataProcessor._identify_constant_type(row)
            
            if constant_type not in groups:
                groups[constant_type] = []
            groups[constant_type].append(row)
            
        return groups
    
    @staticmethod
    def _identify_constant_type(row: Dict) -> str:
        """根据数据行识别弹性常数类型"""
        # 检查计算方法字段
        method = row.get('calculation_method', '')
        
        if 'C11' in method or 'uniaxial' in method:
            strain_dir = row.get('applied_strain_direction', '')
            stress_dir = row.get('measured_stress_direction', '')
            
            if strain_dir == stress_dir:
                return 'C11'
            else:
                return 'C12'
        elif 'C44' in method or 'shear' in method:
            strain_dir = row.get('applied_strain_direction', '')
            if 'yz' in strain_dir:
                return 'C44'
            elif 'xz' in strain_dir:
                return 'C55'
            elif 'xy' in strain_dir:
                return 'C66'
            else:
                return 'C44'  # 默认
        else:
            return 'unknown'
    
    @staticmethod
    def detect_outliers(
        data: List[Dict], 
        strain_key: str = "applied_strain",
        stress_key: str = "measured_stress_GPa",
        method: str = "iqr"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        检测异常值
        
        Parameters
        ----------
        data : List[Dict]
            数据点列表
        strain_key : str
            应变数据键名
        stress_key : str
            应力数据键名
        method : str
            异常值检测方法 ('iqr', 'zscore')
            
        Returns
        -------
        Tuple[List[Dict], List[Dict]]
            (正常数据, 异常数据)
        """
        if len(data) < 4:  # 数据点太少，不进行异常值检测
            return data, []
            
        stresses = np.array([row[stress_key] for row in data])
        
        if method == "iqr":
            Q1 = np.percentile(stresses, 25)
            Q3 = np.percentile(stresses, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            normal_data = [row for row in data 
                          if lower_bound <= row[stress_key] <= upper_bound]
            outlier_data = [row for row in data 
                           if not (lower_bound <= row[stress_key] <= upper_bound)]
                           
        elif method == "zscore":
            mean_stress = np.mean(stresses)
            std_stress = np.std(stresses)
            
            normal_data = [row for row in data 
                          if abs(row[stress_key] - mean_stress) <= 2 * std_stress]
            outlier_data = [row for row in data 
                           if abs(row[stress_key] - mean_stress) > 2 * std_stress]
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
            
        return normal_data, outlier_data