#!/usr/bin/env python3
"""
高精度测试脚本：使用极严格的收敛条件计算fcc铝的弹性常数
针对优化器收敛问题的专门测试
"""

import numpy as np
import sys
import os
import logging
from datetime import datetime

from thermoelasticsim.core.structure import Cell, Atom
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.elastic.deformation_method.zero_temp import ZeroTempDeformationCalculator

def setup_logging(test_name: str = "fcc_aluminum_high_precision") -> str:
    """设置高精度测试的日志系统"""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{test_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # 清除现有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 设置详细的调试级别
    logging.getLogger('thermoelasticsim.elastic.deformation_method.zero_temp').setLevel(logging.INFO)
    logging.getLogger('thermoelasticsim.utils.optimizers').setLevel(logging.INFO)
    
    return log_filepath

def create_fcc_aluminum_supercell(lattice_parameter: float = 4.05, supercell_dims: tuple = (3, 3, 3)) -> Cell:
    """创建fcc铝的超胞结构"""
    a = lattice_parameter
    # 首先创建常规单胞
    unit_cell_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])
    
    fractional_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    cartesian_positions = fractional_positions @ unit_cell_vectors
    
    atoms = []
    al_mass = 26.9815
    for i, pos in enumerate(cartesian_positions):
        atoms.append(Atom(id=i, symbol="Al", mass_amu=al_mass, position=pos))
    
    unit_cell = Cell(unit_cell_vectors, atoms)
    
    # 从单胞构建超胞
    supercell = unit_cell.build_supercell(supercell_dims)
    
    return supercell

def test_high_precision_calculation():
    """执行高精度弹性常数计算"""
    log_filepath = setup_logging()
    
    print("=== 高精度fcc铝弹性常数计算 (3x3x3超胞) ===")
    print(f"目标: 在修正优化器后，使用超胞获得精确的物理结果")
    print(f"策略: 极严格收敛条件 + 增加迭代次数")
    print(f"日志文件: {log_filepath}")
    print("="*60)
    
    logger = logging.getLogger(__name__)
    logger.info("开始高精度fcc铝弹性常数计算 (3x3x3超胞)")
    
    # 1. 创建超胞和势函数
    # 使用一个接近我们静态计算得到的平衡值的初始晶格常数
    cell = create_fcc_aluminum_supercell(lattice_parameter=4.04) 
    potential = EAMAl1Potential(cutoff=6.5)
    
    initial_energy = potential.calculate_energy(cell)
    print(f"初始晶胞原子数: {cell.num_atoms}")
    print(f"初始每原子能量: {initial_energy/cell.num_atoms:.8f} eV")
    
    # 2. 设置极严格的优化参数
    high_precision_params = {
        'optimizer_type': 'L-BFGS',
        'optimizer_params': {
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxiter': 5000 
        }
    }
    
    print("\n高精度优化参数:")
    print(f"  函数收敛阈值: {high_precision_params['optimizer_params']['ftol']:.0e}")
    print(f"  梯度收敛阈值: {high_precision_params['optimizer_params']['gtol']:.0e}")
    print(f"  最大迭代次数: {high_precision_params['optimizer_params']['maxiter']}")
    
    # 3. 创建计算器
    calculator = ZeroTempDeformationCalculator(
        cell=cell,
        potential=potential,
        delta=0.001,  # 使用更小的应变幅度
        num_steps=1,  # 生产模式，只用一个点
        relaxer_params=high_precision_params
    )
    
    # 4. 执行计算
    print("\n开始高精度计算...")
    print("注意: 由于使用超胞和严格收敛条件，计算时间可能较长")
    
    try:
        start_time = datetime.now()
        C_matrix, r2_score = calculator.calculate()
        end_time = datetime.now()
        
        calculation_time = (end_time - start_time).total_seconds()
        print(f"\n计算完成！用时: {calculation_time:.1f} 秒")
        
        # 5. 分析结果
        print("\n=== 计算结果分析 ===")
        
        # 弛豫后状态
        final_energy = potential.calculate_energy(cell)
        # 对于超胞，我们关心的是等效单胞的晶格常数
        final_a = np.linalg.norm(cell.lattice_vectors[0]) / 3.0
        
        print(f"弛豫后每原子能量: {final_energy/cell.num_atoms:.8f} eV")
        print(f"弛豫后等效晶格常数: {final_a:.6f} Å")
        print(f"与静态计算值(4.0424 Å)的误差: {abs(final_a - 4.0424)/4.0424*100:.4f}%")
        print(f"与文献值(4.045 Å)的误差: {abs(final_a - 4.045)/4.045*100:.4f}%")
        
        # 弹性常数
        C11 = C_matrix[0, 0]
        C12 = C_matrix[0, 1]
        C44 = C_matrix[3, 3]
        
        print(f"\n弹性常数结果:")
        print(f"  C₁₁ = {C11:.2f} GPa  (文献值: 110 GPa, 误差: {abs(C11-110)/110*100:.2f}%) ")
        print(f"  C₁₂ = {C12:.2f} GPa  (文献值: 61 GPa,  误差: {abs(C12-61)/61*100:.2f}%) ")
        print(f"  C₄₄ = {C44:.2f} GPa  (文献值: 33 GPa,  误差: {abs(C44-33)/33*100:.2f}%) ")
        print(f"  拟合质量: R² = {r2_score:.6f}")
        
        # 检查基态应力
        potential.calculate_forces(cell)
        stress_tensor = cell.calculate_stress_tensor(potential)
        stress_gpa = np.linalg.norm(stress_tensor) * 160.218
        print(f"  基态应力大小: {stress_gpa:.4f} GPa")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"fcc_aluminum_high_precision_supercell_{timestamp}.txt"
        results_filepath = os.path.join(os.path.dirname(__file__), 'logs', results_filename)
        
        with open(results_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 高精度fcc铝弹性常数计算结果 (3x3x3超胞)\n")
            f.write(f"# 弛豫后等效晶格常数: {final_a:.6f} Å\n")
            f.write(f"# 基态应力大小: {stress_gpa:.4f} GPa\n")
            f.write(f"# R² = {r2_score:.6f}\n")
            f.write(f"# 计算时间: {calculation_time:.1f} 秒\n")
            f.write(f"# 测试时间: {timestamp}\n")
            f.write(f"# 优化参数: ftol={high_precision_params['optimizer_params']['ftol']:.0e}, gtol={high_precision_params['optimizer_params']['gtol']:.0e}\n")
            f.write("# 弹性常数矩阵 (GPa):\n")
            
        np.savetxt(results_filepath, C_matrix, fmt="%.6f")
        
        print(f"\n结果已保存到: {results_filepath}")
        logger.info(f"高精度计算完成，结果保存到: {results_filepath}")
        
        # 评估改进效果
        print(f"\n=== 改进效果评估 ===")
        if stress_gpa < 0.1:
            print(f"✓ 基态应力极低 ({stress_gpa:.4f} GPa)")
        else:
            print(f"✗ 基态应力仍然较大 ({stress_gpa:.4f} GPa)")
            
        if abs(C11-110)/110 < 0.05 and abs(C12-61)/61 < 0.05 and abs(C44-33)/33 < 0.05:
            print("✓ 所有弹性常数精度很高 (误差 < 5%)")
        else:
            print("✗ 弹性常数精度仍需改进")
            
        if r2_score > 0.99:
            print("✓ 拟合质量优秀")
        else:
            print("✗ 拟合质量需要改进")
            
    except Exception as e:
        print(f"计算失败: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"高精度计算失败: {e}")

if __name__ == "__main__":
    test_high_precision_calculation()