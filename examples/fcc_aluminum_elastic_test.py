#!/usr/bin/env python3
"""
测试脚本：使用EAM Al1势和3x3x3超胞计算fcc铝的弹性常数
包含详细的中间输出用于验证实现正确性。
这是一个经过修正的、可靠的验证脚本。
"""

import numpy as np
import sys
import os
import logging
from datetime import datetime

from thermoelasticsim.core.structure import Cell, Atom
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.elastic.deformation_method.zero_temp import ZeroTempDeformationCalculator

def setup_logging(test_name: str = "fcc_aluminum_supercell_test") -> str:
    """
    设置规范的日志系统
    """
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{test_name}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 手动创建文件和控制台handler，设置不同的级别
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件handler - 记录所有DEBUG及以上级别
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 控制台handler - 只显示INFO及以上级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 为我们关心的模块设置特定级别
    logging.getLogger('thermoelasticsim.elastic.deformation_method.zero_temp').setLevel(logging.INFO)
    logging.getLogger('thermoelasticsim.utils.optimizers').setLevel(logging.DEBUG)
    # 减少EAM势能的输出频率，避免刷屏
    logging.getLogger('thermoelasticsim.potentials.eam').setLevel(logging.INFO)
    
    return log_filepath

def create_fcc_aluminum_supercell(lattice_parameter: float, supercell_dims: tuple = (3, 3, 3)) -> Cell:
    """
    创建fcc铝的超胞结构
    """
    a = lattice_parameter
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
    supercell = unit_cell.build_supercell(supercell_dims)
    return supercell

def print_elastic_constants_analysis(C_matrix: np.ndarray, r2_score: float):
    """
    打印弹性常数矩阵的详细分析
    """
    print("\n" + "="*80)
    print("弹性常数矩阵分析")
    print("="*80)
    
    print(f"拟合质量 R² = {r2_score:.6f}")
    print(f"拟合质量评价: {'优秀' if r2_score > 0.99 else '良好' if r2_score > 0.95 else '一般'}")
    
    print("\n完整弹性常数矩阵 C_{ij} (GPa):")
    print("-" * 50)
    for i in range(6):
        row_str = "  ".join(f"{C_matrix[i,j]:8.2f}" for j in range(6))
        print(f"  {row_str}")
    
    C11 = C_matrix[0, 0]
    C12 = C_matrix[0, 1]  
    C44 = C_matrix[3, 3]
    
    print(f"\n立方晶系主要弹性常数:")
    print(f"  C₁₁ = {C11:.2f} GPa")
    print(f"  C₁₂ = {C12:.2f} GPa")
    print(f"  C₄₄ = {C44:.2f} GPa")
    
    K = (C11 + 2*C12) / 3
    print(f"  体模量 K = {K:.2f} GPa")
    
    print(f"\n与EAM Al1文献值对比:")
    print(f"  C₁₁ 文献值: ~110 GPa, 计算值: {C11:.2f} GPa, 误差: {abs(C11-110)/110*100:.2f}%")
    print(f"  C₁₂ 文献值: ~61 GPa,  计算值: {C12:.2f} GPa, 误差: {abs(C12-61)/61*100:.2f}%")
    print(f"  C₄₄ 文献值: ~33 GPa,  计算值: {C44:.2f} GPa, 误差: {abs(C44-33)/33*100:.2f}%")

def print_detailed_cell_info(cell: Cell, title: str, supercell_dims: tuple):
    """
    打印详细的晶胞信息
    """
    print(f"\n{title}:")
    print("-" * 50)
    lattice = cell.lattice_vectors
    
    # 明确计算并报告等效单胞的晶格参数
    a = np.linalg.norm(lattice[0]) / supercell_dims[0]
    b = np.linalg.norm(lattice[1]) / supercell_dims[1]
    c = np.linalg.norm(lattice[2]) / supercell_dims[2]
    print(f"   等效单胞晶格参数: a={a:.5f}, b={b:.5f}, c={c:.5f} Å")
    print(f"   超胞体积: {cell.volume:.6f} Å³")
    print(f"   原子总数: {cell.num_atoms}")

def main():
    """主函数：执行fcc铝弹性常数计算的最终验证"""
    log_filepath = setup_logging("fcc_aluminum_supercell_test")
    
    print("fcc铝弹性常数计算最终验证 (3x3x3超胞)")
    print("="*80)
    print("本脚本使用修正后的优化器和超胞，从远离平衡点的位置开始弛豫。")
    print("预期目标：")
    print("  1. 弛豫后的晶格常数收敛到 ~4.042 Å。")
    print("  2. 计算出的弹性常数与文献值吻合。")
    print(f"日志文件: {log_filepath}")
    print("="*80)
    
    logger = logging.getLogger(__name__)
    logger.info("开始最终验证：fcc铝弹性常数计算 (3x3x3超胞)")
    
    supercell_dims = (3, 3, 3)
    print("\n1. 创建fcc铝初始3x3x3超胞 (a=4.04 Å，接近平衡值)...")
    cell = create_fcc_aluminum_supercell(lattice_parameter=4.04, supercell_dims=supercell_dims)
    print_detailed_cell_info(cell, "初始超胞", supercell_dims)
    
    print("\n2. 加载EAM Al1势...")
    potential = EAMAl1Potential(cutoff=6.5)
    initial_energy = potential.calculate_energy(cell)
    print(f"   初始每原子能量: {initial_energy/cell.num_atoms:.8f} eV/atom")
    
    print("\n3. 创建零温变形计算器...")
    calculator = ZeroTempDeformationCalculator(
        cell=cell,
        potential=potential,
        delta=0.005,
        num_steps=1,
        relaxer_params={
            'optimizer_type': 'L-BFGS',
            'optimizer_params': {
                'ftol': 1e-8,
                'gtol': 1e-7,
                'maxiter': 5000
            }
        },
        supercell_dims=supercell_dims
    )
    
    print("\n4. 开始执行弹性常数计算...")
    try:
        C_matrix, r2_score = calculator.calculate()
        
        print("\n   ✓ 计算完成，显示弛豫后的基态:")
        print_detailed_cell_info(cell, "弛豫后基态超胞", supercell_dims)
        relaxed_energy = potential.calculate_energy(cell)
        print(f"   弛豫后每原子能量: {relaxed_energy/cell.num_atoms:.8f} eV/atom")
        
        final_a_equivalent = np.linalg.norm(cell.lattice_vectors[0]) / supercell_dims[0]
        print(f"   弛豫后等效晶格常数: a = {final_a_equivalent:.5f} Å")
        print(f"   与静态计算值(4.0424 Å)的差异: {abs(final_a_equivalent - 4.0424):.5f} Å ({abs(final_a_equivalent - 4.0424)/4.0424*100:.3f}%)")
        
        print_elastic_constants_analysis(C_matrix, r2_score)
        
    except Exception as e:
        print(f"   ✗ 计算失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
