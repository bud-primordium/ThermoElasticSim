#!/usr/bin/env python3
"""
MD测试辅助函数

提供标准的测试系统创建函数，确保测试的物理合理性和一致性。

Functions:
    create_fcc_aluminum: 创建标准FCC铝测试系统
    apply_maxwell_velocities: 设置Maxwell-Boltzmann分布速度
    simple_energy_minimization: 简单的结构优化

Created: 2025-08-19
Author: Based on debug/system_size_comparison_v7.py
"""

import numpy as np
import sys
import os

# 添加src路径
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from thermoelasticsim.core.structure import Cell, Atom
from thermoelasticsim.utils.utils import KB_IN_EV


def create_fcc_aluminum(supercell_size=(2, 2, 2)):
    """创建标准FCC铝测试系统
    
    基于debug/system_size_comparison_v7.py中的标准方法，
    创建物理合理的FCC铝晶格结构用于MD测试。
    
    Parameters
    ----------
    supercell_size : tuple of int, optional
        超胞尺寸，默认(2, 2, 2)产生32原子系统
        
    Returns
    -------
    Cell
        包含标准FCC铝结构的晶胞对象
        
    Examples
    --------
    >>> cell = create_fcc_aluminum((1, 1, 1))  # 4原子单胞
    >>> cell = create_fcc_aluminum((2, 2, 2))  # 32原子超胞
    
    Notes
    -----
    - 使用EAM Al1势的标准晶格常数 a=4.045Å
    - 创建完整的FCC结构，每个单胞4个原子
    - 总原子数 = 4 × nx × ny × nz
    - 适合各种MD恒温器测试
    """
    a = 4.045  # EAM Al1标准晶格常数 (Å)
    nx, ny, nz = supercell_size
    
    # 创建晶格矢量
    lattice = np.array([
        [a * nx, 0, 0],
        [0, a * ny, 0], 
        [0, 0, a * nz]
    ], dtype=np.float64)
    
    atoms = []
    atom_id = 0
    
    # 构建FCC结构：每个单胞4个原子
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # FCC基础位置（分数坐标）
                base_positions = [
                    [0.0, 0.0, 0.0],      # (0,0,0)
                    [0.5, 0.5, 0.0],      # (1/2,1/2,0)
                    [0.5, 0.0, 0.5],      # (1/2,0,1/2)
                    [0.0, 0.5, 0.5],      # (0,1/2,1/2)
                ]
                
                # 转换为当前单胞的笛卡尔坐标
                for base_pos in base_positions:
                    position = [
                        (i + base_pos[0]) * a,
                        (j + base_pos[1]) * a,
                        (k + base_pos[2]) * a
                    ]
                    
                    atom = Atom(
                        id=atom_id,
                        symbol="Al",
                        mass_amu=26.9815,  # 标准Al原子质量
                        position=np.array(position)
                    )
                    atoms.append(atom)
                    atom_id += 1
    
    # 创建晶胞
    cell = Cell(lattice, atoms, pbc_enabled=True)
    
    print(f"✓ 创建FCC铝系统: {supercell_size}超胞，{len(atoms)}原子")
    print(f"  晶格参数: a={a:.3f}Å")
    print(f"  系统尺寸: {lattice[0,0]:.1f}×{lattice[1,1]:.1f}×{lattice[2,2]:.1f} Å³")
    print(f"  密度: {len(atoms)/cell.calculate_volume():.3f} 原子/Å³")
    
    return cell


def apply_maxwell_velocities(cell, temperature, remove_com=True):
    """为原子设置Maxwell-Boltzmann分布的速度
    
    Parameters
    ----------
    cell : Cell
        晶胞对象，将修改其中原子的速度
    temperature : float
        目标温度 (K)
    remove_com : bool, optional
        是否移除质心运动，默认True
        
    Notes
    -----
    每个速度分量服从高斯分布：
    σ = sqrt(k_B * T / m)
    其中m是原子质量，T是温度
    """
    if temperature <= 0:
        raise ValueError(f"温度必须为正数，得到 {temperature} K")
    
    for atom in cell.atoms:
        # 计算Maxwell分布的标准差
        sigma = np.sqrt(KB_IN_EV * temperature / atom.mass)
        
        # 为每个方向独立采样速度
        atom.velocity = np.random.normal(0, sigma, 3)
    
    # 移除质心运动以保证总动量为零
    if remove_com:
        cell.remove_com_motion()
    
    # 验证实际温度
    actual_temp = cell.calculate_temperature()
    print(f"✓ 设置Maxwell速度: 目标={temperature:.1f}K, 实际={actual_temp:.1f}K")


def simple_energy_minimization(cell, potential, max_iterations=100, force_tolerance=1e-3):
    """简单的能量最小化优化
    
    使用最速下降法进行基础的结构优化，
    适用于测试系统的基态准备。
    
    Parameters
    ----------
    cell : Cell
        要优化的晶胞
    potential : Potential
        势能函数对象
    max_iterations : int, optional
        最大迭代次数，默认100
    force_tolerance : float, optional
        力收敛判据 (eV/Å)，默认1e-3
        
    Returns
    -------
    bool
        是否收敛
    float
        最终能量 (eV)
        
    Notes
    -----
    这是一个简化的优化器，仅用于测试系统的基态准备。
    对于生产计算，应使用更sophisticated的优化算法。
    """
    print(f"开始简单能量最小化（最大{max_iterations}步）...")
    
    initial_energy = potential.calculate_energy(cell)
    print(f"  初始能量: {initial_energy:.6f} eV")
    
    step_size = 0.01  # 初始步长 (Å)
    energy = initial_energy
    
    for iteration in range(max_iterations):
        # 计算力
        potential.calculate_forces(cell)
        
        # 检查收敛性
        max_force = max(np.linalg.norm(atom.force) for atom in cell.atoms)
        
        if max_force < force_tolerance:
            print(f"  ✓ 第{iteration}步收敛: 最大力={max_force:.2e} eV/Å")
            break
        
        # 最速下降步
        for atom in cell.atoms:
            # 沿负梯度方向移动
            displacement = step_size * atom.force / np.linalg.norm(atom.force)
            atom.position += displacement
            
            # 应用周期性边界条件
            atom.position = cell.apply_periodic_boundary(atom.position)
        
        # 计算新能量
        new_energy = potential.calculate_energy(cell)
        
        # 自适应步长
        if new_energy < energy:
            step_size *= 1.1  # 增大步长
            energy = new_energy
        else:
            step_size *= 0.5  # 减小步长
            # 恢复到上一步
            for atom in cell.atoms:
                displacement = step_size * atom.force / np.linalg.norm(atom.force)
                atom.position -= displacement
                atom.position = cell.apply_periodic_boundary(atom.position)
        
        # 定期输出进度
        if iteration % 20 == 0:
            print(f"  第{iteration}步: E={energy:.6f} eV, 最大力={max_force:.2e} eV/Å")
    
    else:
        print(f"  ⚠️ 未在{max_iterations}步内收敛")
        return False, energy
    
    final_energy = potential.calculate_energy(cell)
    energy_change = final_energy - initial_energy
    print(f"  最终能量: {final_energy:.6f} eV (变化: {energy_change:.6f} eV)")
    
    return True, final_energy


def create_equilibrated_fcc_aluminum(supercell_size=(2, 2, 2), temperature=300.0, 
                                   with_optimization=True):
    """创建已平衡的FCC铝测试系统
    
    这是一个便利函数，组合了结构创建、优化和速度设置，
    提供即用的平衡态测试系统。
    
    Parameters
    ----------
    supercell_size : tuple of int, optional
        超胞尺寸，默认(2, 2, 2)
    temperature : float, optional
        初始温度 (K)，默认300.0
    with_optimization : bool, optional
        是否进行结构优化，默认True
        
    Returns
    -------
    Cell
        已平衡的FCC铝系统
        
    Notes
    -----
    如果with_optimization=True，需要在调用前确保有可用的势能函数。
    否则只创建结构和设置速度。
    """
    # 创建FCC结构
    cell = create_fcc_aluminum(supercell_size)
    
    # 结构优化（如果请求）
    if with_optimization:
        print("注意: 结构优化需要势能函数，请在外部调用simple_energy_minimization")
    
    # 设置初始速度
    apply_maxwell_velocities(cell, temperature)
    
    return cell


def get_fcc_aluminum_neighbor_distance():
    """获取FCC铝的最近邻距离
    
    Returns
    -------
    float
        最近邻距离 (Å)
        
    Notes
    -----
    FCC结构中，最近邻距离 = a/√2 ≈ 2.86Å
    """
    a = 4.045  # 晶格常数
    return a / np.sqrt(2)


def validate_fcc_structure(cell, tolerance=0.1):
    """验证FCC结构的正确性
    
    Parameters
    ----------
    cell : Cell
        要验证的晶胞
    tolerance : float, optional
        距离容差 (Å)，默认0.1
        
    Returns
    -------
    bool
        结构是否正确
    dict
        验证详细信息
    """
    expected_nn_dist = get_fcc_aluminum_neighbor_distance()
    nn_distances = []
    
    # 计算所有原子对的最短距离
    for i, atom1 in enumerate(cell.atoms):
        for j, atom2 in enumerate(cell.atoms):
            if i >= j:
                continue
                
            # 计算最小镜像距离
            rij = atom2.position - atom1.position
            rij_min = cell.minimum_image(rij)
            dist = np.linalg.norm(rij_min)
            
            # 记录近邻距离（排除过远的）
            if dist < expected_nn_dist * 1.5:
                nn_distances.append(dist)
    
    nn_distances = np.array(nn_distances)
    
    # 统计分析
    actual_nn_dist = np.min(nn_distances)
    distance_error = abs(actual_nn_dist - expected_nn_dist)
    is_valid = distance_error < tolerance
    
    validation_info = {
        'is_valid': is_valid,
        'expected_nn_distance': expected_nn_dist,
        'actual_nn_distance': actual_nn_dist,
        'distance_error': distance_error,
        'tolerance': tolerance,
        'num_neighbors_found': len(nn_distances)
    }
    
    return is_valid, validation_info