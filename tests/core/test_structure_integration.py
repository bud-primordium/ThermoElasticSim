"""
Structure模块集成测试

测试覆盖：
- Atom和Cell的交互
- 复杂场景和工作流
- 性能和数值稳定性
- 真实物理场景模拟
"""

import pytest
import numpy as np
from thermoelasticsim.core.structure import Atom, Cell


class TestAtomCellIntegration:
    """测试Atom和Cell集成"""
    
    def test_atom_modification_in_cell(self, sample_cell):
        """测试修改Cell中的原子"""
        original_positions = sample_cell.get_positions().copy()
        
        # 移动第一个原子
        displacement = [0.5, 0.3, -0.2]
        sample_cell.atoms[0].move_by(displacement)
        
        # 检查位置变化
        new_positions = sample_cell.get_positions()
        expected_pos = original_positions[0] + displacement
        assert np.allclose(new_positions[0], expected_pos)
        
        # 其他原子不应该受影响
        assert np.allclose(new_positions[1:], original_positions[1:])
    
    def test_velocity_modifications_and_temperature(self, sample_cell):
        """测试速度修改和温度计算的集成"""
        # 设置所有原子初始速度为零
        for atom in sample_cell.atoms:
            atom.velocity = np.zeros(3)
        
        initial_temp = sample_cell.calculate_temperature()
        assert initial_temp == 0.0
        
        # 给原子加速
        sample_cell.atoms[0].accelerate_by([0.1, 0.0, 0.0])
        sample_cell.atoms[1].accelerate_by([0.0, 0.1, 0.0])
        sample_cell.atoms[2].accelerate_by([0.0, 0.0, 0.1])
        
        final_temp = sample_cell.calculate_temperature()
        assert final_temp > 0
    
    def test_com_motion_with_atom_modifications(self, sample_cell):
        """测试原子修改后的质心运动"""
        # 给所有原子相同的速度增量
        uniform_acceleration = [0.1, 0.1, 0.1]
        for atom in sample_cell.atoms:
            atom.accelerate_by(uniform_acceleration)
        
        com_vel_before = sample_cell.get_com_velocity()
        
        # 移除质心运动
        sample_cell.remove_com_motion()
        
        com_vel_after = sample_cell.get_com_velocity()
        assert np.allclose(com_vel_after, [0, 0, 0], atol=1e-10)


class TestDeformationWorkflow:
    """测试变形工作流"""
    
    def test_multiple_deformations(self, sample_cell, deformation_matrices):
        """测试多次变形"""
        original_volume = sample_cell.volume
        
        # 应用压缩
        sample_cell.apply_deformation(deformation_matrices['uniform_compression'])
        compressed_volume = sample_cell.volume
        assert compressed_volume < original_volume
        
        # 再应用拉伸（逆变形）
        expansion_matrix = np.linalg.inv(deformation_matrices['uniform_compression'])
        sample_cell.apply_deformation(expansion_matrix)
        
        # 体积应该回到接近原始值
        final_volume = sample_cell.volume
        assert np.isclose(final_volume, original_volume, rtol=1e-10)
    
    def test_deformation_with_pbc(self, sample_cell, deformation_matrices):
        """测试变形与周期性边界条件的组合"""
        # 先将一些原子移到边界附近
        sample_cell.atoms[0].position = np.array([2.9, 2.9, 2.9])  # 接近边界
        
        # 应用变形
        sample_cell.apply_deformation(deformation_matrices['small_strain'])
        
        # 检查原子仍在合理位置
        positions = sample_cell.get_positions()
        assert np.all(np.isfinite(positions))
    
    def test_locked_vs_unlocked_deformation(self, sample_cell, deformation_matrices):
        """测试锁定和未锁定晶格的变形差异"""
        # 创建两个相同的cell
        cell_unlocked = sample_cell.copy()
        cell_locked = sample_cell.copy()
        cell_locked.lock_lattice_vectors()
        
        F = deformation_matrices['shear']
        
        # 应用相同变形
        cell_unlocked.apply_deformation(F)
        cell_locked.apply_deformation(F)
        
        # 晶格应该不同
        assert not np.allclose(cell_unlocked.lattice_vectors, cell_locked.lattice_vectors)
        
        # 但原子位置变形应该相同
        assert np.allclose(cell_unlocked.get_positions(), cell_locked.get_positions())


class TestSupercellWorkflow:
    """测试超胞工作流"""
    
    def test_supercell_and_deformation(self, sample_cell, deformation_matrices):
        """测试超胞构建后的变形"""
        # 构建超胞
        supercell = sample_cell.build_supercell((2, 2, 1))
        original_atoms = supercell.num_atoms
        
        # 应用变形
        supercell.apply_deformation(deformation_matrices['small_strain'])
        
        # 检查原子数量不变
        assert supercell.num_atoms == original_atoms
        
        # 检查所有原子位置有效
        positions = supercell.get_positions()
        assert np.all(np.isfinite(positions))
    
    def test_supercell_periodic_boundary(self, sample_cell):
        """测试超胞的周期性边界条件"""
        supercell = sample_cell.build_supercell((3, 3, 3))
        
        # 测试一个远离原点的位置
        far_position = np.array([10.0, 10.0, 10.0])
        wrapped = supercell.apply_periodic_boundary(far_position)
        
        # 应该被包装到盒子内
        box_lengths = supercell.get_box_lengths()
        assert np.all(wrapped >= 0)
        assert np.all(wrapped <= box_lengths)
    
    def test_supercell_temperature_conservation(self, sample_cell):
        """测试超胞构建时速度分布保持一致"""
        # 设置原始cell的温度
        for atom in sample_cell.atoms:
            atom.velocity = np.random.random(3) * 0.1
        
        # 构建超胞
        supercell = sample_cell.build_supercell((2, 2, 2))
        
        # 检查速度分布 - 每个原子的副本应该有相同的速度
        original_velocities = sample_cell.get_velocities()
        supercell_velocities = supercell.get_velocities()
        
        # 检查超胞中的原子数量正确
        assert len(supercell.atoms) == len(sample_cell.atoms) * 8  # 2*2*2 = 8
        
        # 检查前几个原子的速度是否正确复制
        for i in range(len(sample_cell.atoms)):
            np.testing.assert_array_almost_equal(
                supercell_velocities[i], original_velocities[i], decimal=10
            )


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_very_small_displacements(self, sample_cell):
        """测试非常小的位移"""
        tiny_displacement = np.array([1e-15, 1e-15, 1e-15])
        
        original_pos = sample_cell.atoms[0].position.copy()
        sample_cell.atoms[0].move_by(tiny_displacement)
        
        # 位置应该正确更新（即使是微小变化）
        new_pos = sample_cell.atoms[0].position
        expected_pos = original_pos + tiny_displacement
        assert np.allclose(new_pos, expected_pos, atol=1e-16)
    
    def test_large_deformations(self, sample_cell):
        """测试大变形的数值稳定性"""
        large_deformation = np.array([
            [2.0, 0.1, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.5]
        ])
        
        original_volume = sample_cell.volume
        sample_cell.apply_deformation(large_deformation)
        
        # 体积变化应该正确
        expected_volume = original_volume * np.linalg.det(large_deformation)
        assert np.isclose(sample_cell.volume, expected_volume)
        
        # 所有位置应该仍然有效
        positions = sample_cell.get_positions()
        assert np.all(np.isfinite(positions))
    
    def test_minimum_image_edge_cases(self, sample_cell):
        """测试最小镜像约定的边界情况"""
        box_lengths = sample_cell.get_box_lengths()
        
        # 测试正好是盒子长度一半的位移
        half_box_disp = box_lengths / 2
        min_disp = sample_cell.minimum_image(half_box_disp)
        assert np.all(np.abs(min_disp) <= box_lengths / 2 + 1e-10)
        
        # 测试负的一半盒子长度
        neg_half_box_disp = -box_lengths / 2
        min_disp = sample_cell.minimum_image(neg_half_box_disp)
        assert np.all(np.abs(min_disp) <= box_lengths / 2 + 1e-10)
    
    def test_pbc_numerical_precision(self, sample_cell):
        """测试周期性边界条件的数值精度"""
        # 测试接近边界的位置
        box_lengths = sample_cell.get_box_lengths()
        
        positions_near_boundary = np.array([
            [box_lengths[0] - 1e-10, 0, 0],      # 非常接近x边界
            [0, box_lengths[1] - 1e-10, 0],      # 非常接近y边界
            [0, 0, box_lengths[2] - 1e-10],      # 非常接近z边界
            [1e-10, 1e-10, 1e-10]                # 非常接近原点
        ])
        
        wrapped = sample_cell.apply_periodic_boundary(positions_near_boundary)
        
        # 所有位置应该仍然有效且在盒子内
        assert np.all(np.isfinite(wrapped))
        assert np.all(wrapped >= 0)
        assert np.all(wrapped <= box_lengths)


class TestPhysicalRealism:
    """测试物理真实性"""
    
    def test_energy_conservation_analog(self, sample_cell):
        """测试类似能量守恒的原理（动量守恒）"""
        # 设置初始动量
        initial_momenta = []
        for i, atom in enumerate(sample_cell.atoms):
            velocity = np.array([0.1 * (i+1), 0.1 * (i+1), 0.1 * (i+1)])
            atom.velocity = velocity
            initial_momenta.append(atom.mass * velocity)
        
        total_initial_momentum = sum(initial_momenta)
        
        # 移除质心运动
        sample_cell.remove_com_motion()
        
        # 计算最终总动量
        final_momenta = []
        for atom in sample_cell.atoms:
            final_momenta.append(atom.mass * atom.velocity)
        
        total_final_momentum = sum(final_momenta)
        
        # 总动量应该接近零（移除质心运动后）
        assert np.allclose(total_final_momentum, [0, 0, 0], atol=1e-10)
    
    def test_isotropy_under_uniform_scaling(self, sample_cell):
        """测试均匀缩放下的各向同性"""
        scale_factor = 1.1
        uniform_scaling = np.eye(3) * scale_factor
        
        original_volume = sample_cell.volume
        original_distances = []
        
        # 计算原始原子间距离
        positions = sample_cell.get_positions()
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                original_distances.append(dist)
        
        # 应用均匀缩放
        sample_cell.apply_deformation(uniform_scaling)
        
        # 检查体积缩放
        new_volume = sample_cell.volume
        expected_volume = original_volume * (scale_factor ** 3)
        assert np.isclose(new_volume, expected_volume)
        
        # 检查距离缩放
        new_positions = sample_cell.get_positions()
        new_distances = []
        for i in range(len(new_positions)):
            for j in range(i+1, len(new_positions)):
                dist = np.linalg.norm(new_positions[i] - new_positions[j])
                new_distances.append(dist)
        
        # 所有距离应该按相同因子缩放
        for orig_dist, new_dist in zip(original_distances, new_distances):
            expected_dist = orig_dist * scale_factor
            assert np.isclose(new_dist, expected_dist, rtol=1e-10)
    
    def test_crystal_symmetry_preservation(self, orthorhombic_lattice):
        """测试晶体对称性保持"""
        # 创建简单立方结构
        atoms = [
            Atom(id=0, symbol='Cu', mass_amu=63.5, position=[0, 0, 0]),
            Atom(id=1, symbol='Cu', mass_amu=63.5, position=[2, 0, 0]),
            Atom(id=2, symbol='Cu', mass_amu=63.5, position=[0, 2.5, 0]),
            Atom(id=3, symbol='Cu', mass_amu=63.5, position=[2, 2.5, 0])
        ]
        
        cell = Cell(lattice_vectors=orthorhombic_lattice, atoms=atoms)
        
        # 应用保持对称性的变形（均匀压缩）
        symmetric_deformation = np.array([
            [0.95, 0.0, 0.0],
            [0.0, 0.95, 0.0],
            [0.0, 0.0, 0.95]
        ])
        
        cell.apply_deformation(symmetric_deformation)
        
        # 检查相对位置关系保持
        positions = cell.get_positions()
        
        # 原子0和1的x距离应该保持相对关系
        x_dist_01 = abs(positions[1][0] - positions[0][0])
        x_dist_23 = abs(positions[3][0] - positions[2][0])
        assert np.isclose(x_dist_01, x_dist_23, rtol=1e-10)
        
        # 原子0和2的y距离应该保持相对关系
        y_dist_02 = abs(positions[2][1] - positions[0][1])
        y_dist_13 = abs(positions[3][1] - positions[1][1])
        assert np.isclose(y_dist_02, y_dist_13, rtol=1e-10)