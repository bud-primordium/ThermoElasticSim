"""
Cell类的单元测试

测试覆盖：
- 构造器和验证
- 晶格操作
- 几何变换
- 周期性边界条件
- 物理量计算
- 错误处理
"""

import pytest
import numpy as np
from thermoelasticsim.core.structure import Atom, Cell


class TestCellConstructor:
    """测试Cell构造器和验证"""
    
    def test_basic_construction(self, simple_lattice, sample_atoms):
        """测试基本构造"""
        cell = Cell(lattice_vectors=simple_lattice, atoms=sample_atoms, pbc_enabled=True)
        
        assert np.allclose(cell.lattice_vectors, simple_lattice)
        assert len(cell.atoms) == len(sample_atoms)
        assert cell.pbc_enabled is True
        assert cell.lattice_locked is False
        assert cell.volume > 0
    
    def test_constructor_with_pbc_disabled(self, simple_lattice, sample_atoms):
        """测试禁用周期性边界条件"""
        cell = Cell(lattice_vectors=simple_lattice, atoms=sample_atoms, pbc_enabled=False)
        assert cell.pbc_enabled is False
    
    def test_empty_atoms_list_error(self, simple_lattice):
        """测试空原子列表错误"""
        with pytest.raises(ValueError, match="原子列表不能为空"):
            Cell(lattice_vectors=simple_lattice, atoms=[], pbc_enabled=True)
    
    def test_invalid_lattice_vectors(self, sample_atoms):
        """测试无效晶格向量"""
        # 非方阵
        invalid_lattice1 = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="Invalid lattice vectors"):
            Cell(lattice_vectors=invalid_lattice1, atoms=sample_atoms)
        
        # 奇异矩阵（不可逆）
        invalid_lattice2 = np.array([
            [1, 0, 0],
            [2, 0, 0],  # 线性相关
            [0, 0, 1]
        ])
        with pytest.raises(ValueError, match="Invalid lattice vectors"):
            Cell(lattice_vectors=invalid_lattice2, atoms=sample_atoms)
        
        # 负体积（左手坐标系）
        invalid_lattice3 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        with pytest.raises(ValueError, match="Invalid lattice vectors"):
            Cell(lattice_vectors=invalid_lattice3, atoms=sample_atoms)
    
    def test_duplicate_atom_ids_error(self, simple_lattice):
        """测试重复原子ID错误"""
        atoms_with_duplicate_ids = [
            Atom(id=1, symbol='H', mass_amu=1.0, position=[0, 0, 0]),
            Atom(id=1, symbol='O', mass_amu=16.0, position=[1, 0, 0])  # 重复ID
        ]
        with pytest.raises(ValueError, match="原子ID 1 重复"):
            Cell(lattice_vectors=simple_lattice, atoms=atoms_with_duplicate_ids)
    
    def test_invalid_atom_mass_error(self, simple_lattice):
        """测试无效原子质量错误"""
        atoms_with_zero_mass = [
            Atom(id=1, symbol='H', mass_amu=0.0, position=[0, 0, 0])
        ]
        with pytest.raises(ValueError, match="原子 1 的质量必须为正数"):
            Cell(lattice_vectors=simple_lattice, atoms=atoms_with_zero_mass)
    
    def test_nan_atom_position_error(self, simple_lattice):
        """测试NaN原子位置错误"""
        atoms_with_nan = [
            Atom(id=1, symbol='H', mass_amu=1.0, position=[np.nan, 0, 0])
        ]
        with pytest.raises(ValueError, match="原子 1 的位置包含无效值"):
            Cell(lattice_vectors=simple_lattice, atoms=atoms_with_nan)


class TestCellLatticeOperations:
    """测试晶格操作"""
    
    def test_calculate_volume(self, sample_cell):
        """测试体积计算"""
        expected_volume = np.linalg.det(sample_cell.lattice_vectors)
        assert np.isclose(sample_cell.volume, expected_volume)
        assert sample_cell.calculate_volume() == expected_volume
    
    def test_get_box_lengths(self, sample_cell):
        """测试盒子长度计算"""
        lengths = sample_cell.get_box_lengths()
        expected_lengths = np.linalg.norm(sample_cell.lattice_vectors, axis=1)
        assert np.allclose(lengths, expected_lengths)
        assert len(lengths) == 3
    
    def test_lock_unlock_lattice(self, sample_cell):
        """测试晶格锁定和解锁"""
        # 初始状态
        assert sample_cell.lattice_locked is False
        
        # 锁定
        sample_cell.lock_lattice_vectors()
        assert sample_cell.lattice_locked is True
        
        # 解锁
        sample_cell.unlock_lattice_vectors()
        assert sample_cell.lattice_locked is False


class TestCellDeformation:
    """测试变形操作"""
    
    def test_apply_deformation_unlocked(self, sample_cell, deformation_matrices):
        """测试未锁定晶格的变形"""
        original_lattice = sample_cell.lattice_vectors.copy()
        original_positions = sample_cell.get_positions().copy()
        original_volume = sample_cell.volume
        
        F = deformation_matrices['small_strain']
        sample_cell.apply_deformation(F)
        
        # 检查晶格变形
        expected_lattice = F @ original_lattice
        assert np.allclose(sample_cell.lattice_vectors, expected_lattice)
        
        # 检查体积变化
        expected_volume = original_volume * np.linalg.det(F)
        assert np.isclose(sample_cell.volume, expected_volume)
        
        # 检查原子位置变形（考虑PBC包装）
        expected_positions = original_positions @ F.T
        actual_positions = sample_cell.get_positions()
        
        # 由于PBC会将位置包装到最小镜像范围，我们需要检查等价性
        for i in range(len(expected_positions)):
            # 计算期望位置和实际位置的差
            diff = expected_positions[i] - actual_positions[i]
            # 应用最小镜像检查差异是否小于数值误差
            min_image_diff = sample_cell.minimum_image(diff)
            assert np.allclose(min_image_diff, 0.0, atol=1e-10)
    
    def test_apply_deformation_locked(self, sample_cell, deformation_matrices):
        """测试锁定晶格的变形"""
        sample_cell.lock_lattice_vectors()
        
        original_lattice = sample_cell.lattice_vectors.copy()
        original_positions = sample_cell.get_positions().copy()
        original_volume = sample_cell.volume
        
        F = deformation_matrices['shear']
        sample_cell.apply_deformation(F)
        
        # 检查晶格未变
        assert np.allclose(sample_cell.lattice_vectors, original_lattice)
        assert np.isclose(sample_cell.volume, original_volume)
        
        # 检查原子位置仍然变形（考虑PBC包装）
        expected_positions = original_positions @ F.T
        actual_positions = sample_cell.get_positions()
        
        # 由于PBC会将位置包装到最小镜像范围，我们需要检查等价性
        for i in range(len(expected_positions)):
            # 计算期望位置和实际位置的差
            diff = expected_positions[i] - actual_positions[i]
            # 应用最小镜像检查差异是否小于数值误差
            min_image_diff = sample_cell.minimum_image(diff)
            assert np.allclose(min_image_diff, 0.0, atol=1e-10)
    
    def test_identity_deformation(self, sample_cell, deformation_matrices):
        """测试单位变形矩阵"""
        original_lattice = sample_cell.lattice_vectors.copy()
        original_positions = sample_cell.get_positions().copy()
        
        sample_cell.apply_deformation(deformation_matrices['identity'])
        
        # 检查晶格保持不变
        assert np.allclose(sample_cell.lattice_vectors, original_lattice)
        
        # 检查位置保持不变（考虑PBC包装）
        actual_positions = sample_cell.get_positions()
        for i in range(len(original_positions)):
            # 计算原始位置和实际位置的差
            diff = original_positions[i] - actual_positions[i]
            # 应用最小镜像检查差异是否小于数值误差
            min_image_diff = sample_cell.minimum_image(diff)
            assert np.allclose(min_image_diff, 0.0, atol=1e-10)


class TestCellPeriodicBoundary:
    """测试周期性边界条件"""
    
    def test_apply_periodic_boundary_single_atom(self, sample_cell):
        """测试单原子周期性边界条件"""
        # 将原子移到盒子外
        outside_position = np.array([5.0, -2.0, 4.5])  # 盒子大小是3x3x3
        
        wrapped_position = sample_cell.apply_periodic_boundary(outside_position)
        
        # 使用最小镜像约定，坐标应该在[-box_length/2, box_length/2)范围内
        box_lengths = sample_cell.get_box_lengths()
        assert np.all(wrapped_position >= -box_lengths/2)
        assert np.all(wrapped_position < box_lengths/2)
    
    def test_apply_periodic_boundary_multiple_atoms(self, sample_cell):
        """测试多原子周期性边界条件"""
        positions = np.array([
            [4.0, 1.0, 1.0],   # x超出
            [1.0, -1.0, 1.0],  # y负值
            [1.0, 1.0, 7.0]    # z超出
        ])
        
        wrapped_positions = sample_cell.apply_periodic_boundary(positions)
        
        # 使用最小镜像约定，所有坐标应该在[-box_length/2, box_length/2)范围内
        box_lengths = sample_cell.get_box_lengths()
        assert wrapped_positions.shape == positions.shape
        assert np.all(wrapped_positions >= -box_lengths/2)
        assert np.all(wrapped_positions < box_lengths/2)
    
    def test_pbc_disabled(self, simple_lattice, sample_atoms):
        """测试禁用周期性边界条件"""
        cell = Cell(lattice_vectors=simple_lattice, atoms=sample_atoms, pbc_enabled=False)
        
        outside_position = np.array([5.0, -2.0, 4.5])
        result = cell.apply_periodic_boundary(outside_position)
        
        # 禁用PBC时位置不应该改变
        assert np.allclose(result, outside_position)
    
    def test_minimum_image_convention(self, sample_cell):
        """测试最小镜像约定"""
        # 测试各种位移向量
        displacements = [
            [1.0, 0.0, 0.0],   # 盒内
            [4.0, 0.0, 0.0],   # 需要镜像
            [-2.0, 1.0, 0.0],  # 负位移
            [0.0, 0.0, 5.0]    # z方向大位移
        ]
        
        for disp in displacements:
            min_disp = sample_cell.minimum_image(np.array(disp))
            
            # 最小镜像位移的分量应该在[-L/2, L/2]范围内
            box_lengths = sample_cell.get_box_lengths()
            assert np.all(np.abs(min_disp) <= box_lengths / 2 + 1e-10)
    
    def test_minimum_image_wrong_shape(self, sample_cell):
        """测试最小镜像约定错误形状"""
        with pytest.raises(ValueError, match="位移向量必须是3D向量"):
            sample_cell.minimum_image([1, 2])  # 只有2个分量


class TestCellPhysicalProperties:
    """测试物理属性计算"""
    
    def test_calculate_temperature_multi_atom(self, sample_cell):
        """测试多原子系统温度计算"""
        # 设置一些速度
        sample_cell.atoms[0].velocity = np.array([0.1, 0.2, 0.0])
        sample_cell.atoms[1].velocity = np.array([-0.1, 0.0, 0.1])
        sample_cell.atoms[2].velocity = np.array([0.0, -0.1, -0.1])
        
        temperature = sample_cell.calculate_temperature()
        
        # 温度应该为正数
        assert temperature >= 0
        # 有运动的原子系统温度应该大于0
        assert temperature > 0
    
    def test_calculate_temperature_zero_velocity(self, sample_cell):
        """测试零速度系统温度"""
        # 所有原子速度为零
        for atom in sample_cell.atoms:
            atom.velocity = np.zeros(3)
        
        temperature = sample_cell.calculate_temperature()
        assert temperature == 0.0
    
    def test_calculate_temperature_single_atom(self, simple_lattice):
        """测试单原子系统温度计算"""
        single_atom = [Atom(id=1, symbol='H', mass_amu=1.0, 
                           position=[0, 0, 0], velocity=[0.1, 0.1, 0.1])]
        cell = Cell(lattice_vectors=simple_lattice, atoms=single_atom)
        
        temperature = cell.calculate_temperature()
        assert temperature >= 0
    
    def test_com_velocity_calculation(self, sample_cell):
        """测试质心速度计算"""
        # 设置已知速度
        sample_cell.atoms[0].velocity = np.array([1.0, 0.0, 0.0])
        sample_cell.atoms[1].velocity = np.array([0.0, 1.0, 0.0])
        sample_cell.atoms[2].velocity = np.array([0.0, 0.0, 1.0])
        
        com_vel = sample_cell.get_com_velocity()
        assert len(com_vel) == 3
        # COM速度应该是质量加权平均
        assert np.all(np.isfinite(com_vel))
    
    def test_remove_com_motion(self, sample_cell):
        """测试质心运动移除"""
        # 设置一些速度
        sample_cell.atoms[0].velocity = np.array([1.0, 1.0, 1.0])
        sample_cell.atoms[1].velocity = np.array([2.0, 2.0, 2.0])
        sample_cell.atoms[2].velocity = np.array([3.0, 3.0, 3.0])
        
        sample_cell.remove_com_motion()
        
        # 移除质心运动后，质心速度应该接近零
        com_vel = sample_cell.get_com_velocity()
        assert np.allclose(com_vel, [0, 0, 0], atol=1e-10)
    
    def test_com_position_calculation(self, sample_cell):
        """测试质心位置计算"""
        com_pos = sample_cell.get_com_position()
        assert len(com_pos) == 3
        assert np.all(np.isfinite(com_pos))


class TestCellGetterMethods:
    """测试获取器方法"""
    
    def test_get_positions(self, sample_cell):
        """测试获取位置"""
        positions = sample_cell.get_positions()
        
        assert positions.shape == (len(sample_cell.atoms), 3)
        assert positions.dtype == np.float64
        
        # 检查与原子位置一致
        for i, atom in enumerate(sample_cell.atoms):
            assert np.allclose(positions[i], atom.position)
    
    def test_get_velocities(self, sample_cell):
        """测试获取速度"""
        velocities = sample_cell.get_velocities()
        
        assert velocities.shape == (len(sample_cell.atoms), 3)
        assert velocities.dtype == np.float64
        
        # 检查与原子速度一致
        for i, atom in enumerate(sample_cell.atoms):
            assert np.allclose(velocities[i], atom.velocity)
    
    def test_get_forces(self, sample_cell):
        """测试获取力"""
        forces = sample_cell.get_forces()
        
        assert forces.shape == (len(sample_cell.atoms), 3)
        assert forces.dtype == np.float64
        
        # 默认力应该为零
        assert np.allclose(forces, 0.0)
    
    def test_num_atoms_property(self, sample_cell):
        """测试原子数量属性"""
        assert sample_cell.num_atoms == len(sample_cell.atoms)


class TestCellCopy:
    """测试Cell拷贝功能"""
    
    def test_copy_basic(self, sample_cell):
        """测试基本深拷贝"""
        cell_copy = sample_cell.copy()
        
        # 检查基本属性
        assert np.allclose(cell_copy.lattice_vectors, sample_cell.lattice_vectors)
        assert cell_copy.pbc_enabled == sample_cell.pbc_enabled
        assert cell_copy.lattice_locked == sample_cell.lattice_locked
        assert len(cell_copy.atoms) == len(sample_cell.atoms)
    
    def test_copy_independence(self, sample_cell):
        """测试拷贝独立性"""
        cell_copy = sample_cell.copy()
        
        # 修改原始cell的晶格
        sample_cell.lattice_vectors[0, 0] = 999.0
        assert cell_copy.lattice_vectors[0, 0] != 999.0
        
        # 修改原始cell的原子位置
        sample_cell.atoms[0].move_by([10, 10, 10])
        assert not np.allclose(cell_copy.atoms[0].position, 
                              sample_cell.atoms[0].position)
    
    def test_copy_lattice_lock_state(self, sample_cell):
        """测试拷贝晶格锁定状态"""
        sample_cell.lock_lattice_vectors()
        cell_copy = sample_cell.copy()
        
        assert cell_copy.lattice_locked is True


class TestCellSupercell:
    """测试超胞构建"""
    
    def test_build_supercell_basic(self, sample_cell):
        """测试基本超胞构建"""
        repetition = (2, 2, 2)
        supercell = sample_cell.build_supercell(repetition)
        
        # 检查原子数量
        expected_atoms = len(sample_cell.atoms) * 2 * 2 * 2
        assert supercell.num_atoms == expected_atoms
        
        # 检查晶格向量
        expected_lattice = sample_cell.lattice_vectors * np.array([2, 2, 2])[:, np.newaxis]
        assert np.allclose(supercell.lattice_vectors, expected_lattice)
        
        # 检查体积
        expected_volume = sample_cell.volume * 8  # 2*2*2
        assert np.isclose(supercell.volume, expected_volume)
    
    def test_build_supercell_asymmetric(self, sample_cell):
        """测试非对称超胞"""
        repetition = (3, 1, 2)
        supercell = sample_cell.build_supercell(repetition)
        
        expected_atoms = len(sample_cell.atoms) * 3 * 1 * 2
        assert supercell.num_atoms == expected_atoms
        
        expected_volume = sample_cell.volume * 6  # 3*1*2
        assert np.isclose(supercell.volume, expected_volume)
    
    def test_build_supercell_atom_properties(self, sample_cell):
        """测试超胞原子属性保持"""
        repetition = (2, 1, 1)
        supercell = sample_cell.build_supercell(repetition)
        
        # 检查原子符号和质量保持不变
        original_symbols = [atom.symbol for atom in sample_cell.atoms]
        original_masses = [atom.mass_amu for atom in sample_cell.atoms]
        
        supercell_symbols = [atom.symbol for atom in supercell.atoms]
        supercell_masses = [atom.mass_amu for atom in supercell.atoms]
        
        # 检查总原子数
        repetition_factor = repetition[0] * repetition[1] * repetition[2]  # 2 * 1 * 1 = 2
        assert len(supercell.atoms) == len(sample_cell.atoms) * repetition_factor
        
        # 检查每种符号的原子数量
        from collections import Counter
        original_symbol_counts = Counter(original_symbols)
        supercell_symbol_counts = Counter(supercell_symbols)
        
        for symbol, original_count in original_symbol_counts.items():
            expected_count = original_count * repetition_factor
            assert supercell_symbol_counts[symbol] == expected_count


class TestCellEdgeCases:
    """测试边界情况"""
    
    def test_very_small_lattice(self, sample_atoms):
        """测试非常小的晶格"""
        tiny_lattice = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
        
        # 应该能构造，但可能会有警告
        cell = Cell(lattice_vectors=tiny_lattice, atoms=sample_atoms[:1])  # 只用一个原子
        assert cell.volume > 0
        assert cell.volume < 0.1  # 非常小的体积
    
    def test_large_lattice(self, sample_atoms):
        """测试大晶格"""
        large_lattice = np.array([
            [1000.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0],
            [0.0, 0.0, 1000.0]
        ])
        
        cell = Cell(lattice_vectors=large_lattice, atoms=sample_atoms)
        assert abs(cell.volume - 1e9) < 1e-6  # 1000^3，考虑浮点精度
    
    def test_triclinic_lattice(self, triclinic_lattice, sample_atoms):
        """测试三斜晶格"""
        cell = Cell(lattice_vectors=triclinic_lattice, atoms=sample_atoms)
        
        # 基本功能应该正常工作
        assert cell.volume > 0
        assert len(cell.get_positions()) == len(sample_atoms)
        
        # 周期性边界条件应该工作
        test_pos = np.array([5.0, 5.0, 5.0])
        wrapped = cell.apply_periodic_boundary(test_pos)
        assert len(wrapped) == 3