"""
Atom类的单元测试

测试覆盖：
- 构造器和初始化
- 属性访问和修改
- 核心操作方法
- 错误处理和边界情况
- 深拷贝功能
"""

import pytest
import numpy as np
from thermoelasticsim.core.structure import Atom


class TestAtomConstructor:
    """测试Atom构造器"""
    
    def test_basic_construction(self):
        """测试基本构造"""
        atom = Atom(id=1, symbol='H', mass_amu=1.008, position=[0, 0, 0])
        
        assert atom.id == 1
        assert atom.symbol == 'H'
        assert atom.mass_amu == 1.008
        assert np.allclose(atom.position, [0, 0, 0])
        assert np.allclose(atom.velocity, [0, 0, 0])
        assert np.allclose(atom.force, [0, 0, 0])
    
    def test_constructor_with_velocity(self):
        """测试带初始速度的构造"""
        velocity = [0.1, 0.2, 0.3]
        atom = Atom(id=1, symbol='O', mass_amu=15.999, 
                   position=[1, 2, 3], velocity=velocity)
        
        assert np.allclose(atom.velocity, velocity)
    
    def test_mass_conversion(self):
        """测试质量单位转换"""
        atom = Atom(id=1, symbol='H', mass_amu=1.008, position=[0, 0, 0])
        
        # 检查是否正确转换了质量单位
        assert atom.mass_amu == 1.008
        assert atom.mass > 0  # 转换后的质量应该为正
        # 简单检查转换是否合理（不需要知道具体转换常数）
        assert atom.mass != atom.mass_amu
    
    def test_position_array_conversion(self):
        """测试位置数组自动转换"""
        # 测试列表输入
        atom1 = Atom(id=1, symbol='H', mass_amu=1.0, position=[1, 2, 3])
        assert isinstance(atom1.position, np.ndarray)
        assert atom1.position.dtype == np.float64
        
        # 测试numpy数组输入
        pos_array = np.array([4, 5, 6])
        atom2 = Atom(id=2, symbol='O', mass_amu=16.0, position=pos_array)
        assert isinstance(atom2.position, np.ndarray)
        assert np.allclose(atom2.position, [4, 5, 6])


class TestAtomOperations:
    """测试Atom操作方法"""
    
    def test_move_by_basic(self, sample_atom):
        """测试基本位置移动"""
        original_pos = sample_atom.position.copy()
        displacement = np.array([0.1, 0.2, -0.1])
        
        sample_atom.move_by(displacement)
        
        expected_pos = original_pos + displacement
        assert np.allclose(sample_atom.position, expected_pos)
    
    def test_move_by_list_input(self, sample_atom):
        """测试列表输入的位置移动"""
        original_pos = sample_atom.position.copy()
        displacement = [0.5, -0.3, 0.1]
        
        sample_atom.move_by(displacement)
        
        expected_pos = original_pos + np.array(displacement)
        assert np.allclose(sample_atom.position, expected_pos)
    
    def test_move_by_wrong_shape(self, sample_atom):
        """测试错误形状的位移输入"""
        with pytest.raises(ValueError, match="位置增量必须是3D向量"):
            sample_atom.move_by([1, 2])  # 只有2个分量
        
        with pytest.raises(ValueError, match="位置增量必须是3D向量"):
            sample_atom.move_by([[1, 2, 3], [4, 5, 6]])  # 2x3矩阵
    
    def test_accelerate_by_basic(self, sample_atom):
        """测试基本速度变化"""
        original_vel = sample_atom.velocity.copy()
        velocity_change = np.array([0.05, -0.1, 0.02])
        
        sample_atom.accelerate_by(velocity_change)
        
        expected_vel = original_vel + velocity_change
        assert np.allclose(sample_atom.velocity, expected_vel)
    
    def test_accelerate_by_list_input(self, sample_atom):
        """测试列表输入的速度变化"""
        original_vel = sample_atom.velocity.copy()
        velocity_change = [-0.1, 0.2, -0.05]
        
        sample_atom.accelerate_by(velocity_change)
        
        expected_vel = original_vel + np.array(velocity_change)
        assert np.allclose(sample_atom.velocity, expected_vel)
    
    def test_accelerate_by_wrong_shape(self, sample_atom):
        """测试错误形状的速度变化输入"""
        with pytest.raises(ValueError, match="速度增量必须是3D向量"):
            sample_atom.accelerate_by([1, 2, 3, 4])  # 4个分量
        
        with pytest.raises(ValueError, match="速度增量必须是3D向量"):
            sample_atom.accelerate_by(np.array([[1], [2], [3]]))  # 错误形状


class TestAtomCopy:
    """测试Atom拷贝功能"""
    
    def test_copy_basic(self, sample_atom):
        """测试基本深拷贝功能"""
        atom_copy = sample_atom.copy()
        
        # 检查所有属性都正确复制
        assert atom_copy.id == sample_atom.id
        assert atom_copy.symbol == sample_atom.symbol
        assert atom_copy.mass_amu == sample_atom.mass_amu
        assert atom_copy.mass == sample_atom.mass
        assert np.allclose(atom_copy.position, sample_atom.position)
        assert np.allclose(atom_copy.velocity, sample_atom.velocity)
        assert np.allclose(atom_copy.force, sample_atom.force)
    
    def test_copy_independence(self, sample_atom):
        """测试拷贝的独立性"""
        atom_copy = sample_atom.copy()
        
        # 修改原子的位置
        sample_atom.move_by([1, 1, 1])
        
        # 检查拷贝不受影响
        assert not np.allclose(atom_copy.position, sample_atom.position)
        
        # 修改拷贝的速度
        atom_copy.accelerate_by([0.1, 0.1, 0.1])
        
        # 检查原子不受影响
        assert not np.allclose(atom_copy.velocity, sample_atom.velocity)
    
    def test_copy_array_independence(self):
        """测试数组独立性"""
        position1 = np.array([1.0, 2.0, 3.0])
        velocity1 = np.array([0.1, 0.2, 0.3])
        
        atom1 = Atom(id=1, symbol='H', mass_amu=1.0, 
                    position=position1, velocity=velocity1)
        atom2 = atom1.copy()
        
        # 直接修改数组
        atom1.position[0] = 99.0
        atom1.velocity[1] = 99.0
        
        # 检查拷贝不受影响
        assert atom2.position[0] != 99.0
        assert atom2.velocity[1] != 99.0


class TestAtomEdgeCases:
    """测试边界情况和错误处理"""
    
    def test_zero_mass(self):
        """测试零质量原子"""
        # 虽然物理上不合理，但构造器应该能处理
        atom = Atom(id=1, symbol='X', mass_amu=0.0, position=[0, 0, 0])
        assert atom.mass_amu == 0.0
    
    def test_negative_mass(self):
        """测试负质量原子"""
        # 物理上不合理，但构造器应该能处理
        atom = Atom(id=1, symbol='X', mass_amu=-1.0, position=[0, 0, 0])
        assert atom.mass_amu == -1.0
    
    def test_large_values(self):
        """测试大数值"""
        large_pos = [1e10, -1e10, 1e5]
        large_vel = [1e6, -1e6, 0]
        
        atom = Atom(id=999, symbol='Au', mass_amu=196.97, 
                   position=large_pos, velocity=large_vel)
        
        assert np.allclose(atom.position, large_pos)
        assert np.allclose(atom.velocity, large_vel)
    
    def test_nan_inf_handling(self):
        """测试NaN和无穷大值的处理"""
        # 这些值应该能被构造器接受，但会在使用时被发现
        atom = Atom(id=1, symbol='H', mass_amu=1.0, 
                   position=[np.nan, 0, 0])
        
        assert np.isnan(atom.position[0])
        
        # 无穷大值
        atom2 = Atom(id=2, symbol='O', mass_amu=16.0,
                    position=[np.inf, -np.inf, 0])
        
        assert np.isinf(atom2.position[0])
        assert np.isinf(atom2.position[1])
    
    def test_string_symbol_validation(self):
        """测试字符串符号"""
        # 正常符号
        atom1 = Atom(id=1, symbol='H', mass_amu=1.0, position=[0, 0, 0])
        assert atom1.symbol == 'H'
        
        # 多字符符号
        atom2 = Atom(id=2, symbol='Au', mass_amu=197.0, position=[0, 0, 0])
        assert atom2.symbol == 'Au'
        
        # 空字符串
        atom3 = Atom(id=3, symbol='', mass_amu=1.0, position=[0, 0, 0])
        assert atom3.symbol == ''


class TestAtomRepr:
    """测试Atom的字符串表示"""
    
    def test_basic_attributes_access(self, sample_atom):
        """测试基本属性访问"""
        # 这不是真正的repr测试，但确保所有属性都可以访问
        assert hasattr(sample_atom, 'id')
        assert hasattr(sample_atom, 'symbol')
        assert hasattr(sample_atom, 'mass_amu')
        assert hasattr(sample_atom, 'mass')
        assert hasattr(sample_atom, 'position')
        assert hasattr(sample_atom, 'velocity')
        assert hasattr(sample_atom, 'force')