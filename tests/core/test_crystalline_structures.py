#!/usr/bin/env python3
"""
测试晶体结构生成器模块
"""

import numpy as np
import pytest

from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder


class TestCrystallineStructureBuilder:
    """测试CrystallineStructureBuilder类"""

    def setup_method(self):
        """设置测试环境"""
        self.builder = CrystallineStructureBuilder()

    def test_fcc_aluminum_basic(self):
        """测试基本的铝FCC结构生成"""
        cell = self.builder.create_fcc("Al", 4.05, (2, 2, 2))

        # 验证原子数：4 * 2 * 2 * 2 = 32
        assert cell.num_atoms == 32

        # 验证所有原子都是Al
        for atom in cell.atoms:
            assert atom.symbol == "Al"
            assert atom.mass_amu == pytest.approx(26.9815)

    def test_fcc_copper_basic(self):
        """测试基本的铜FCC结构生成"""
        cell = self.builder.create_fcc("Cu", 3.615, (1, 1, 1))

        # 验证原子数：4 * 1 * 1 * 1 = 4
        assert cell.num_atoms == 4

        # 验证所有原子都是Cu
        for atom in cell.atoms:
            assert atom.symbol == "Cu"
            assert atom.mass_amu == pytest.approx(63.546)

    def test_fcc_lattice_vectors(self):
        """测试FCC结构的晶格矢量"""
        lattice_constant = 4.05
        supercell = (3, 2, 1)
        cell = self.builder.create_fcc("Al", lattice_constant, supercell)

        expected_vectors = np.diag(
            [
                lattice_constant * 3,  # nx
                lattice_constant * 2,  # ny
                lattice_constant * 1,  # nz
            ]
        )

        np.testing.assert_allclose(cell.lattice_vectors, expected_vectors, rtol=1e-10)

    def test_fcc_pbc_enabled(self):
        """测试FCC结构默认启用周期性边界条件"""
        cell = self.builder.create_fcc("Al", 4.05, (1, 1, 1))
        assert cell.pbc_enabled is True

    def test_invalid_element(self):
        """测试无效元素符号"""
        with pytest.raises(ValueError, match="不支持的元素"):
            self.builder.create_fcc("XX", 4.0, (1, 1, 1))

    def test_invalid_lattice_constant(self):
        """测试无效晶格常数"""
        with pytest.raises(ValueError, match="晶格常数必须为正数"):
            self.builder.create_fcc("Al", -4.0, (1, 1, 1))

        with pytest.raises(ValueError, match="晶格常数必须为正数"):
            self.builder.create_fcc("Al", 0, (1, 1, 1))

    def test_invalid_supercell(self):
        """测试无效超胞尺寸"""
        with pytest.raises(TypeError, match="超胞尺寸必须为正整数三元组"):
            self.builder.create_fcc("Al", 4.0, (1, 1))  # 不够3个元素

        with pytest.raises(TypeError, match="超胞尺寸必须为正整数三元组"):
            self.builder.create_fcc("Al", 4.0, (1, 1, 0))  # 包含0

        with pytest.raises(TypeError, match="超胞尺寸必须为正整数三元组"):
            self.builder.create_fcc("Al", 4.0, (1.5, 2, 3))  # 包含浮点数

    def test_get_supported_elements(self):
        """测试获取支持的元素列表"""
        elements = CrystallineStructureBuilder.get_supported_elements()

        # 验证包含常见元素
        assert "Al" in elements
        assert "Cu" in elements
        assert "Au" in elements
        assert "Fe" in elements

        # 验证返回列表类型
        assert isinstance(elements, list)
        assert len(elements) > 0

    def test_get_atomic_mass(self):
        """测试获取原子质量"""
        assert CrystallineStructureBuilder.get_atomic_mass("Al") == pytest.approx(
            26.9815
        )
        assert CrystallineStructureBuilder.get_atomic_mass("Cu") == pytest.approx(
            63.546
        )

        with pytest.raises(ValueError, match="不支持的元素"):
            CrystallineStructureBuilder.get_atomic_mass("XX")

    def test_bcc_not_implemented(self):
        """测试BCC结构未实现"""
        with pytest.raises(NotImplementedError, match="BCC结构生成将在后续版本中实现"):
            self.builder.create_bcc("Fe", 2.87, (1, 1, 1))

    def test_hcp_not_implemented(self):
        """测试HCP结构未实现"""
        with pytest.raises(NotImplementedError, match="HCP结构生成将在后续版本中实现"):
            self.builder.create_hcp("Zn", 2.66, (1, 1, 1))

    def test_fcc_positions_correctness(self):
        """测试FCC结构的原子位置正确性"""
        # 创建1x1x1的FCC结构进行详细验证
        cell = self.builder.create_fcc("Al", 4.0, (1, 1, 1))

        # 应该有4个原子
        assert cell.num_atoms == 4

        # 获取所有原子位置
        positions = [atom.position for atom in cell.atoms]

        # 期望的FCC位置 (直角坐标)
        expected_positions = np.array(
            [
                [0.0, 0.0, 0.0],  # 角原子
                [2.0, 2.0, 0.0],  # xy面心
                [2.0, 0.0, 2.0],  # xz面心
                [0.0, 2.0, 2.0],  # yz面心
            ]
        )

        # 转换为numpy数组便于比较
        actual_positions = np.array(positions)

        # 由于原子顺序可能不同，需要找到匹配
        for expected_pos in expected_positions:
            # 检查是否存在与期望位置非常接近的原子
            distances = np.linalg.norm(actual_positions - expected_pos, axis=1)
            min_distance = np.min(distances)
            assert min_distance < 1e-10, f"未找到期望位置 {expected_pos} 的原子"


class TestCrystallineStructureBuilderDiamond:
    """金刚石（diamond）结构相关测试"""

    def test_diamond_basic_unit_cell(self):
        a = 3.5656
        builder = CrystallineStructureBuilder()
        cell = builder.create_diamond("C", a, (1, 1, 1))
        assert cell.num_atoms == 8
        assert cell.pbc_enabled is True

        L = cell.lattice_vectors
        assert np.allclose(np.diag(L), [a, a, a], atol=1e-8)
        assert np.allclose(L - np.diag(np.diag(L)), 0.0, atol=1e-12)

    def test_diamond_supercell_222(self):
        a = 3.5656
        builder = CrystallineStructureBuilder()
        cell = builder.create_diamond("C", a, (2, 2, 2))
        assert cell.num_atoms == 64

    def test_diamond_nearest_neighbor_distance(self):
        a = 3.5656
        builder = CrystallineStructureBuilder()
        cell = builder.create_diamond("C", a, (1, 1, 1))
        d_ref = a * np.sqrt(3.0) / 4.0
        pos = cell.get_positions()
        d_min = 1e9
        for i in range(cell.num_atoms):
            for j in range(i + 1, cell.num_atoms):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < d_min:
                    d_min = d
        assert abs(d_min - d_ref) < 0.1
