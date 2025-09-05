#!/usr/bin/env python3
"""Pipeline common模块测试

测试pipeline共用工具函数的正确性，包括材料创建、势函数初始化等功能。
"""

import pytest

from thermoelasticsim.cli.pipelines.common import (
    build_cell,
    get_material_by_spec,
    make_potential,
)
from thermoelasticsim.core.structure import Cell
from thermoelasticsim.potentials.eam import EAMAl1Potential, EAMCu1Potential
from thermoelasticsim.potentials.tersoff import TersoffC1988Potential


class TestMaterialAndStructureCreation:
    """材料和结构创建测试"""

    def test_aluminum_fcc_creation(self):
        """测试铝FCC结构创建"""
        material = get_material_by_spec("Al", "fcc")
        cell = build_cell(material, (2, 2, 2))

        assert isinstance(cell, Cell)
        assert cell.num_atoms == 32  # 4原子/单胞 × 2×2×2
        assert cell.pbc_enabled

    def test_copper_fcc_creation(self):
        """测试铜FCC结构创建"""
        material = get_material_by_spec("Cu", "fcc")
        cell = build_cell(material, (3, 3, 3))

        assert isinstance(cell, Cell)
        assert cell.num_atoms == 108  # 4原子/单胞 × 3×3×3

    def test_carbon_diamond_creation(self):
        """测试碳金刚石结构创建"""
        material = get_material_by_spec("C", "diamond")
        cell = build_cell(material, (2, 2, 2))

        assert isinstance(cell, Cell)
        assert cell.num_atoms == 64  # 8原子/单胞 × 2×2×2

    def test_invalid_material_symbol(self):
        """测试无效材料符号"""
        with pytest.raises((ValueError, KeyError)):
            get_material_by_spec("Xx", "fcc")

    def test_invalid_structure_type(self):
        """测试无效结构类型"""
        # 对于C（需要明确结构），无效结构会抛出异常
        with pytest.raises(ValueError):
            get_material_by_spec("C", "invalid_structure")


class TestPotentialCreation:
    """势函数创建测试"""

    def test_eam_al1_potential_creation(self):
        """测试EAM Al1势函数创建"""
        potential = make_potential("EAM_Al1")

        assert isinstance(potential, EAMAl1Potential)
        assert potential.cutoff > 0

    def test_eam_cu1_potential_creation(self):
        """测试EAM Cu1势函数创建"""
        potential = make_potential("EAM_Cu1")

        assert isinstance(potential, EAMCu1Potential)
        assert potential.cutoff > 0

    def test_tersoff_c1988_potential_creation(self):
        """测试Tersoff C1988势函数创建"""
        potential = make_potential("Tersoff_C1988")

        assert isinstance(potential, TersoffC1988Potential)
        assert potential.cutoff > 0

    def test_invalid_potential_type(self):
        """测试无效势函数类型"""
        with pytest.raises((ValueError, KeyError)):
            make_potential("Invalid_Potential")

    def test_potential_type_case_insensitive(self):
        """测试势函数类型大小写不敏感"""
        # 应该支持大小写不敏感
        potential = make_potential("eam_al1")  # 小写应该成功
        assert isinstance(potential, EAMAl1Potential)


class TestIntegration:
    """集成测试"""

    def test_material_potential_compatibility(self):
        """测试材料与势函数兼容性"""
        # Al + EAM_Al1 应该兼容
        material = get_material_by_spec("Al", "fcc")
        cell = build_cell(material, (2, 2, 2))
        potential = make_potential("EAM_Al1")

        # 应该能够成功计算能量（不抛出异常）
        energy = potential.calculate_energy(cell)
        assert isinstance(energy, float)

        # Cu + EAM_Cu1 应该兼容
        material = get_material_by_spec("Cu", "fcc")
        cell = build_cell(material, (2, 2, 2))
        potential = make_potential("EAM_Cu1")

        energy = potential.calculate_energy(cell)
        assert isinstance(energy, float)

        # C + Tersoff_C1988 应该兼容
        material = get_material_by_spec("C", "diamond")
        cell = build_cell(material, (2, 2, 2))
        potential = make_potential("Tersoff_C1988")

        energy = potential.calculate_energy(cell)
        assert isinstance(energy, float)
