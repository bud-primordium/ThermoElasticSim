#!/usr/bin/env python3
"""
测试材料参数配置模块
"""

import pytest

from thermoelasticsim.elastic.materials import (
    ALUMINUM_FCC,
    CARBON_DIAMOND,
    COPPER_FCC,
    GOLD_FCC,
    MaterialParameters,
    compare_elastic_constants,
    get_all_materials,
    get_material_by_symbol,
)


class TestMaterialParameters:
    """测试MaterialParameters数据类"""

    def test_aluminum_fcc_parameters(self):
        """测试预定义铝FCC参数"""
        assert ALUMINUM_FCC.name == "Aluminum"
        assert ALUMINUM_FCC.symbol == "Al"
        assert ALUMINUM_FCC.mass_amu == pytest.approx(26.9815)
        assert ALUMINUM_FCC.lattice_constant == pytest.approx(4.045)
        assert ALUMINUM_FCC.structure == "fcc"
        assert ALUMINUM_FCC.temperature == pytest.approx(0.0)

        # 验证弹性常数
        assert ALUMINUM_FCC.literature_elastic_constants["C11"] == pytest.approx(110)
        assert ALUMINUM_FCC.literature_elastic_constants["C12"] == pytest.approx(61)
        assert ALUMINUM_FCC.literature_elastic_constants["C44"] == pytest.approx(33)

    def test_copper_fcc_parameters(self):
        """测试预定义铜FCC参数"""
        assert COPPER_FCC.name == "Copper"
        assert COPPER_FCC.symbol == "Cu"
        assert COPPER_FCC.mass_amu == pytest.approx(63.546)
        assert COPPER_FCC.lattice_constant == pytest.approx(3.639)
        assert COPPER_FCC.structure == "fcc"

        # 验证弹性常数
        assert COPPER_FCC.literature_elastic_constants["C11"] == pytest.approx(175.0)
        assert COPPER_FCC.literature_elastic_constants["C12"] == pytest.approx(128.0)
        assert COPPER_FCC.literature_elastic_constants["C44"] == pytest.approx(84.0)

    def test_gold_fcc_parameters(self):
        """测试预定义金FCC参数"""
        assert GOLD_FCC.name == "Gold"
        assert GOLD_FCC.symbol == "Au"
        assert GOLD_FCC.mass_amu == pytest.approx(196.966569)
        assert GOLD_FCC.lattice_constant == pytest.approx(4.08)
        assert GOLD_FCC.structure == "fcc"
        assert GOLD_FCC.temperature == pytest.approx(300.0)

        # 验证弹性常数
        assert GOLD_FCC.literature_elastic_constants["C11"] == pytest.approx(192.0)
        assert GOLD_FCC.literature_elastic_constants["C12"] == pytest.approx(163.0)
        assert GOLD_FCC.literature_elastic_constants["C44"] == pytest.approx(41.5)

    def test_cubic_symmetry(self):
        """测试立方对称性"""
        for material in [ALUMINUM_FCC, COPPER_FCC, GOLD_FCC]:
            constants = material.literature_elastic_constants

            # 验证立方对称关系
            assert constants["C11"] == pytest.approx(constants["C22"])
            assert constants["C11"] == pytest.approx(constants["C33"])
            assert constants["C44"] == pytest.approx(constants["C55"])
            assert constants["C44"] == pytest.approx(constants["C66"])
            assert constants["C12"] == pytest.approx(constants["C13"])
            assert constants["C12"] == pytest.approx(constants["C23"])

    def test_elastic_moduli_calculation(self):
        """测试弹性模量计算"""
        # 使用铝参数测试
        al = ALUMINUM_FCC

        # 体积模量: K = (C11 + 2*C12) / 3
        expected_bulk = (110 + 2 * 61) / 3
        assert al.bulk_modulus == pytest.approx(expected_bulk, rel=1e-10)

        # 剪切模量: G = C44
        assert al.shear_modulus == pytest.approx(33, rel=1e-10)

        # 杨氏模量: E = 9*K*G / (3*K + G)
        K = expected_bulk
        G = 33
        expected_young = 9 * K * G / (3 * K + G)
        assert al.young_modulus == pytest.approx(expected_young, rel=1e-6)

        # 泊松比: ν = (3*K - 2*G) / (6*K + 2*G)
        expected_poisson = (3 * K - 2 * G) / (6 * K + 2 * G)
        assert al.poisson_ratio == pytest.approx(expected_poisson, rel=1e-6)

    def test_custom_material_creation(self):
        """测试创建自定义材料参数"""
        custom_material = MaterialParameters(
            name="CustomMetal",
            symbol="CM",
            mass_amu=100.0,
            lattice_constant=3.5,
            structure="fcc",
            literature_elastic_constants={
                "C11": 200.0,
                "C12": 100.0,
                "C44": 50.0,
            },
            temperature=300.0,
            description="测试材料",
        )

        assert custom_material.name == "CustomMetal"
        assert custom_material.symbol == "CM"
        assert custom_material.bulk_modulus == pytest.approx((200.0 + 2 * 100.0) / 3)
        assert custom_material.shear_modulus == pytest.approx(50.0)

    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试负数原子质量
        with pytest.raises(ValueError, match="原子质量必须为正数"):
            MaterialParameters(
                name="BadMaterial",
                symbol="BM",
                mass_amu=-10.0,
                lattice_constant=4.0,
                structure="fcc",
                literature_elastic_constants={"C11": 100, "C12": 50, "C44": 25},
            )

        # 测试负数晶格常数
        with pytest.raises(ValueError, match="晶格常数必须为正数"):
            MaterialParameters(
                name="BadMaterial",
                symbol="BM",
                mass_amu=10.0,
                lattice_constant=0.0,
                structure="fcc",
                literature_elastic_constants={"C11": 100, "C12": 50, "C44": 25},
            )

        # 测试不支持的晶体结构
        with pytest.raises(ValueError, match="不支持的晶体结构"):
            MaterialParameters(
                name="BadMaterial",
                symbol="BM",
                mass_amu=10.0,
                lattice_constant=4.0,
                structure="xyz",
                literature_elastic_constants={"C11": 100, "C12": 50, "C44": 25},
            )

        # 测试缺少弹性常数
        with pytest.raises(ValueError, match="缺少必要的弹性常数"):
            MaterialParameters(
                name="BadMaterial",
                symbol="BM",
                mass_amu=10.0,
                lattice_constant=4.0,
                structure="fcc",
                literature_elastic_constants={"C11": 100, "C12": 50},  # 缺少C44
            )


class TestMaterialUtilities:
    """测试材料工具函数"""

    def test_get_material_by_symbol(self):
        """测试根据符号获取材料"""
        al_material = get_material_by_symbol("Al")
        assert al_material is not None
        assert al_material.name == "Aluminum"

        cu_material = get_material_by_symbol("Cu")
        assert cu_material is not None
        assert cu_material.name == "Copper"

        au_material = get_material_by_symbol("Au")
        assert au_material is not None
        assert au_material.name == "Gold"

        # 测试不存在的材料
        unknown = get_material_by_symbol("XX")
        assert unknown is None

    def test_get_all_materials(self):
        """测试获取所有材料"""
        materials = get_all_materials()

        assert isinstance(materials, dict)
        assert len(materials) >= 3

        assert "Aluminum" in materials
        assert "Copper" in materials
        assert "Gold" in materials

        assert materials["Aluminum"] is ALUMINUM_FCC
        assert materials["Copper"] is COPPER_FCC
        assert materials["Gold"] is GOLD_FCC


class TestMaterialDiamond:
    """针对 Carbon (diamond) 材料的参数与工具函数补充测试"""

    def test_carbon_diamond_parameters(self):
        mat = CARBON_DIAMOND
        assert mat.symbol == "C"
        assert mat.structure == "diamond"
        assert 3.5 < mat.lattice_constant < 3.7
        for k in ("C11", "C12", "C44"):
            assert k in mat.literature_elastic_constants

    def test_carbon_diamond_cubic_symmetry(self):
        c = CARBON_DIAMOND.literature_elastic_constants
        assert c["C22"] == c["C11"]
        assert c["C33"] == c["C11"]
        assert c["C55"] == c["C44"]
        assert c["C66"] == c["C44"]
        assert c["C13"] == c["C12"]
        assert c["C23"] == c["C12"]

    def test_get_material_by_symbol_includes_carbon(self):
        mat = get_material_by_symbol("C")
        assert mat is not None and mat.name.startswith("Carbon")

    def test_get_all_materials_includes_carbon(self):
        mats = get_all_materials()
        assert any(name.startswith("Carbon") for name in mats)

    def test_compare_elastic_constants(self):
        """测试弹性常数比较"""
        al = ALUMINUM_FCC
        cu = COPPER_FCC

        comparison = compare_elastic_constants(al, cu)

        # 验证返回结构
        assert "material1" in comparison
        assert "material2" in comparison
        assert "absolute_differences" in comparison
        assert "relative_differences" in comparison
        assert "ratios" in comparison

        assert comparison["material1"] == "Aluminum"
        assert comparison["material2"] == "Copper"

        # 验证C11比较
        c11_diff = 175.0 - 110  # Cu - Al
        c11_rel_diff = c11_diff / 110 * 100
        c11_ratio = 175.0 / 110

        assert comparison["absolute_differences"]["C11"] == pytest.approx(c11_diff)
        assert comparison["relative_differences"]["C11"] == pytest.approx(c11_rel_diff)
        assert comparison["ratios"]["C11"] == pytest.approx(c11_ratio)

        # 验证C44比较
        c44_diff = 84.0 - 33
        c44_ratio = 84.0 / 33

        assert comparison["absolute_differences"]["C44"] == pytest.approx(c44_diff)
        assert comparison["ratios"]["C44"] == pytest.approx(c44_ratio)


class TestMaterialProperties:
    """测试材料物理性质"""

    def test_aluminum_properties_reasonable(self):
        """测试铝的物理性质合理性"""
        al = ALUMINUM_FCC

        # 验证弹性模量范围合理 (基于EAM Al1势函数)
        assert 70 < al.bulk_modulus < 90  # Al的体积模量约79.4 GPa
        assert 25 < al.shear_modulus < 35  # Al的剪切模量约33 GPa
        assert 75 < al.young_modulus < 90  # Al的杨氏模量约83.7 GPa (EAM计算值)
        assert 0.3 < al.poisson_ratio < 0.4  # Al的泊松比约0.35

    def test_copper_properties_reasonable(self):
        """测试铜的物理性质合理性"""
        cu = COPPER_FCC

        # 验证弹性模量范围合理 (基于EAM Cu1势函数)
        assert 135 < cu.bulk_modulus < 150  # Cu的体积模量约143.7 GPa
        assert 75 < cu.shear_modulus < 90  # Cu的剪切模量约84 GPa
        assert 200 < cu.young_modulus < 220  # Cu的杨氏模量约210.9 GPa (EAM计算值)
        assert 0.2 < cu.poisson_ratio < 0.3  # Cu的泊松比约0.255 (EAM计算值)

    def test_material_stiffness_ordering(self):
        """测试材料刚度排序"""
        al = ALUMINUM_FCC
        cu = COPPER_FCC

        # 铜应该比铝更硬
        assert cu.bulk_modulus > al.bulk_modulus
        assert cu.shear_modulus > al.shear_modulus
        assert cu.young_modulus > al.young_modulus

        # C11和C44应该遵循相同趋势
        assert (
            cu.literature_elastic_constants["C11"]
            > al.literature_elastic_constants["C11"]
        )
        assert (
            cu.literature_elastic_constants["C44"]
            > al.literature_elastic_constants["C44"]
        )


class TestMaterialTheoreticalDensity:
    """测试理论密度估算属性。"""

    def test_fcc_al_cu_density(self):
        # 立方晶胞估算，应与常识接近（允许一定误差）
        assert ALUMINUM_FCC.theoretical_density == pytest.approx(2.70, rel=0.05)
        assert COPPER_FCC.theoretical_density == pytest.approx(8.96, rel=0.05)

    def test_diamond_c_density(self):
        # 金刚石结构（8原子/常规立方晶胞）
        assert CARBON_DIAMOND.theoretical_density == pytest.approx(3.51, rel=0.05)
