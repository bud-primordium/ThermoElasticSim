#!/usr/bin/env python3
"""
测试ShearDeformationMethod类
"""

import numpy as np
import pytest

from thermoelasticsim.core import CrystallineStructureBuilder
from thermoelasticsim.elastic.deformation_method.zero_temp import (
    ShearDeformationMethod,
    StructureRelaxer,
)
from thermoelasticsim.potentials import EAMAl1Potential


class TestShearDeformationMethod:
    """测试ShearDeformationMethod类"""

    def setup_method(self):
        """设置测试环境"""
        # 创建小型测试系统
        builder = CrystallineStructureBuilder()
        self.cell = builder.create_fcc("Al", 4.05, (2, 2, 2))

        # 创建势函数
        self.potential = EAMAl1Potential()

        # 创建剪切形变方法
        self.shear_method = ShearDeformationMethod()

        # 创建弛豫器
        self.relaxer = StructureRelaxer(
            optimizer_type="L-BFGS",
            optimizer_params={"ftol": 1e-6, "gtol": 1e-6, "maxiter": 1000},
        )

    def test_initialization(self):
        """测试初始化"""
        method = ShearDeformationMethod()

        # 验证支持的方向
        assert 4 in method.supported_directions
        assert 5 in method.supported_directions
        assert 6 in method.supported_directions

        # 验证应力分量映射
        assert method.direction_to_stress_indices[4] == (1, 2)  # yz -> σ23
        assert method.direction_to_stress_indices[5] == (0, 2)  # xz -> σ13
        assert method.direction_to_stress_indices[6] == (0, 1)  # xy -> σ12

    def test_apply_shear_deformation_xy(self):
        """测试xy剪切形变"""
        strain_magnitude = 0.005
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction=6, strain_magnitude=strain_magnitude
        )

        # 验证晶胞没有被原地修改
        assert deformed_cell is not self.cell

        # 验证原子数量保持不变
        assert deformed_cell.num_atoms == self.cell.num_atoms

        # 验证晶格矢量变化（xy剪切应该改变lattice[1,0]）
        original_lattice = self.cell.lattice_vectors
        deformed_lattice = deformed_cell.lattice_vectors

        expected_change = strain_magnitude * original_lattice[1, 1]
        actual_change = deformed_lattice[1, 0] - original_lattice[1, 0]

        assert abs(actual_change - expected_change) < 1e-10

        # 验证原子位置变化（x坐标应该根据y坐标调整）
        original_pos = self.cell.get_positions()
        deformed_pos = deformed_cell.get_positions()

        for i in range(len(original_pos)):
            expected_x = original_pos[i, 0] + strain_magnitude * original_pos[i, 1]
            assert abs(deformed_pos[i, 0] - expected_x) < 1e-10
            # y和z坐标应该保持不变
            assert abs(deformed_pos[i, 1] - original_pos[i, 1]) < 1e-10
            assert abs(deformed_pos[i, 2] - original_pos[i, 2]) < 1e-10

    def test_apply_shear_deformation_yz(self):
        """测试yz剪切形变"""
        strain_magnitude = 0.003
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction=4, strain_magnitude=strain_magnitude
        )

        # 验证晶格矢量变化（yz剪切应该改变lattice[2,1]）
        original_lattice = self.cell.lattice_vectors
        deformed_lattice = deformed_cell.lattice_vectors

        expected_change = strain_magnitude * original_lattice[2, 2]
        actual_change = deformed_lattice[2, 1] - original_lattice[2, 1]

        assert abs(actual_change - expected_change) < 1e-10

    def test_apply_shear_deformation_xz(self):
        """测试xz剪切形变"""
        strain_magnitude = 0.002
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction=5, strain_magnitude=strain_magnitude
        )

        # 验证晶格矢量变化（xz剪切应该改变lattice[2,0]）
        original_lattice = self.cell.lattice_vectors
        deformed_lattice = deformed_cell.lattice_vectors

        expected_change = strain_magnitude * original_lattice[2, 2]
        actual_change = deformed_lattice[2, 0] - original_lattice[2, 0]

        assert abs(actual_change - expected_change) < 1e-10

    def test_invalid_direction(self):
        """测试无效的剪切方向"""
        with pytest.raises(ValueError, match="不支持的剪切方向"):
            self.shear_method.apply_shear_deformation(
                self.cell, direction=7, strain_magnitude=0.005
            )

        with pytest.raises(ValueError, match="不支持的剪切方向"):
            self.shear_method.apply_shear_deformation(
                self.cell, direction=1, strain_magnitude=0.005
            )

    def test_zero_strain(self):
        """测试零应变情况"""
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction=6, strain_magnitude=0.0
        )

        # 零应变时，晶格和原子位置应该保持不变
        original_lattice = self.cell.lattice_vectors
        deformed_lattice = deformed_cell.lattice_vectors

        np.testing.assert_array_almost_equal(
            original_lattice, deformed_lattice, decimal=12
        )

        original_pos = self.cell.get_positions()
        deformed_pos = deformed_cell.get_positions()

        np.testing.assert_array_almost_equal(original_pos, deformed_pos, decimal=12)

    def test_calculate_c44_response_basic(self):
        """测试基本的C44响应计算"""
        # 使用小的应变范围进行快速测试
        strain_amplitudes = np.array([-0.002, 0.0, 0.002])

        # 简化弛豫器设置以加快测试
        relaxer = StructureRelaxer(
            optimizer_type="L-BFGS",
            optimizer_params={
                "ftol": 1e-5,  # 放宽收敛条件加快测试
                "gtol": 1e-5,
                "maxiter": 100,
            },
        )

        results = self.shear_method.calculate_c44_response(
            self.cell, self.potential, strain_amplitudes, relaxer
        )

        # 验证返回结果的结构
        assert "directions" in results
        assert "detailed_results" in results
        assert "summary" in results
        assert "method" in results

        # 验证方向信息
        assert results["directions"] == [4, 5, 6]

        # 验证详细结果
        assert len(results["detailed_results"]) == 3
        for result in results["detailed_results"]:
            assert "direction" in result
            assert "name" in result
            assert "strains" in result
            assert "stresses" in result
            assert "elastic_constant" in result
            assert "r2_score" in result

        # 验证汇总结果
        summary = results["summary"]
        assert "C44" in summary
        assert "C55" in summary
        assert "C66" in summary
        assert "average_c44" in summary
        assert "converged_ratio" in summary

        # C44值应该在合理范围内（对于Al大约20-40 GPa）
        assert 10 < summary["C44"] < 80  # 宽松的范围，考虑小系统误差

    def test_shear_method_preserves_atoms(self):
        """测试剪切方法保持原子属性"""
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction=6, strain_magnitude=0.005
        )

        # 验证原子数量
        assert len(deformed_cell.atoms) == len(self.cell.atoms)

        # 验证原子属性
        position_changed_count = 0
        for i, (original, deformed) in enumerate(
            zip(self.cell.atoms, deformed_cell.atoms, strict=False)
        ):
            assert original.id == deformed.id
            assert original.symbol == deformed.symbol
            assert original.mass_amu == deformed.mass_amu

            # 统计位置改变的原子数量
            if not np.array_equal(original.position, deformed.position):
                position_changed_count += 1

        # 至少应该有一些原子的位置发生了变化
        # (原点处的原子在某些剪切中位置可能不变，这是正常的)
        assert position_changed_count > 0

    @pytest.mark.parametrize("direction", [4, 5, 6])
    def test_all_directions_work(self, direction):
        """测试所有剪切方向都能正常工作"""
        strain_magnitude = 0.001

        # 每个方向都应该能成功施加形变
        deformed_cell = self.shear_method.apply_shear_deformation(
            self.cell, direction, strain_magnitude
        )

        # 验证形变后的晶胞是有效的
        assert deformed_cell.num_atoms == self.cell.num_atoms
        assert deformed_cell.volume > 0
        assert deformed_cell.pbc_enabled == self.cell.pbc_enabled
