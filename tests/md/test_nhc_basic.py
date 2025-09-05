#!/usr/bin/env python3
"""
Nose-Hoover链恒温器基础功能测试

测试NoseHooverChainPropagator的基础功能：
1. Q参数自动计算验证
2. 链初始化正确性
3. 温度计算一致性
4. Suzuki-Yoshida系数验证
5. 基础传播功能

创建时间: 2025-08-19
基于: NoseHooverChainPropagator实现
目标: 验证NHC基础功能的正确性
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# 添加src路径
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.md.propagators import (
    FOURTH_ORDER_COEFFS,
    NoseHooverChainPropagator,
)
from thermoelasticsim.utils.utils import KB_IN_EV

# 导入测试辅助函数
from .test_helpers import apply_maxwell_velocities, create_fcc_aluminum


class TestSuzukiYoshidaCoefficients:
    """测试四阶Suzuki-Yoshida系数"""

    def test_coefficient_values(self):
        """验证系数精确值"""
        # 理论值
        w1_w3_expected = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        w2_expected = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

        assert abs(FOURTH_ORDER_COEFFS[0] - w1_w3_expected) < 1e-14
        assert abs(FOURTH_ORDER_COEFFS[1] - w2_expected) < 1e-14
        assert abs(FOURTH_ORDER_COEFFS[2] - w1_w3_expected) < 1e-14

    def test_coefficient_sum(self):
        """验证系数和为1"""
        coeff_sum = sum(FOURTH_ORDER_COEFFS)
        assert abs(coeff_sum - 1.0) < 1e-14

    def test_coefficient_count(self):
        """验证系数数量"""
        assert len(FOURTH_ORDER_COEFFS) == 3


class TestNoseHooverChainInitialization:
    """测试NoseHooverChainPropagator初始化"""

    def test_valid_initialization(self):
        """测试正常初始化"""
        nhc = NoseHooverChainPropagator(
            target_temperature=300.0, tdamp=100.0, tchain=3, tloop=1
        )

        assert nhc.target_temperature == 300.0
        assert nhc.tdamp == 100.0
        assert nhc.tchain == 3
        assert nhc.tloop == 1
        assert not nhc._initialized
        assert nhc.Q is None
        assert len(nhc.p_zeta) == 3
        assert len(nhc.zeta) == 3
        assert np.allclose(nhc.p_zeta, 0.0)
        assert np.allclose(nhc.zeta, 0.0)

    def test_invalid_temperature(self):
        """测试无效温度"""
        with pytest.raises(ValueError, match="目标温度必须为正数"):
            NoseHooverChainPropagator(target_temperature=-100.0)

        with pytest.raises(ValueError, match="目标温度必须为正数"):
            NoseHooverChainPropagator(target_temperature=0.0)

    def test_invalid_tdamp(self):
        """测试无效时间常数"""
        with pytest.raises(ValueError, match="时间常数必须为正数"):
            NoseHooverChainPropagator(target_temperature=300.0, tdamp=-50.0)

    def test_invalid_tchain(self):
        """测试无效链长"""
        with pytest.raises(ValueError, match="链长度必须≥1"):
            NoseHooverChainPropagator(target_temperature=300.0, tchain=0)

    def test_invalid_tloop(self):
        """测试无效循环数"""
        with pytest.raises(ValueError, match="循环次数必须≥1"):
            NoseHooverChainPropagator(target_temperature=300.0, tloop=0)


class TestQParameterCalculation:
    """测试Q参数计算"""

    def create_test_cell(self, num_atoms: int) -> Cell:
        """创建标准FCC铝测试晶胞"""
        if num_atoms <= 4:
            # 使用单胞FCC结构（4原子）
            cell = create_fcc_aluminum((1, 1, 1))
            # 如果只需要少于4个原子，保留前几个
            if num_atoms < 4:
                cell.atoms = cell.atoms[:num_atoms]
        elif num_atoms <= 32:
            # 使用2x2x2超胞（32原子）
            cell = create_fcc_aluminum((2, 2, 2))
            # 如果需要少于32个原子，保留前几个
            if num_atoms < 32:
                cell.atoms = cell.atoms[:num_atoms]
        else:
            # 使用更大的超胞
            target_size = int(np.ceil((num_atoms / 4) ** (1 / 3)))
            cell = create_fcc_aluminum((target_size, target_size, target_size))
            if num_atoms < len(cell.atoms):
                cell.atoms = cell.atoms[:num_atoms]

        # 设置合理的初始速度
        apply_maxwell_velocities(cell, temperature=50.0)

        print(f"✓ 创建测试晶胞: {len(cell.atoms)}原子FCC结构")

        return cell

    def test_single_atom_q_calculation(self):
        """测试单原子系统Q参数"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0, tdamp=50.0, tchain=2)
        cell = self.create_test_cell(1)

        nhc._initialize_Q_parameters(cell)

        # 单原子系统：N_f = 3
        N_f = 3
        kB_T = KB_IN_EV * 300.0
        expected_Q0 = N_f * kB_T * 50.0**2
        expected_Q1 = kB_T * 50.0**2

        assert nhc._initialized
        assert nhc._num_atoms_global == 1
        assert len(nhc.Q) == 2
        assert abs(nhc.Q[0] - expected_Q0) < 1e-12
        assert abs(nhc.Q[1] - expected_Q1) < 1e-12

    def test_multi_atom_q_calculation(self):
        """测试多原子系统Q参数"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0, tdamp=100.0, tchain=3)
        cell = self.create_test_cell(10)

        nhc._initialize_Q_parameters(cell)

        # 多原子系统：N_f = 3*N - 3（与温度统计一致，移除质心平动）
        N_f = 3 * 10 - 3
        kB_T = KB_IN_EV * 300.0
        expected_Q0 = N_f * kB_T * 100.0**2
        expected_Q_rest = kB_T * 100.0**2

        assert nhc._num_atoms_global == 10
        assert len(nhc.Q) == 3
        assert abs(nhc.Q[0] - expected_Q0) < 1e-12
        assert abs(nhc.Q[1] - expected_Q_rest) < 1e-12
        assert abs(nhc.Q[2] - expected_Q_rest) < 1e-12

    def test_q_parameter_scaling(self):
        """测试Q参数与温度和时间常数的缩放关系"""
        nhc1 = NoseHooverChainPropagator(target_temperature=300.0, tdamp=50.0)
        nhc2 = NoseHooverChainPropagator(target_temperature=600.0, tdamp=100.0)

        cell = self.create_test_cell(5)
        nhc1._initialize_Q_parameters(cell)
        nhc2._initialize_Q_parameters(cell)

        # Q应该与T和τ²成正比
        temp_ratio = 600.0 / 300.0  # 2倍
        tdamp_ratio = (100.0 / 50.0) ** 2  # 4倍
        expected_ratio = temp_ratio * tdamp_ratio  # 8倍

        actual_ratio = nhc2.Q[0] / nhc1.Q[0]
        assert abs(actual_ratio - expected_ratio) < 1e-12


class TestKineticEnergyCalculation:
    """测试动能计算"""

    def test_kinetic_energy_calculation(self):
        """测试瞬时动能计算"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0)

        # 创建测试系统
        lattice = np.eye(3) * 10.0

        # 添加已知速度的原子
        atom1 = Atom(id=1, symbol="Al", mass_amu=1.0, position=np.array([0, 0, 0]))
        atom1.velocity = np.array([1.0, 0.0, 0.0])  # v² = 1.0

        atom2 = Atom(id=2, symbol="Al", mass_amu=2.0, position=np.array([1, 0, 0]))
        atom2.velocity = np.array([0.0, 1.0, 1.0])  # v² = 2.0

        atoms = [atom1, atom2]
        cell = Cell(lattice, atoms)

        # 计算动能
        kinetic = nhc._calculate_instantaneous_kinetic_energy(cell)

        # 期望：考虑质量单位转换 AMU_TO_EVFSA2 = 104.3968445
        # atom1: 0.5 * (1.0 * 104.3968445) * 1.0 = 52.19842225
        # atom2: 0.5 * (2.0 * 104.3968445) * 2.0 = 208.793689
        # 总计: 52.19842225 + 208.793689 = 260.99211125
        from thermoelasticsim.utils.utils import AMU_TO_EVFSA2

        expected = 0.5 * (1.0 * AMU_TO_EVFSA2) * 1.0 + 0.5 * (2.0 * AMU_TO_EVFSA2) * 2.0
        assert abs(kinetic - expected) < 1e-12


class TestBasicPropagation:
    """测试基础传播功能"""

    def test_propagation_without_crash(self):
        """测试传播不会崩溃"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0, tdamp=50.0)

        # 创建小系统
        lattice = np.eye(3) * 5.0
        atom = Atom(id=1, symbol="Al", mass_amu=26.98, position=np.array([0, 0, 0]))
        atom.velocity = np.array([0.1, 0.1, 0.1])

        cell = Cell(lattice, [atom])

        # Mock势能计算
        cell.calculate_potential_energy = MagicMock(return_value=1.0)

        # 执行几步传播
        for _ in range(5):
            nhc.propagate(cell, dt=1.0)

        # 验证状态合理
        assert nhc._initialized
        assert nhc._total_steps == 5
        assert len(nhc._temp_history) == 5
        assert not np.any(np.isnan(nhc.p_zeta))
        assert not np.any(np.isinf(nhc.p_zeta))

    def test_statistics_collection(self):
        """测试统计信息收集"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0, tdamp=50.0, tchain=2)

        # 初始统计
        stats = nhc.get_statistics()
        assert stats["total_steps"] == 0
        assert stats["target_temperature"] == 300.0
        assert stats["tdamp"] == 50.0
        assert stats["tchain"] == 2
        assert stats["average_temperature"] == 0.0

        # 创建系统并运行
        lattice = np.eye(3) * 5.0
        atom = Atom(id=1, symbol="Al", mass_amu=26.98, position=np.array([0, 0, 0]))
        atom.velocity = np.array([0.1, 0.1, 0.1])
        cell = Cell(lattice, [atom])

        cell.calculate_potential_energy = MagicMock(return_value=1.0)

        # 运行几步
        for _ in range(3):
            nhc.propagate(cell, dt=1.0)

        # 检查统计
        stats = nhc.get_statistics()
        assert stats["total_steps"] == 3
        assert len(stats["current_p_zeta"]) == 2
        assert len(stats["current_zeta"]) == 2
        assert len(stats["Q_parameters"]) == 2
        assert stats["average_temperature"] > 0.0


class TestResetFunctionality:
    """测试重置功能"""

    def test_reset_statistics(self):
        """测试统计重置"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0)

        # 手动设置一些统计
        nhc._total_steps = 10
        nhc._temp_history = [300.0, 310.0, 295.0]
        nhc._conserved_energy_history = [1.0, 1.1, 0.9]

        nhc.reset_statistics()

        assert nhc._total_steps == 0
        assert len(nhc._temp_history) == 0
        assert len(nhc._conserved_energy_history) == 0

    def test_reset_thermostat_state(self):
        """测试恒温器状态重置"""
        nhc = NoseHooverChainPropagator(target_temperature=300.0, tchain=3)

        # 设置一些非零状态
        nhc.p_zeta = np.array([1.0, 2.0, 3.0])
        nhc.zeta = np.array([0.1, 0.2, 0.3])
        nhc._total_steps = 5

        nhc.reset_thermostat_state()

        assert np.allclose(nhc.p_zeta, 0.0)
        assert np.allclose(nhc.zeta, 0.0)
        assert nhc._total_steps == 0


def test_import_and_basic_usage():
    """测试导入和基本使用"""
    # 验证可以正常导入和创建
    nhc = NoseHooverChainPropagator(target_temperature=300.0)
    assert isinstance(nhc, NoseHooverChainPropagator)
    assert nhc.target_temperature == 300.0


if __name__ == "__main__":
    # 直接运行时执行基础测试
    print("运行Nose-Hoover链基础功能测试...")

    # 测试系数
    print("✓ 四阶Suzuki-Yoshida系数验证")
    assert abs(sum(FOURTH_ORDER_COEFFS) - 1.0) < 1e-14

    # 测试初始化
    print("✓ NHC初始化测试")
    nhc = NoseHooverChainPropagator(target_temperature=300.0, tdamp=50.0)
    assert nhc.tchain == 3  # 默认值

    # 测试Q参数计算
    print("✓ Q参数计算测试")
    lattice = np.eye(3) * 5.0
    atom = Atom(id=1, symbol="Al", mass_amu=26.98, position=np.array([0, 0, 0]))
    cell = Cell(lattice, [atom])

    nhc._initialize_Q_parameters(cell)
    assert nhc._initialized
    assert len(nhc.Q) == 3

    print("✅ 所有基础测试通过！")
    print(f"Q参数: Q[0]={nhc.Q[0]:.3e}, Q[1:]={nhc.Q[1]:.3e}")
