# tests/test_nose_hoover.py

import pytest
import numpy as np
from python.thermostats import NoseHooverThermostat
from python.structure import Atom, Cell
from python.utils import AMU_TO_EVFSA2  # 确保正确导入
from conftest import generate_fcc_positions  # 从 conftest 导入


def test_nose_hoover_apply(nose_hoover_thermostat, two_atom_cell):
    """
    @brief 测试 Nose-Hoover 恒温器的 apply 方法。
    """
    # 初始化速度和力
    two_atom_cell.atoms[0].velocity = np.array([1.0, 0.0, 0.0])
    two_atom_cell.atoms[1].velocity = np.array([-1.0, 0.0, 0.0])
    two_atom_cell.atoms[0].force = np.array([0.0, 0.0, 0.0])
    two_atom_cell.atoms[1].force = np.array([0.0, 0.0, 0.0])

    # 应用恒温器
    dt = 1.0  # fs
    nose_hoover_thermostat.apply(two_atom_cell.atoms, dt)

    # 检查 xi 是否被更新
    assert nose_hoover_thermostat.xi[0] != 0.0, "xi 未被更新。"

    # 检查速度是否被更新（根据恒温器的实现，速度应被缩放）
    # 这里只检查速度是否发生变化
    assert not np.allclose(
        two_atom_cell.atoms[0].velocity, [1.0, 0.0, 0.0]
    ), "原子1的速度未被更新。"
    assert not np.allclose(
        two_atom_cell.atoms[1].velocity, [-1.0, 0.0, 0.0]
    ), "原子2的速度未被更新。"
