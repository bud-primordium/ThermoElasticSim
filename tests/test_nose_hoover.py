# tests/test_nose_hoover.py

import pytest
import numpy as np
from python.interfaces.cpp_interface import CppInterface


@pytest.fixture
def nose_hoover_interface():
    """
    @fixture 创建 Nose-Hoover C++ 接口实例
    """
    return CppInterface("nose_hoover")


def test_nose_hoover(nose_hoover_interface):
    """
    @brief 测试 C++ 实现的 Nose-Hoover 恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([1.0, 1.0], dtype=np.float64)
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    xi = 0.0
    Q = 10.0
    target_temperature = 300.0

    # 调用 Nose-Hoover
    updated_xi = nose_hoover_interface.nose_hoover(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi,
        Q,
        target_temperature,
    )

    # 由于初始 xi=0，动能接近 0，xi 应该有变化
    assert isinstance(updated_xi, float)
    assert not np.isclose(updated_xi, xi)
