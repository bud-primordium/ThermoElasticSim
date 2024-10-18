# tests/test_visualization.py

import pytest
from python.visualization import Visualizer
from python.structure import Cell, Atom
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 使用无头后端，防止弹出图形窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 Axes3D


def test_plot_cell_structure():
    """
    @brief 测试 Visualizer.plot_cell_structure 函数
    """
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(
            id=1, mass=26.9815, position=np.array([2.55, 2.55, 2.55]), symbol="Al"
        ),  # sigma=2.55 Å
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(atoms=atoms, lattice_vectors=lattice_vectors)

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    # 调用绘图函数，捕获图形对象
    try:
        fig, ax = visualizer.plot_cell_structure(cell, show=False)
        assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
        assert isinstance(ax, Axes3D), "返回的对象不是 Matplotlib Axes3D"
    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")


def test_plot_stress_strain():
    """
    @brief 测试 Visualizer.plot_stress_strain 函数
    """
    # 创建应变和应力数据
    strain_data = np.random.rand(10, 6)  # 示例数据
    stress_data = np.random.rand(10, 6)  # 示例数据

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    # 调用绘图函数，确保不抛出异常
    try:
        fig, ax = visualizer.plot_stress_strain(strain_data, stress_data, show=False)
        assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
        assert isinstance(ax, plt.Axes), "返回的对象不是 Matplotlib Axes"
    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")
