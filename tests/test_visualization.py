# tests/test_visualization.py

import pytest
from python.visualization import Visualizer
from python.structure import Cell, Atom
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")  # 使用无头后端，防止弹出图形窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 Axes3D
from matplotlib.collections import PathCollection  # 导入 PathCollection 类


def test_plot_cell_structure(tmp_path):
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

        # 检查是否绘制了正确数量的原子
        scatter = [
            child for child in ax.get_children() if isinstance(child, PathCollection)
        ]
        total_points = 0  # 记录所有散点的数量
        for path_collection in scatter:
            offsets = path_collection._offsets3d
            x_data, y_data, z_data = offsets
            total_points += len(x_data)

        # 检查总的原子散点数是否匹配
        assert total_points == len(
            atoms
        ), f"Expected {len(atoms)} atoms plotted, found {total_points}"

        # 保存图形到指定文件夹
        plot_path = tmp_path / "cell_structure.png"
        fig.savefig(plot_path)
        assert os.path.exists(plot_path), "Plot file was not saved successfully."

    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")


def test_plot_stress_strain():
    """
    @brief 测试 Visualizer.plot_stress_strain 函数，使用随机数据
    """
    strain_data = np.random.rand(10, 6)  # 生成随机应变数据
    stress_data = np.random.rand(10, 6)  # 生成随机应力数据

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    try:
        fig, ax = visualizer.plot_stress_strain(strain_data, stress_data, show=False)
        assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
        assert isinstance(ax, plt.Axes), "返回的对象不是 Matplotlib Axes"
    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")


def test_plot_stress_strain_with_zero_data():
    """
    @brief 测试 Visualizer.plot_stress_strain 函数，使用全零数据
    """
    strain_data = np.zeros((10, 6))  # 全零应变数据
    stress_data = np.zeros((10, 6))  # 全零应力数据

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    try:
        fig, ax = visualizer.plot_stress_strain(strain_data, stress_data, show=False)
        assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
        assert isinstance(ax, plt.Axes), "返回的对象不是 Matplotlib Axes"
    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")


def test_plot_stress_strain_with_mismatched_data():
    """
    @brief 测试 Visualizer.plot_stress_strain 函数，使用不匹配大小的数据
    """
    strain_data = np.random.rand(10, 6)  # 10x6 的应变数据
    stress_data = np.random.rand(8, 6)  # 8x6 的应力数据，不匹配

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    with pytest.raises(ValueError):
        visualizer.plot_stress_strain(strain_data, stress_data)


def test_plot_empty_cell_structure():
    """
    @brief 测试空晶胞结构的绘图行为
    """
    atoms = []  # 空原子列表
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(atoms=atoms, lattice_vectors=lattice_vectors)

    # 创建 Visualizer 实例
    visualizer = Visualizer()

    # 调用绘图函数，捕获图形对象
    fig, ax = visualizer.plot_cell_structure(cell, show=False)
    assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
    assert isinstance(ax, Axes3D), "返回的对象不是 Matplotlib Axes3D"

    # 检查没有绘制任何原子
    scatter = [
        child for child in ax.get_children() if isinstance(child, PathCollection)
    ]
    assert len(scatter) == 0, "在空晶胞结构中不应有任何原子绘制"
