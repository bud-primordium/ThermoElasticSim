# tests/test_visualization.py

import pytest
from python.visualization import Visualizer
from python.structure import Cell, Atom
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PathCollection
import os


def test_plot_cell_structure(tmp_path, two_atom_cell):
    """
    @brief 测试 Visualizer.plot_cell_structure 函数
    """
    # 创建 Visualizer 实例
    visualizer = Visualizer()

    # 调用绘图函数，捕获图形对象
    try:
        fig, ax = visualizer.plot_cell_structure(two_atom_cell, show=False)
        assert isinstance(fig, plt.Figure), "返回的对象不是 Matplotlib Figure"
        assert isinstance(ax, Axes3D), "返回的对象不是 Matplotlib Axes3D"

        # 检查是否绘制了正确数量的原子
        scatter = [
            child for child in ax.get_children() if isinstance(child, PathCollection)
        ]
        total_points = 0  # 记录所有散点的数量
        for path_collection in scatter:
            offsets = path_collection.get_offsets()
            if hasattr(path_collection, "_offsets3d"):
                x_data, y_data, z_data = path_collection._offsets3d
                total_points += len(x_data)
            else:
                total_points += len(offsets)

        # 检查总的原子散点数是否匹配
        assert total_points == len(
            two_atom_cell.atoms
        ), f"Expected {len(two_atom_cell.atoms)} atoms plotted, found {total_points}"

        # 保存图形到指定文件夹
        plot_path = tmp_path / "cell_structure.png"
        fig.savefig(plot_path)
        assert os.path.exists(plot_path), "Plot file was not saved successfully."

    except Exception as e:
        pytest.fail(f"绘图函数抛出异常: {e}")
