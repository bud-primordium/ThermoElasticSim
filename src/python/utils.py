# src/python/utils.py

import numpy as np


class TensorConverter:
    @staticmethod
    def to_voigt(tensor):
        """
        将 3x3 张量转换为 Voigt 表示的长度为6的向量。

        顺序为 [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]
        """
        voigt = np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2],
                tensor[0, 2],
                tensor[0, 1],
            ]
        )
        return voigt


class DataCollector:
    """
    @class DataCollector
    @brief 数据收集器，用于在模拟过程中收集数据。
    """

    def __init__(self):
        self.data = []

    def collect(self, cell):
        # 收集需要的数据
        positions = [atom.position.copy() for atom in cell.atoms]
        velocities = [atom.velocity.copy() for atom in cell.atoms]
        self.data.append({"positions": positions, "velocities": velocities})
