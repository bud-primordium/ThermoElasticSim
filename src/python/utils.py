# src/python/utils.py

import numpy as np


class TensorConverter:
    """
    @class TensorConverter
    @brief 用于在张量和 Voigt 表示之间转换的工具类。
    """

    @staticmethod
    def to_voigt(tensor):
        if tensor.shape != (3, 3):
            raise ValueError("输入张量必须是 3x3 矩阵。")
        if not np.allclose(tensor, tensor.T):
            raise ValueError("输入张量必须是对称的。")

        voigt = np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2] * 2,
                tensor[0, 2] * 2,
                tensor[0, 1] * 2,
            ]
        )
        return voigt

    @staticmethod
    def from_voigt(voigt):
        if voigt.shape != (6,):
            raise ValueError("Voigt 表示必须是长度为 6 的数组。")

        tensor = np.array(
            [
                [voigt[0], voigt[5] / 2, voigt[4] / 2],
                [voigt[5] / 2, voigt[1], voigt[3] / 2],
                [voigt[4] / 2, voigt[3] / 2, voigt[2]],
            ]
        )
        return tensor


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
