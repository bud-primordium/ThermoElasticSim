# 文件名: utils.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现张量转换工具类和数据收集类，并定义了一些常用的单位转换常量。

"""
工具模块

包含 TensorConverter 类用于张量与 Voigt 表示之间的转换，DataCollector 类用于收集模拟过程中的数据，及一些常用的单位转换常量
"""

import numpy as np


class TensorConverter:
    """
    张量转换工具类，用于 3x3 张量和 Voigt 表示的 6 元素向量之间的转换
    """

    @staticmethod
    def to_voigt(tensor):
        """
        将 3x3 张量转换为 Voigt 表示的 6 元素向量

        Parameters
        ----------
        tensor : numpy.ndarray
            形状为 (3, 3) 的张量

        Returns
        -------
        numpy.ndarray
            形状为 (6,) 的 Voigt 表示向量
        """
        voigt = np.array(
            [
                tensor[0, 0],  # xx
                tensor[1, 1],  # yy
                tensor[2, 2],  # zz
                tensor[1, 2],  # yz
                tensor[0, 2],  # xz
                tensor[0, 1],  # xy
            ]
        )
        return voigt

    @staticmethod
    def from_voigt(voigt):
        """
        将 Voigt 表示的 6 元素向量转换为 3x3 张量

        Parameters
        ----------
        voigt : numpy.ndarray
            形状为 (6,) 的 Voigt 表示向量

        Returns
        -------
        numpy.ndarray
            形状为 (3, 3) 的张量
        """
        tensor = np.array(
            [
                [voigt[0], voigt[5], voigt[4]],  # xx, xy, xz
                [voigt[5], voigt[1], voigt[3]],  # xy, yy, yz
                [voigt[4], voigt[3], voigt[2]],  # xz, yz, zz
            ]
        )
        return tensor


class DataCollector:
    """
    数据收集工具类，用于收集模拟过程中的原子位置和速度数据
    """

    def __init__(self):
        """
        初始化数据收集器
        """
        self.data = []

    def collect(self, cell):
        """
        收集晶胞中的原子位置信息和速度信息

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        """
        positions = [atom.position.copy() for atom in cell.atoms]  # 复制原子位置
        velocities = [atom.velocity.copy() for atom in cell.atoms]  # 复制原子速度
        # 保存到数据字典
        self.data.append({"positions": positions, "velocities": velocities})


# 单位转换常量
# 定义常见单位转换的常量，用于模拟中单位的转换

# 1 amu = 104.3968445 eV·fs²/Å²
AMU_TO_EVFSA2 = 104.3968445

# 1 eV/Å^3 = 160.21766208 GPa
EV_TO_GPA = 160.2176565
