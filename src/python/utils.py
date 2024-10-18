# src/python/utils.py

import numpy as np


class TensorConverter:
    """
    @class TensorConverter
    @brief 张量转换工具类。
    """

    @staticmethod
    def to_voigt(tensor):
        """
        @brief 将 3x3 张量转换为 Voigt 表示的 6 元素向量。
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
    @brief 数据收集工具类，用于收集模拟过程中的各种数据。
    """

    def __init__(self):
        self.data = []

    def collect(self, cell):
        """
        @brief 收集晶胞中的原子位置和速度。
        """
        positions = [atom.position.copy() for atom in cell.atoms]
        velocities = [atom.velocity.copy() for atom in cell.atoms]
        self.data.append({"positions": positions, "velocities": velocities})


# 添加单位转换常量
# 单位转换常量
# 1 amu = 104.3968445 # src/python/utils.py

import numpy as np


class TensorConverter:
    """
    @class TensorConverter
    @brief 张量转换工具类。
    """

    @staticmethod
    def to_voigt(tensor):
        """
        @brief 将 3x3 张量转换为 Voigt 表示的 6 元素向量。
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
    @brief 数据收集工具类，用于收集模拟过程中的各种数据。
    """

    def __init__(self):
        self.data = []

    def collect(self, cell):
        """
        @brief 收集晶胞中的原子位置和速度。
        """
        positions = [atom.position.copy() for atom in cell.atoms]
        velocities = [atom.velocity.copy() for atom in cell.atoms]
        self.data.append({"positions": positions, "velocities": velocities})


# 添加单位转换常量
# 单位转换常量
# 1 amu = 104.3968445 eV·fs²/Å²
AMU_TO_EVFSA2 = 104.3968445
