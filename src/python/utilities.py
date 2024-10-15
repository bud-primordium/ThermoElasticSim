# src/python/utilities.py

"""
@file utilities.py
@brief 提供辅助功能的模块，如文件读写和张量转换。
"""

import numpy as np
import yaml
from typing import Any, Dict


class IOHandler:
    """
    @class IOHandler
    @brief 处理文件输入输出的类。
    """

    def __init__(self) -> None:
        """
        @brief 初始化 IOHandler 实例。
        """
        pass

    def read_structure(self, filepath: str) -> "Cell":
        """
        @brief 从文件读取晶体结构数据。

        @param filepath 结构文件路径。
        @return Cell 实例。
        """
        # 实现读取结构文件的逻辑
        raise NotImplementedError("IOHandler.read_structure 尚未实现。")

    def write_results(self, filepath: str, data: Any) -> None:
        """
        @brief 将结果数据写入文件。

        @param filepath 结果文件路径。
        @param data 要写入的数据。
        """
        # 实现写入结果文件的逻辑
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)


class TensorConverter:
    """
    @class TensorConverter
    @brief 张量转换工具类。
    """

    @staticmethod
    def to_voigt(tensor: np.ndarray) -> np.ndarray:
        """
        @brief 将3x3张量转换为Voigt形式。

        @param tensor numpy.ndarray: 3x3张量。
        @return numpy.ndarray: Voigt形式的6元张量。
        """
        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2],
                tensor[0, 2],
                tensor[0, 1],
            ]
        )

    @staticmethod
    def from_voigt(voigt: np.ndarray) -> np.ndarray:
        """
        @brief 将Voigt形式的6元张量转换回3x3张量。

        @param voigt numpy.ndarray: Voigt形式的6元张量。
        @return numpy.ndarray: 3x3张量。
        """
        tensor = np.zeros((3, 3))
        tensor[0, 0] = voigt[0]
        tensor[1, 1] = voigt[1]
        tensor[2, 2] = voigt[2]
        tensor[1, 2] = tensor[2, 1] = voigt[3]
        tensor[0, 2] = tensor[2, 0] = voigt[4]
        tensor[0, 1] = tensor[1, 0] = voigt[5]
        return tensor
