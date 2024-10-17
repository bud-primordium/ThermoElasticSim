# tests/test_utilities.py

"""
@file test_utilities.py
@brief 测试 utilities.py 模块中的 IOHandler 和 TensorConverter 类。
"""

import unittest
import numpy as np
import os
from python.utils import IOHandler, TensorConverter
from src.python.structure import Cell, Atom


class TestIOHandler(unittest.TestCase):
    """
    @class TestIOHandler
    @brief 测试 IOHandler 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.io_handler = IOHandler()
        self.test_filepath = "test_results.yaml"
        self.test_data = {
            "elastic_constants": [
                [10000, 0, 0, 0, 0, 0],
                [0, 10000, 0, 0, 0, 0],
                [0, 0, 10000, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        }

    def test_write_and_read_results(self) -> None:
        """
        @brief 测试写入和读取结果文件是否正确。
        """
        # 写入测试数据
        self.io_handler.write_results(self.test_filepath, self.test_data)
        self.assertTrue(os.path.exists(self.test_filepath))

        # 读取测试数据
        read_data = self.io_handler.read_structure(self.test_filepath)
        # 由于 read_structure 尚未实现，预期抛出 NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = self.io_handler.read_structure(self.test_filepath)

        # 清理
        os.remove(self.test_filepath)


class TestTensorConverter(unittest.TestCase):
    """
    @class TestTensorConverter
    @brief 测试 TensorConverter 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.voigt = np.array([1.0, 5.0, 9.0, 6.0, 7.0, 2.0])

    def test_to_voigt(self) -> None:
        """
        @brief 测试将张量转换为 Voigt 形式。
        """
        converted_voigt = TensorConverter.to_voigt(self.tensor)
        np.testing.assert_array_equal(converted_voigt, self.voigt)

    def test_from_voigt(self) -> None:
        """
        @brief 测试将 Voigt 形式转换回张量。
        """
        converted_tensor = TensorConverter.from_voigt(self.voigt)
        expected_tensor = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]])
        np.testing.assert_array_equal(converted_tensor, expected_tensor)


if __name__ == "__main__":
    unittest.main()
