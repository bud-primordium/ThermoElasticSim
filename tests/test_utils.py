# tests/test_utils.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.utils import TensorConverter


class TestUtils(unittest.TestCase):
    def test_tensor_to_voigt(self):
        # 对称张量
        tensor = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        voigt = TensorConverter.to_voigt(tensor)
        expected_voigt = np.array([1, 4, 6, 2, 5, 3])  # Sxx, Syy, Szz, Sxy, Syz, Szx
        np.testing.assert_array_almost_equal(voigt, expected_voigt)

    def test_voigt_to_tensor(self):
        # Voigt 表示
        voigt = np.array([1, 4, 6, 2, 5, 3])
        tensor = TensorConverter.from_voigt(voigt)
        expected_tensor = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        np.testing.assert_array_almost_equal(tensor, expected_tensor)

    def test_round_trip_conversion(self):
        # 随机对称张量
        tensor = np.array([[3, 1, 2], [1, 4, 5], [2, 5, 6]])
        voigt = TensorConverter.to_voigt(tensor)
        tensor_reconstructed = TensorConverter.from_voigt(voigt)
        np.testing.assert_array_almost_equal(tensor, tensor_reconstructed)


if __name__ == "__main__":
    unittest.main()
