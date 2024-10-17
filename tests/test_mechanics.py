# tests/test_mechanics.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.mechanics import (
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from python.utils import TensorConverter


class TestMechanics(unittest.TestCase):
    def test_tensor_converter(self):
        tensor = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        voigt = TensorConverter.to_voigt(tensor)
        expected_voigt = np.array([1, 4, 6, 2, 5, 3])  # Sxx, Syy, Szz, Sxy, Syz, Szx
        np.testing.assert_array_almost_equal(voigt, expected_voigt)

    def test_elastic_constants_solver(self):
        strains = [np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])]
        stresses = [np.array([100, 0, 0, 0, 0, 0])]
        solver = ElasticConstantsSolver()
        C = solver.solve(strains, stresses)
        expected_C = np.zeros((6, 6))
        expected_C[0, 0] = 100 / 0.01  # 10000
        np.testing.assert_array_almost_equal(C, expected_C)


if __name__ == "__main__":
    unittest.main()
