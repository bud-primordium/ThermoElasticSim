# tests/test_simple_optimizer.py

import numpy as np
import logging
from python.optimizers import GradientDescentOptimizer


def test_gradient_descent_optimizer_simple(lj_potential, simple_cell):
    """
    测试梯度下降优化器，使用两个原子，无周期性边界条件。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        maxiter=10000, tol=1e-4, step_size=1e-2, energy_tol=1e-4
    )
    cell = simple_cell.copy()  # 使用深拷贝以避免修改原始晶胞

    optimizer.optimize(cell, lj_potential)

    assert (
        optimizer.converged
    ), "Gradient Descent Optimizer did not converge for simple system"

    energy = lj_potential.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Simple GD Optimizer - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Simple GD Optimizer - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {optimizer.tol}"

    # 检查原子间距离是否接近 sigma*2^(1/6)
    expected_distance = 2 ** (1 / 6) * lj_potential.parameters["sigma"]  # 约为2.04 Å
    actual_distance = np.linalg.norm(cell.atoms[1].position - cell.atoms[0].position)
    assert np.isclose(
        actual_distance, expected_distance, atol=1e-2
    ), f"Actual distance {actual_distance} is not close to expected {expected_distance}"
