# src/python/optimizers.py

import numpy as np
from scipy.optimize import minimize
import logging


class Optimizer:
    """
    @class Optimizer
    @brief 优化器基类
    """

    def optimize(self, cell, potential):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    """
    @class GradientDescentOptimizer
    @brief 梯度下降优化器
    """

    def __init__(self, max_steps=10000, tol=1e-6, step_size=1e-43):
        self.max_steps = max_steps
        self.tol = tol
        self.step_size = step_size
        self.converged = False  # 添加收敛标志

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)
        atoms = cell.atoms
        potential.calculate_forces(cell)
        for step in range(self.max_steps):
            # 计算最大力
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            kinetic_energy = sum(
                0.5 * atom.mass * np.linalg.norm(atom.velocity) ** 2 for atom in atoms
            )
            potential_energy = potential.calculate_energy(cell)
            total_energy = kinetic_energy + potential_energy

            logger.debug(
                f"GD Step {step}: Max force = {max_force:.6f} eV/Å, Total Energy = {total_energy:.6f} eV"
            )

            if max_force < self.tol:
                logger.info(f"Gradient Descent converged after {step} steps.")
                self.converged = True
                break
            # 更新 position
            for atom in atoms:
                displacement = self.step_size * atom.force
                atom.position -= displacement
                # 应用周期性边界条件
                atom.position = cell.apply_periodic_boundary(atom.position)
            # 重新计算力
            potential.calculate_forces(cell)
        else:
            logger.warning(
                "Gradient Descent Optimization did not converge within the maximum number of steps."
            )
            self.converged = False


class BFGSOptimizer(Optimizer):
    """
    @class BFGSOptimizer
    @brief BFGS 优化器，使用 scipy.optimize.minimize 实现
    """

    def __init__(self, tol=1e-6):
        self.tol = tol
        self.converged = False  # 添加收敛标志

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)
        num_atoms = cell.num_atoms
        initial_positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()

        def objective(x):
            # Set positions
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # Calculate potential energy
            energy = potential.calculate_energy(cell)
            return energy

        def gradient(x):
            # Set positions
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # Calculate forces
            potential.calculate_forces(cell)
            # Gradient is -force
            grad = -np.array([atom.force for atom in cell.atoms]).flatten()
            return grad

        iteration = [0]  # 使用列表以便在回调中修改

        def callback(xk):
            iteration[0] += 1
            energy = objective(xk)
            grad = gradient(xk)
            grad_norm = np.linalg.norm(grad)
            logger.debug(
                f"BFGS Iteration {iteration[0]}: Energy = {energy:.6f} eV, Gradient Norm = {grad_norm:.6f} eV/Å"
            )

        result = minimize(
            objective,
            initial_positions,
            method="BFGS",
            jac=gradient,
            callback=callback,
            options={"gtol": self.tol, "disp": False},  # 关闭内部显示
        )

        # Update positions after optimization
        optimized_positions = result.x
        for i, atom in enumerate(cell.atoms):
            atom.position = optimized_positions[3 * i : 3 * i + 3]
        # Recalculate forces
        potential.calculate_forces(cell)

        if result.success:
            logger.info(f"BFGS Optimization converged after {iteration[0]} iterations.")
        else:
            logger.warning(f"BFGS Optimization did not converge: {result.message}")

        self.converged = result.success
