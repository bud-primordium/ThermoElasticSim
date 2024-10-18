# src/python/optimizers.py

import numpy as np
from scipy.optimize import minimize


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

    def __init__(self, max_steps=1000, tol=1e-6, step_size=1e-4):
        self.max_steps = max_steps
        self.tol = tol
        self.step_size = step_size

    def optimize(self, cell, potential):
        atoms = cell.atoms
        potential.calculate_forces(cell)
        for step in range(self.max_steps):
            # 计算最大力
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            print(f"Step {step}: Max force = {max_force}")
            if max_force < self.tol:
                print(f"Converged after {step} steps")
                break
            # 更新 position
            for atom in atoms:
                displacement = self.step_size * atom.force
                atom.position += displacement
                # 应用周期性边界条件
                atom.position = cell.apply_periodic_boundary(atom.position)
            # 重新计算力
            potential.calculate_forces(cell)
        else:
            print("Optimization did not converge within the maximum number of steps.")


class BFGSOptimizer(Optimizer):
    """
    @class BFGSOptimizer
    @brief BFGS 优化器，使用 scipy.optimize.minimize 实现
    """

    def __init__(self, tol=1e-6):
        self.tol = tol

    def optimize(self, cell, potential):
        num_atoms = len(cell.atoms)
        initial_positions = np.array([atom.position for atom in cell.atoms]).flatten()

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

        result = minimize(
            objective,
            initial_positions,
            method="BFGS",
            jac=gradient,
            options={"gtol": self.tol, "disp": True},
        )

        # Update positions after optimization
        optimized_positions = result.x
        for i, atom in enumerate(cell.atoms):
            atom.position = optimized_positions[3 * i : 3 * i + 3]
        # Recalculate forces
        potential.calculate_forces(cell)
