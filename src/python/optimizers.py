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

    def __init__(self, max_steps=10000, tol=1e-4, step_size=1e-2, energy_tol=1e-4):
        """
        初始化梯度下降优化器。

        @param max_steps 最大迭代步数。
        @param tol 力的收敛阈值。
        @param step_size 更新步长。
        @param energy_tol 能量变化的收敛阈值。
        """
        self.max_steps = max_steps
        self.tol = tol
        self.step_size = step_size
        self.energy_tol = energy_tol
        self.converged = False  # 收敛标志

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)
        atoms = cell.atoms
        potential.calculate_forces(cell)
        previous_energy = potential.calculate_energy(cell)

        for step in range(1, self.max_steps + 1):
            # 记录原子位置和力
            positions = cell.get_positions()
            forces = cell.get_forces()
            logger.debug(f"Step {step} - Atom positions: {positions}")
            logger.debug(f"Step {step} - Atom forces: {forces}")

            # 计算最大力
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            potential_energy = potential.calculate_energy(cell)
            total_energy = potential_energy  # 仅考虑势能

            # 日志记录
            logger.debug(
                f"GD Step {step}: Max force = {max_force:.6f} eV/Å, Total Energy = {total_energy:.6f} eV, Energy Change = {abs(total_energy - previous_energy):.6e} eV"
            )

            # 检查收敛条件
            energy_change = abs(total_energy - previous_energy)
            if max_force < self.tol and energy_change < self.energy_tol:
                logger.info(f"Gradient Descent converged after {step} steps.")
                self.converged = True
                break

            previous_energy = total_energy

            # 更新位置，沿着负梯度方向移动
            for atom in atoms:
                displacement = self.step_size * atom.force  # F = -dV/dx
                atom.position += displacement  # position += step_size * F = position - step_size * dV/dx
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

    def __init__(self, tol=1e-4, maxiter=10000):
        """
        初始化 BFGS 优化器。

        @param tol 梯度的收敛阈值。
        @param maxiter 最大迭代次数。
        """
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False  # 收敛标志

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)
        num_atoms = cell.num_atoms
        initial_positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()

        def objective(x):
            # 设置原子位置
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # 计算势能
            energy = potential.calculate_energy(cell)
            return energy

        def gradient(x):
            # 设置原子位置
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # 计算力
            potential.calculate_forces(cell)
            # 梯度是 -力
            grad = -np.array([atom.force for atom in cell.atoms]).flatten()
            return grad

        iteration = [0]  # 使用列表以便在回调中修改

        def callback(xk):
            iteration[0] += 1
            energy = objective(xk)
            grad = gradient(xk)
            grad_norm = np.linalg.norm(grad)
            logger.debug(
                f"BFGS Iteration {iteration[0]}: Energy = {energy:.6f} eV, Gradient Norm = {grad_norm:.6e} eV/Å"
            )

        result = minimize(
            objective,
            initial_positions,
            method="BFGS",
            jac=gradient,
            callback=callback,
            options={"gtol": self.tol, "disp": False, "maxiter": self.maxiter},
        )

        # 更新位置
        optimized_positions = result.x
        for i, atom in enumerate(cell.atoms):
            atom.position = optimized_positions[3 * i : 3 * i + 3]
        # 重新计算力
        potential.calculate_forces(cell)

        if result.success:
            logger.info(f"BFGS Optimization converged after {iteration[0]} iterations.")
        else:
            logger.warning(f"BFGS Optimization did not converge: {result.message}")

        self.converged = result.success


class LBFGSOptimizer(Optimizer):
    """
    @class LBFGSOptimizer
    @brief L-BFGS 优化器，使用 scipy.optimize.minimize 的 L-BFGS-B 方法实现
    """

    def __init__(self, tol=1e-4, maxiter=10000):
        """
        初始化 L-BFGS 优化器。

        @param tol 梯度的收敛阈值。
        @param maxiter 最大迭代次数。
        """
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False  # 收敛标志

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)
        num_atoms = cell.num_atoms
        initial_positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()

        def objective(x):
            # 设置原子位置
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # 计算势能
            energy = potential.calculate_energy(cell)
            return energy

        def gradient(x):
            # 设置原子位置
            for i, atom in enumerate(cell.atoms):
                atom.position = x[3 * i : 3 * i + 3]
            # 计算力
            potential.calculate_forces(cell)
            # 梯度是 -力
            grad = -np.array([atom.force for atom in cell.atoms]).flatten()
            return grad

        iteration = [0]  # 使用列表以便在回调中修改

        def callback(xk):
            iteration[0] += 1
            energy = objective(xk)
            grad = gradient(xk)
            grad_norm = np.linalg.norm(grad)
            logger.debug(
                f"L-BFGS Iteration {iteration[0]}: Energy = {energy:.6f} eV, Gradient Norm = {grad_norm:.6e} eV/Å"
            )

        result = minimize(
            objective,
            initial_positions,
            method="L-BFGS-B",
            jac=gradient,
            callback=callback,
            options={"gtol": self.tol, "disp": False, "maxiter": self.maxiter},
        )

        # 更新位置
        optimized_positions = result.x
        for i, atom in enumerate(cell.atoms):
            atom.position = optimized_positions[3 * i : 3 * i + 3]
        # 重新计算力
        potential.calculate_forces(cell)

        if result.success:
            logger.info(
                f"L-BFGS Optimization converged after {iteration[0]} iterations."
            )
        else:
            logger.warning(f"L-BFGS Optimization did not converge: {result.message}")

        self.converged = result.success
