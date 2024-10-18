# src/python/optimizers.py

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


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

    def __init__(self, max_steps=10000, tol=1e-3, step_size=1e-3, energy_tol=1e-4):
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
            logger.debug(f"Step {step} - Atom positions:\n{positions}")
            logger.debug(f"Step {step} - Atom forces:\n{forces}")

            # 计算最大力
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            potential_energy = potential.calculate_energy(cell)
            total_energy = potential_energy  # 仅考虑势能

            # 日志记录
            energy_change = abs(total_energy - previous_energy)
            logger.debug(
                f"GD Step {step}: Max force = {max_force:.6f} eV/Å, Total Energy = {total_energy:.6f} eV, Energy Change = {energy_change:.6e} eV"
            )

            # 检查收敛条件
            if max_force < self.tol and energy_change < self.energy_tol:
                logger.info(f"Gradient Descent converged after {step} steps.")
                self.converged = True
                break

            # 检查能量和力的有效性
            if np.isnan(total_energy) or np.isinf(total_energy):
                logger.error(
                    f"Energy is nan or inf at step {step}. Terminating optimization."
                )
                self.converged = False
                break
            if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
                logger.error(
                    f"Force contains nan or inf at step {step}. Terminating optimization."
                )
                self.converged = False
                break

            # 确保能量大体上递减
            if total_energy > 3 * abs(previous_energy + self.energy_tol):
                logger.warning(
                    f"Energy increased from {previous_energy:.6f} eV to {total_energy:.6f} eV at step {step}. Terminating optimization."
                )
                self.converged = False
                break

            previous_energy = total_energy

            # 更新位置，沿着负梯度方向移动
            for atom in atoms:
                displacement = self.step_size * atom.force  # F = -dV/dx
                atom.position -= displacement  # position += step_size * F = position - step_size * dV/dx
                # 应用周期性边界条件
                atom.position = cell.apply_periodic_boundary(atom.position)
                logger.debug(f"Atom {atom.id} new position: {atom.position}")

            # 重新计算力
            potential.calculate_forces(cell)

            # 检查原子间距离
            min_distance = np.inf
            num_atoms = len(atoms)
            min_pair = (-1, -1)  # 初始化最小距离原子对
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    rij = atoms[j].position - atoms[i].position
                    # 应用最小镜像规则
                    for dim in range(3):
                        rij[dim] -= (
                            round(rij[dim] / cell.lattice_vectors[dim, dim])
                            * cell.lattice_vectors[dim, dim]
                        )
                    r = np.linalg.norm(rij)
                    if r < min_distance:
                        min_distance = r
                        min_pair = (atoms[i].id, atoms[j].id)
            logger.debug(
                f"Step {step}: Min distance = {min_distance:.3f} Å between atoms {min_pair[0]} and {min_pair[1]}"
            )

            if min_distance < 0.8 * potential.sigma:
                logger.error(
                    f"Step {step}: Minimum inter-atomic distance {min_distance:.3f} Å is too small between atoms {min_pair[0]} and {min_pair[1]}. Terminating optimization."
                )
                self.converged = False
                break
        else:
            logger.warning(
                "Gradient Descent Optimization did not converge within the maximum number of steps."
            )
            self.converged = False


class BFGSOptimizer(Optimizer):
    """
    @class BFGSOptimizer
    @brief BFGS 优化器，基于 scipy.optimize.minimize
    """

    def __init__(self, tol=1e-6, maxiter=10000):
        """
        初始化 BFGS 优化器。

        @param tol 收敛容差。
        @param maxiter 最大迭代次数。
        """
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False

    def optimize(self, cell, potential):
        logger = logging.getLogger(__name__)

        # 定义能量函数
        def energy_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
            return potential.calculate_energy(cell)

        # 定义梯度函数（力）
        def grad_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
            potential.calculate_forces(cell)
            forces = cell.get_forces()
            return forces.flatten()

        # 获取初始位置
        initial_positions = cell.get_positions().flatten()

        # 执行 BFGS 优化
        result = minimize(
            energy_fn,
            initial_positions,
            method="BFGS",
            jac=grad_fn,
            tol=self.tol,
            options={"maxiter": self.maxiter, "disp": False},
        )

        if result.success:
            self.converged = True
            # 更新原子的位置
            optimized_positions = result.x.reshape((-1, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = optimized_positions[i]
            logger.info("BFGS Optimizer converged successfully.")
        else:
            self.converged = False
            logger.warning(f"BFGS Optimizer did not converge: {result.message}")
