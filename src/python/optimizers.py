# src/python/optimizers.py

import numpy as np


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
            # 更新位置
            for atom in atoms:
                displacement = self.step_size * atom.force
                atom.position += displacement
                # 应用周期性边界条件
                atom.position = cell.apply_periodic_boundary(atom.position)
                # print(f"Atom {atom.id} Position: {atom.position}")
            # 重新计算力
            potential.calculate_forces(cell)
            # 打印新力
            # for atom in atoms:
            #     print(f"Atom {atom.id} Force: {atom.force}")
        else:
            print("Optimization did not converge within the maximum number of steps.")
