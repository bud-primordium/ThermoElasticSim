# 文件名: optimizers.py
# 作者: Gilbert Young
# 修改日期: 2024-10-28
# 文件描述: 实现梯度下降和 BFGS 优化器。

"""
优化器模块

包含 GradientDescentOptimizer 和 BFGSOptimizer，用于分子动力学模拟优化
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class Optimizer:
    """
    优化器基类，定义优化方法的接口
    """

    def optimize(self, cell, potential):
        """执行优化，需子类实现"""
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    """
    带动量项的梯度下降优化器

    Parameters
    ----------
    max_steps : int
        最大迭代步数
    tol : float
        力的收敛阈值
    step_size : float
        更新步长
    energy_tol : float
        能量变化的收敛阈值
    beta : float
        动量项的系数，范围在 [0, 1)
    """

    def __init__(
        self, maxiter=10000, tol=1e-3, step_size=1e-3, energy_tol=1e-4, beta=0.9
    ):
        """初始化带动量项的梯度下降优化器"""
        self.maxiter = maxiter
        self.tol = tol
        self.step_size = step_size
        self.energy_tol = energy_tol
        self.beta = beta  # 动量项的系数
        self.converged = False  # 收敛标志
        self.trajectory = []  # 记录轨迹数据

    def optimize(self, cell, potential):
        """
        执行带动量项的梯度下降优化

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力和能量
        """
        logger = logging.getLogger(__name__)
        atoms = cell.atoms
        potential.calculate_forces(cell)
        previous_energy = potential.calculate_energy(cell)
        velocities = [np.zeros(3, dtype=np.float64) for _ in atoms]  # 初始化动量

        for step in range(1, self.maxiter + 1):
            positions = cell.get_positions()
            forces = cell.get_forces()
            volume = cell.volume
            lattice_vectors = cell.lattice_vectors.copy()

            # 记录当前状态
            self.trajectory.append(
                {
                    "step": step,
                    "positions": positions.copy(),
                    "volume": volume,
                    "lattice_vectors": lattice_vectors.copy(),
                }
            )

            logger.debug(f"Step {step} - Atom positions:\n{positions}")
            logger.debug(f"Step {step} - Atom forces:\n{forces}")

            # 计算最大力
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            potential_energy = potential.calculate_energy(cell)
            total_energy = potential_energy

            energy_change = abs(total_energy - previous_energy)
            logger.debug(
                f"GD with Momentum Step {step}: Max force = {max_force:.6f} eV/Å, "
                f"Total Energy = {total_energy:.6f} eV, Energy Change = {energy_change:.6e} eV"
            )

            # 检查收敛条件
            if max_force < self.tol and energy_change < self.energy_tol:
                logger.info(
                    f"Gradient Descent with Momentum converged after {step} steps."
                )
                self.converged = True
                break

            # 更新 previous_energy
            previous_energy = total_energy

            # 使用动量更新原子位置
            for i, atom in enumerate(atoms):
                # 更新动量项
                velocities[i] = self.beta * velocities[i] + (1 - self.beta) * atom.force
                displacement = self.step_size * velocities[i]
                atom.position += displacement
                # 应用周期性边界条件
                atom.position = cell.apply_periodic_boundary(atom.position)
                logger.debug(
                    f"Atom {atom.id} new position with momentum: {atom.position}"
                )

            # 重新计算力
            potential.calculate_forces(cell)

            # 检查原子间距离
            min_distance = np.inf
            num_atoms = len(atoms)
            min_pair = (-1, -1)
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    rij = atoms[j].position - atoms[i].position
                    if cell.pbc_enabled:
                        rij = cell.minimum_image(rij)
                    r = np.linalg.norm(rij)
                    if r < min_distance:
                        min_distance = r
                        min_pair = (atoms[i].id, atoms[j].id)
            logger.debug(
                f"Step {step}: Min distance = {min_distance:.3f} Å between atoms {min_pair[0]} and {min_pair[1]}"
            )

            if min_distance < 1.0:
                logger.error(
                    f"Step {step}: Minimum inter-atomic distance {min_distance:.3f} Å is too small between atoms {min_pair[0]} and {min_pair[1]}. Terminating optimization."
                )
                self.converged = False
                break
        else:
            logger.warning(
                "Gradient Descent with Momentum did not converge within the maximum number of steps."
            )
            self.converged = False

        # 输出最终原子位置
        final_positions = cell.get_positions()
        volume = cell.volume
        lattice_vectors = cell.lattice_vectors.copy()
        self.trajectory.append(
            {
                "step": self.maxiter + 1,
                "positions": final_positions.copy(),
                "volume": volume,
                "lattice_vectors": lattice_vectors.copy(),
            }
        )
        logger.debug(f"Gradient Descent Optimizer final positions:\n{final_positions}")

        return self.converged, self.trajectory


class BFGSOptimizer(Optimizer):
    """
    BFGS 优化器，基于 scipy.optimize.minimize

    Parameters
    ----------
    tol : float
        收敛阈值
    maxiter : int
        最大迭代步数
    """

    def __init__(self, tol=1e-6, maxiter=10000):
        """初始化 BFGS 优化器"""
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False
        self.trajectory = []  # 记录轨迹数据

    def optimize(self, cell, potential):
        """
        执行 BFGS 优化

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力和能量
        """
        logger = logging.getLogger(__name__)

        # 定义能量函数
        def energy_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            energy = potential.calculate_energy(cell)
            return energy

        # 定义梯度函数（力）
        def grad_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            potential.calculate_forces(cell)
            forces = cell.get_forces()
            return -forces.flatten()  # 修正符号为 -forces

        # 获取初始位置
        initial_positions = cell.get_positions().flatten()

        # 定义回调函数，在每次迭代结束后记录轨迹
        def callback(xk):
            positions = xk.reshape((-1, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            volume = cell.volume
            lattice_vectors = cell.lattice_vectors.copy()
            # 记录轨迹数据
            self.trajectory.append(
                {
                    "step": len(self.trajectory) + 1,
                    "positions": positions.copy(),
                    "volume": volume,
                    "lattice_vectors": lattice_vectors.copy(),
                }
            )
            logger.debug(f"Callback at iteration {len(self.trajectory)}")

        # 执行 BFGS 优化
        result = minimize(
            energy_fn,
            initial_positions,
            method="BFGS",
            jac=grad_fn,
            tol=self.tol,
            options={"maxiter": self.maxiter, "disp": False},
            callback=callback,  # 设置回调函数
        )

        if result.success:
            self.converged = True
            # 更新原子的位置
            optimized_positions = result.x.reshape((-1, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = optimized_positions[i]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            logger.info("BFGS Optimizer converged successfully.")
        else:
            self.converged = False
            logger.warning(f"BFGS Optimizer did not converge: {result.message}")

        # 在优化结束后，记录最终状态
        volume = cell.volume
        lattice_vectors = cell.lattice_vectors.copy()
        self.trajectory.append(
            {
                "step": len(self.trajectory) + 1,
                "positions": cell.get_positions().copy(),
                "volume": volume,
                "lattice_vectors": lattice_vectors.copy(),
            }
        )
        logger.debug(f"BFGS Optimizer final positions:\n{cell.get_positions()}")

        return self.converged, self.trajectory


class LBFGSOptimizer(Optimizer):
    """
    改进的 L-BFGS 优化器

    Parameters
    ----------
    ftol : float, optional
        函数值相对变化的收敛阈值，默认为 1e-6
    gtol : float, optional
        梯度（力）的收敛阈值，默认为 1e-5
    maxcor : int, optional
        存储的向量数量，默认为 10
    maxls : int, optional
        每次迭代中线搜索的最大步数，默认为 20
    maxiter : int, optional
        最大迭代步数，默认为 10000
    """

    def __init__(
        self,
        tol=1e-6,  # 防止有傻瓜两个都写
        ftol=1e-6,  # 保持与原来的 tol 一致
        gtol=1e-5,  # scipy 默认值
        maxcor=10,  # scipy 默认值
        maxls=20,  # scipy 默认值
        maxiter=10000,  # 保持与原来一致
    ):
        self.tol = tol
        self.ftol = ftol
        self.gtol = gtol
        self.maxcor = maxcor
        self.maxls = maxls
        self.maxiter = maxiter
        self.converged = False
        self.trajectory = []

    def optimize(self, cell, potential):
        """
        执行 L-BFGS 优化

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算作用力和能量

        Returns
        -------
        tuple
            (converged: bool, trajectory: list)
        """
        logger = logging.getLogger(__name__)

        # 定义能量函数
        def energy_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            energy = potential.calculate_energy(cell)
            return energy

        # 定义梯度函数（力）
        def grad_fn(positions):
            # 更新所有原子的位置
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i * 3 : (i + 1) * 3]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            potential.calculate_forces(cell)
            forces = cell.get_forces()
            return -forces.flatten()  # 修正符号为 -forces

        # 获取初始位置
        initial_positions = cell.get_positions().flatten()

        # 定义回调函数，在每次迭代结束后记录轨迹
        def callback(xk):
            positions = xk.reshape((-1, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            volume = cell.volume
            lattice_vectors = cell.lattice_vectors.copy()
            # 记录轨迹数据
            self.trajectory.append(
                {
                    "step": len(self.trajectory) + 1,
                    "positions": positions.copy(),
                    "volume": volume,
                    "lattice_vectors": lattice_vectors.copy(),
                }
            )
            logger.debug(f"Callback at iteration {len(self.trajectory)}")

        # 执行 L-BFGS-B 优化
        result = minimize(
            energy_fn,
            initial_positions,
            method="L-BFGS-B",
            tol=self.ftol,  # 替代之前未使用的 tol 参数
            jac=grad_fn,
            options={
                "ftol": self.ftol,  # 替代之前未使用的 tol 参数
                "gtol": self.gtol,  # 控制力的收敛
                "maxcor": self.maxcor,  # 控制内存使用
                "maxls": self.maxls,  # 控制线搜索
                "maxiter": self.maxiter,
                "disp": False,
            },
            callback=callback,
        )

        if result.success:
            self.converged = True
            # 更新原子的位置
            optimized_positions = result.x.reshape((-1, 3))
            for i, atom in enumerate(cell.atoms):
                atom.position = optimized_positions[i]
                # 应用 PBC
                atom.position = cell.apply_periodic_boundary(atom.position)
            logger.info("L-BFGS Optimizer converged successfully.")
        else:
            self.converged = False
            logger.warning(f"L-BFGS Optimizer did not converge: {result.message}")

        # 在优化结束后，记录最终状态
        volume = cell.volume
        lattice_vectors = cell.lattice_vectors.copy()
        self.trajectory.append(
            {
                "step": len(self.trajectory) + 1,
                "positions": cell.get_positions().copy(),
                "volume": volume,
                "lattice_vectors": lattice_vectors.copy(),
            }
        )
        logger.debug(f"L-BFGS Optimizer final positions:\n{cell.get_positions()}")

        return self.converged, self.trajectory
