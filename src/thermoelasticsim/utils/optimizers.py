# 文件名: optimizers.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 提供多种优化算法用于原子结构优化

"""
分子动力学模拟优化器模块

提供多种优化算法用于原子结构优化，包括：
- GradientDescentOptimizer: 带动量项的梯度下降
- BFGSOptimizer: 基于scipy.optimize.minimize的BFGS
- LBFGSOptimizer: 改进的L-BFGS(有限内存BFGS)
"""

import logging

import numpy as np
from scipy.optimize import minimize

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.potentials import Potential

logger = logging.getLogger(__name__)


class Optimizer:
    """优化器抽象基类，定义优化方法的接口

    子类需要实现optimize()方法来执行具体的优化算法。

    Methods
    -------
    optimize(cell, potential)
        执行优化算法，需子类实现
    """

    def optimize(self, cell: Cell, potential: Potential) -> tuple[bool, list]:
        """执行优化算法

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能函数

        Returns
        -------
        tuple[bool, list]
            返回(是否收敛, 轨迹数据)

        Notes
        -----
        轨迹数据包含每步的:
        - 原子位置
        - 晶胞体积
        - 晶格矢量
        """
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    """带动量项的梯度下降优化器

    Parameters
    ----------
    maxiter : int, optional
        最大迭代步数，默认10000
    tol : float, optional
        力收敛阈值，默认1e-3 eV/Å
    step_size : float, optional
        步长，默认1e-3 Å
    energy_tol : float, optional
        能量变化阈值，默认1e-4 eV
    beta : float, optional
        动量系数[0,1)，默认0.9

    Attributes
    ----------
    converged : bool
        优化是否收敛的标志
    trajectory : list
        记录优化轨迹的列表
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

    def optimize(self, cell: Cell, potential: Potential) -> tuple[bool, list[dict]]:
        """执行带动量项的梯度下降优化

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能函数

        Returns
        -------
        tuple[bool, list[dict]]
            返回(是否收敛, 轨迹数据字典列表)

        Notes
        -----
        收敛条件:
        - 最大原子力 < tol
        - 能量变化 < energy_tol
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
    """BFGS准牛顿优化器

    Parameters
    ----------
    tol : float, optional
        收敛阈值，默认1e-6
    maxiter : int, optional
        最大迭代步数，默认10000

    Attributes
    ----------
    converged : bool
        优化是否收敛的标志
    trajectory : list
        记录优化轨迹的列表

    Notes
    -----
    使用scipy.optimize.minimize的BFGS实现
    适用于中等规模系统优化
    """

    def __init__(self, tol=1e-6, maxiter=10000):
        """初始化 BFGS 优化器"""
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False
        self.trajectory = []  # 记录轨迹数据

    def optimize(self, cell: Cell, potential: Potential) -> tuple[bool, list[dict]]:
        """执行 BFGS 优化

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能函数

        Returns
        -------
        tuple[bool, list[dict]]
            返回(是否收敛, 轨迹数据字典列表)
        """
        logger = logging.getLogger(__name__)

        # 采用分数坐标优化，避免跨PBC边界的不连续导致的精度损失
        lattice = cell.lattice_vectors.copy()

        def cartesian_from_frac(frac_coords):
            wrapped = frac_coords % 1.0
            return wrapped @ lattice

        # 定义能量函数（自变量为分数坐标）
        def energy_fn(frac_flat):
            frac = frac_flat.reshape(-1, 3)
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
            energy = potential.calculate_energy(cell)
            return energy

        # 定义梯度函数：dE/d(frac) = (dE/dpos) · lattice^T = -F · lattice^T
        def grad_fn(frac_flat):
            frac = frac_flat.reshape(-1, 3)
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
            potential.calculate_forces(cell)
            forces = cell.get_forces()  # (N,3)
            grad_frac = -(forces @ lattice.T)  # (N,3)
            return grad_frac.flatten()

        # 初始分数坐标
        positions0 = cell.get_positions()
        initial_frac = np.linalg.solve(lattice.T, positions0.T).T.flatten()

        # 定义回调函数，在每次迭代结束后记录轨迹
        def callback(xk):
            frac = xk.reshape((-1, 3))
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
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
            initial_frac,
            method="BFGS",
            jac=grad_fn,
            tol=self.tol,
            options={"maxiter": self.maxiter, "disp": False},
            callback=callback,  # 设置回调函数
        )

        if result.success:
            self.converged = True
            # 更新原子的位置（分数坐标→笛卡尔）
            optimized_frac = result.x.reshape((-1, 3))
            optimized_positions = cartesian_from_frac(optimized_frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = optimized_positions[i]
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


class CGOptimizer(Optimizer):
    """共轭梯度优化器（分数坐标），对剪切后的浅谷更稳健。

    Parameters
    ----------
    tol : float, optional
        梯度收敛阈值（映射到 scipy 的 gtol），默认 1e-6
    maxiter : int, optional
        最大迭代步数，默认 10000
    """

    def __init__(self, tol: float = 1e-6, maxiter: int = 10000):
        self.tol = tol
        self.maxiter = maxiter
        self.converged = False
        self.trajectory = []

    def optimize(self, cell: Cell, potential: Potential) -> tuple[bool, list[dict]]:
        """执行 CG 优化（变量为分数坐标）。"""
        logger = logging.getLogger(__name__)

        lattice = cell.lattice_vectors.copy()

        def cartesian_from_frac(frac_coords):
            wrapped = frac_coords % 1.0
            return wrapped @ lattice

        def energy_fn(frac_flat):
            frac = frac_flat.reshape(-1, 3)
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
            return potential.calculate_energy(cell)

        def grad_fn(frac_flat):
            frac = frac_flat.reshape(-1, 3)
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
            potential.calculate_forces(cell)
            forces = cell.get_forces()
            grad_frac = -(forces @ lattice.T)
            return grad_frac.flatten()

        positions0 = cell.get_positions()
        initial_frac = np.linalg.solve(lattice.T, positions0.T).T.flatten()

        def callback(xk):
            frac = xk.reshape((-1, 3))
            positions = cartesian_from_frac(frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = positions[i]
            self.trajectory.append(
                {
                    "step": len(self.trajectory) + 1,
                    "positions": positions.copy(),
                    "volume": cell.volume,
                    "lattice_vectors": cell.lattice_vectors.copy(),
                }
            )

        result = minimize(
            energy_fn,
            initial_frac,
            method="CG",
            jac=grad_fn,
            options={"maxiter": self.maxiter, "gtol": self.tol, "disp": False},
            callback=callback,
        )

        if result.success:
            self.converged = True
            optimized_frac = result.x.reshape((-1, 3))
            optimized_positions = cartesian_from_frac(optimized_frac)
            for i, atom in enumerate(cell.atoms):
                atom.position = optimized_positions[i]
            logger.info("CG Optimizer converged successfully.")
        else:
            self.converged = False
            logger.warning(f"CG Optimizer did not converge: {result.message}")

        return self.converged, self.trajectory


class LBFGSOptimizer(Optimizer):
    """L-BFGS优化器(有限内存BFGS)

    该优化器能够处理两种模式：
    1.  仅优化原子位置 (默认)。
    2.  同时优化原子位置和晶胞形状 (完全弛豫)。

    Parameters
    ----------
    ftol : float, optional
        函数收敛阈值，默认1e-8
    gtol : float, optional
        梯度收敛阈值，默认1e-6
    maxiter : int, optional
        最大迭代步数，默认1000
    **kwargs : dict, optional
        其他要传递给 `scipy.optimize.minimize` 的选项, 例如 `{'disp': True}`
    """

    def __init__(
        self, ftol=1e-8, gtol=1e-6, maxiter=1000, supercell_dims=None, **kwargs
    ):
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter
        self.supercell_dims = supercell_dims
        self.extra_options = kwargs
        self.converged = False
        self.trajectory = []

    def optimize(
        self, cell: Cell, potential: Potential, relax_cell: bool = False
    ) -> tuple[bool, list[dict]]:
        """执行 L-BFGS 优化。"""
        if relax_cell:
            return self._optimize_full(cell, potential)
        else:
            return self._optimize_positions_only(cell, potential)

    def _optimize_positions_only(
        self, cell: Cell, potential: Potential
    ) -> tuple[bool, list[dict]]:
        """仅优化原子位置。"""
        logger.debug("L-BFGS: 开始仅优化原子位置...")

        def energy_fn(positions_flat):
            cell.set_positions(positions_flat.reshape(-1, 3))
            return potential.calculate_energy(cell)

        def grad_fn(positions_flat):
            cell.set_positions(positions_flat.reshape(-1, 3))
            potential.calculate_forces(cell)
            return -cell.get_forces().flatten()

        initial_positions = cell.get_positions().flatten()

        # 记录固定的晶格参数用于显示
        fixed_lattice = cell.lattice_vectors
        fixed_a = np.linalg.norm(fixed_lattice[0])
        fixed_volume = cell.volume

        # 创建迭代记录器用于内部弛豫
        class PositionIterationLogger:
            def __init__(self, supercell_dims):
                self.niter = 0
                self.supercell_dims = supercell_dims

            def __call__(self, xk):
                self.niter += 1
                energy = energy_fn(xk)

                # 计算等效单胞参数用于显示
                if self.supercell_dims is not None:
                    equiv_a = fixed_a / self.supercell_dims[0]
                    equiv_volume = fixed_volume / (
                        self.supercell_dims[0]
                        * self.supercell_dims[1]
                        * self.supercell_dims[2]
                    )
                    logger.debug(
                        f"  内部弛豫迭代 {self.niter:4d}: Energy={energy:12.6f} eV (固定等效单胞 a={equiv_a:.6f} Å, V={equiv_volume:.2f} Å³)"
                    )
                else:
                    logger.debug(
                        f"  内部弛豫迭代 {self.niter:4d}: Energy={energy:12.6f} eV (固定 a={fixed_a:.6f} Å, V={fixed_volume:.2f} Å³)"
                    )

                # 每10步或前几步显示进度
                if self.niter <= 3 or self.niter % 10 == 0:
                    logger.info(
                        f"  内部弛豫进度: 第{self.niter}步, 能量={energy:.6f} eV"
                    )

        position_logger = PositionIterationLogger(self.supercell_dims)

        options = {
            "ftol": self.ftol,
            "gtol": self.gtol,
            "maxiter": self.maxiter,
            "disp": True,
        }
        options.update(self.extra_options)

        # 详细记录优化参数用于诊断
        logger.debug(f"L-BFGS优化参数诊断: {options}")

        result = minimize(
            energy_fn,
            initial_positions,
            method="L-BFGS-B",
            jac=grad_fn,
            options=options,
            callback=position_logger,
        )

        self.converged = result.success
        if self.converged:
            cell.set_positions(result.x.reshape(-1, 3))
            logger.info("L-BFGS (仅原子) 优化收敛。")
        else:
            # 原始信息，不加工
            logger.warning("L-BFGS (仅原子) 优化未收敛")
            logger.warning(f"  原始message: '{result.message}'")
            logger.warning(f"  success: {result.success}")
            logger.warning(f"  status: {result.status}")
            logger.warning(f"  nfev: {result.nfev}")
            logger.warning(f"  nit: {result.nit}")
            logger.warning(f"  fun: {result.fun}")
            if hasattr(result, "jac"):
                logger.warning(f"  jac norm: {np.linalg.norm(result.jac):.6e}")
            logger.warning(f"  完整result对象: {result}")

            # 记录maxls参数是否生效 - 检查实际传递的options字典
            logger.debug(f"  设置的maxls: {options.get('maxls', '未设置')}")
            logger.debug(f"  设置的maxfun: {options.get('maxfun', '未设置')}")
            logger.debug(f"  完整options: {options}")

        return self.converged, []

    def _optimize_full(
        self, cell: Cell, potential: Potential
    ) -> tuple[bool, list[dict]]:
        """同时优化原子位置和晶胞形状。"""
        logger.debug("L-BFGS: 开始完全弛豫 (原子+晶胞)...")

        initial_lattice = cell.lattice_vectors.copy()
        initial_volume = cell.volume
        initial_a = np.linalg.norm(initial_lattice[0])

        atom_info = [(atom.id, atom.symbol, atom.mass_amu) for atom in cell.atoms]
        num_atoms = cell.num_atoms

        def pack_variables(positions, lattice):
            frac_coords = np.linalg.solve(lattice.T, positions.T).T
            return np.concatenate([frac_coords.flatten(), lattice.flatten()])

        def unpack_variables(x):
            frac_coords = x[: 3 * num_atoms].reshape(num_atoms, 3)
            lattice = x[3 * num_atoms :].reshape(3, 3)
            positions = frac_coords @ lattice
            return positions, lattice, frac_coords

        def create_cell_from_variables(positions, lattice):
            atoms = [
                Atom(id=info[0], symbol=info[1], mass_amu=info[2], position=pos)
                for info, pos in zip(atom_info, positions, strict=False)
            ]
            return Cell(lattice, atoms)

        def energy_fn(x):
            positions, lattice, _ = unpack_variables(x)
            temp_cell = create_cell_from_variables(positions, lattice)
            return potential.calculate_energy(temp_cell)

        x0 = pack_variables(cell.get_positions(), cell.lattice_vectors)

        options = {"ftol": self.ftol, "gtol": self.gtol, "maxiter": self.maxiter}
        options.update(self.extra_options)

        # 创建一个类来存储迭代信息，用于回调
        class IterationLogger:
            def __init__(self, supercell_dims, ftol, initial_energy, initial_lattice):
                self.niter = 0
                self.supercell_dims = supercell_dims
                self.prev_energy = None
                self.prev_a = None
                self.ftol = ftol
                self.initial_energy = initial_energy
                if supercell_dims is not None:
                    self.initial_a = (
                        np.linalg.norm(initial_lattice[0]) / supercell_dims[0]
                    )
                else:
                    self.initial_a = np.linalg.norm(initial_lattice[0])
                logger.debug("优化迭代监控器初始化完成")

            def __call__(self, xk):
                self.niter += 1
                logger.debug(f"优化器回调函数被调用，第{self.niter}次")
                energy = energy_fn(xk)
                positions, lattice, _ = unpack_variables(xk)
                current_a = np.linalg.norm(lattice[0])
                # current_volume = np.abs(np.linalg.det(lattice))  # 体积监控，暂未使用

                # 计算相对于上一步的变化量
                energy_change_step = (
                    energy - self.prev_energy
                    if self.prev_energy is not None
                    else energy - self.initial_energy
                )
                # 计算相对于初始值的总变化量
                energy_change_total = energy - self.initial_energy

                # 计算等效单胞参数（如果是超胞的话）
                if self.supercell_dims is not None:
                    equiv_a = current_a / self.supercell_dims[0]
                    # equiv_volume = current_volume / (
                    #     self.supercell_dims[0]
                    #     * self.supercell_dims[1]
                    #     * self.supercell_dims[2]
                    # )  # 等效体积，暂未使用
                    a_change_step = (
                        equiv_a - self.prev_a
                        if self.prev_a is not None
                        else equiv_a - self.initial_a
                    )
                    a_change_total = equiv_a - self.initial_a

                    logger.debug(
                        f"  L-BFGS第{self.niter:2d}步: 能量={energy:12.6f} eV (步间Δ={energy_change_step:+.6f}, 总Δ={energy_change_total:+.6f})"
                    )
                    logger.debug(
                        f"           等效单胞 a={equiv_a:.6f} Å (步间Δ={a_change_step:+.6f}, 总Δ={a_change_total:+.6f})"
                    )

                    # 估算收敛状态
                    if self.niter > 1:
                        ftol_check = abs(energy_change_step) / max(abs(energy), 1e-12)
                        logger.debug(
                            f"           收敛检查: |步间ΔE|/|E| = {ftol_check:.2e} (目标: {self.ftol:.1e})"
                        )

                    self.prev_a = equiv_a
                else:
                    logger.debug(
                        f"  L-BFGS第{self.niter:2d}步: 能量={energy:12.6f} eV (Δ={energy_change_step:+.6f}), a={current_a:.6f} Å"
                    )

                # 每5步或前几步显示进度
                if self.niter <= 3 or self.niter % 5 == 0:
                    if self.supercell_dims is not None:
                        equiv_a = current_a / self.supercell_dims[0]
                        logger.info(
                            f"  优化进度: 第{self.niter}步, 能量={energy:.6f} eV (总Δ={energy_change_total:+.6f}), a={equiv_a:.6f} Å"
                        )
                    else:
                        logger.info(
                            f"  优化进度: 第{self.niter}步, 能量={energy:.6f} eV (Δ={energy_change_step:+.6f})"
                        )

                self.prev_energy = energy

        # 获取初始状态用于计算变化量
        initial_energy = energy_fn(x0)
        initial_positions, initial_lattice, _ = unpack_variables(x0)

        iteration_logger = IterationLogger(
            self.supercell_dims, self.ftol, initial_energy, initial_lattice
        )

        result = minimize(
            energy_fn,
            x0,
            method="L-BFGS-B",
            jac=None,
            options=options,
            callback=iteration_logger,
        )

        self.converged = result.success

        if self.converged:
            final_positions, final_lattice, _ = unpack_variables(result.x)
            cell.set_lattice_vectors(final_lattice)
            cell.set_positions(final_positions)

            final_volume = cell.volume
            final_a = np.linalg.norm(final_lattice[0])

            logger.info("L-BFGS (完全数值梯度) 优化收敛。")
            logger.info(f"晶格常数变化: {initial_a:.6f} → {final_a:.6f} Å")
            logger.info(f"体积变化: {initial_volume:.6f} → {final_volume:.6f} Å³")
            logger.info(
                f"相对变化: Δa/a = {(final_a - initial_a) / initial_a * 100:.3f}%, ΔV/V = {(final_volume - initial_volume) / initial_volume * 100:.3f}%"
            )
        else:
            logger.warning(f"L-BFGS (完全数值梯度) 优化未收敛: {result.message}")

        return self.converged, []
