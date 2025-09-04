# 文件名: utils.py
# 作者: Gilbert Young
# 修改日期: 2025-07-11
# 文件描述: 实现张量转换工具类和数据收集类，并定义了一些常用的单位转换常量。

"""
工具模块

包含 TensorConverter 类用于张量与 Voigt 表示之间的转换，DataCollector 类用于收集模拟过程中的数据，
及一些常用的单位转换常量
"""

import concurrent.futures as _cf
import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# 保持原有的单位转换常量
AMU_TO_EVFSA2: float = 104.3968445
"""原子质量单位 (amu) 转换为 eV·fs²/Å² 的系数。"""

EV_TO_GPA: float = 160.2176634
"""应力单位换算系数：eV/Å³ → GPa。"""

KB_IN_EV: float = 8.617332385e-5
"""玻尔兹曼常数 kB（单位 eV/K）。"""


class TensorConverter:
    """张量转换工具类，支持应力和应变张量的 Voigt 与 3x3 矩阵表示之间的相互转换。"""

    @staticmethod
    def to_voigt(tensor: np.ndarray, tensor_type: str, tol: float = 1e-8) -> np.ndarray:
        """
        将一个对称的 3x3 张量转换为 Voigt 表示的 6 元素向量。

        Parameters
        ----------
        tensor : np.ndarray
            形状为 (3, 3) 的张量。
        tensor_type : str
            张量类型，必须是 'stress' 或 'strain'。
        tol : float, optional
            用于检查张量对称性的容差。如果非对称分量差异大于此值，将记录警告。

        Returns
        -------
        np.ndarray
            形状为 (6,) 的 Voigt 表示向量。
        """
        if tensor.shape != (3, 3):
            raise ValueError(f"输入张量必须是 3x3 矩阵，但得到形状 {tensor.shape}")
        if tensor_type not in {"stress", "strain"}:
            raise ValueError(
                f"无效的张量类型 '{tensor_type}'，必须是 'stress' 或 'strain'"
            )

        # 检查对称性
        if not np.allclose(tensor, tensor.T, atol=tol):
            logger.warning("输入张量不对称。将使用其对称部分进行计算。")
            tensor = 0.5 * (tensor + tensor.T)

        factor = 2.0 if tensor_type == "strain" else 1.0

        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2] * factor,
                tensor[0, 2] * factor,
                tensor[0, 1] * factor,
            ]
        )

    @staticmethod
    def from_voigt(voigt: np.ndarray, tensor_type: str) -> np.ndarray:
        """
        将 Voigt 表示的 6 元素向量转换为 3x3 的对称张量。

        Parameters
        ----------
        voigt : np.ndarray
            形状为 (6,) 的 Voigt 向量。
        tensor_type : str
            张量类型，必须是 'stress' 或 'strain'。

        Returns
        -------
        np.ndarray
            形状为 (3, 3) 的对称张量。
        """
        if voigt.shape != (6,):
            raise ValueError(
                f"输入 Voigt 向量必须有 6 个元素，但得到 {voigt.shape[0]} 个"
            )
        if tensor_type not in {"stress", "strain"}:
            raise ValueError(
                f"无效的张量类型 '{tensor_type}'，必须是 'stress' 或 'strain'"
            )

        factor = 0.5 if tensor_type == "strain" else 1.0

        return np.array(
            [
                [voigt[0], voigt[5] * factor, voigt[4] * factor],
                [voigt[5] * factor, voigt[1], voigt[3] * factor],
                [voigt[4] * factor, voigt[3] * factor, voigt[2]],
            ]
        )


class DataCollector:
    """
    数据收集器，用于记录和分析模拟轨迹

    提供了系统状态的收集、统计和保存功能，支持异步数据处理和自动保存。

    Parameters
    ----------
    capacity : int, optional
        存储容量限制，None 表示无限制
    save_interval : int, optional
        数据自动保存间隔，默认为 1000 步
    use_threading : bool, optional
        是否使用线程进行数据处理，默认为 False
    output_file : str, optional
        数据保存的文件名，默认为 'trajectory.h5'
    """

    def __init__(
        self,
        capacity: int | None = None,
        save_interval: int = 1000,
        use_threading: bool = False,
        output_file: str = "trajectory.h5",
    ):
        self.data = []
        self._capacity = capacity
        self.save_interval = save_interval
        self.use_threading = use_threading
        self.output_file = output_file
        self._executor = _cf.ThreadPoolExecutor() if use_threading else None

        # 初始化统计数据
        self.stats = {
            "temperature": [],
            "potential_energy": [],
            "kinetic_energy": [],
            "total_energy": [],
            "volume": [],
            "pressure": [],
        }

        logger.debug(
            f"Initialized DataCollector with capacity={capacity}, "
            f"save_interval={save_interval}, use_threading={use_threading}"
        )

    def __del__(self):
        """确保线程池正确关闭"""
        if self._executor:
            self._executor.shutdown()
            logger.debug("Shut down thread executor")

    def collect(self, cell, potential=None):
        """
        收集完整的系统状态数据

        Parameters
        ----------
        cell : Cell
            当前的晶胞对象
        potential : Potential, optional
            势能对象，用于计算系统能量
        """
        if self.use_threading:
            self._executor.submit(self._collect_async, cell, potential)
        else:
            self._collect_sync(cell, potential)

    def _collect_sync(self, cell, potential=None):
        """同步收集数据"""
        try:
            state = self._create_state_dict(cell, potential)
            self._process_state(state)
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            raise

    def _collect_async(self, cell, potential=None):
        """异步收集数据"""
        try:
            state = self._create_state_dict(cell, potential)
            self._process_state(state)
        except Exception as e:
            logger.error(f"Error in async data collection: {e}")

    def _create_state_dict(self, cell, potential=None) -> dict:
        """创建系统状态字典"""
        state = {
            "step": len(self.data),
            "time": len(self.data) * cell.dt if hasattr(cell, "dt") else None,
            "positions": np.array([atom.position.copy() for atom in cell.atoms]),
            "velocities": np.array([atom.velocity.copy() for atom in cell.atoms]),
            "forces": np.array([atom.force.copy() for atom in cell.atoms]),
            "cell_vectors": cell.lattice_vectors.copy(),
            "volume": cell.volume,
            "temperature": cell.calculate_temperature(),
        }

        if potential:
            try:
                state["potential_energy"] = potential.calculate_energy(cell)
            except Exception as e:
                logger.warning(f"Could not calculate potential energy: {e}")

        return state

    def _process_state(self, state: dict):
        """处理并存储状态数据"""
        if self._capacity and len(self.data) >= self._capacity:
            self.data.pop(0)

        self.data.append(state)
        self._update_stats(state)

        # 检查是否需要自动保存
        if len(self.data) % self.save_interval == 0:
            self.save_trajectory()

    def _update_stats(self, state: dict):
        """更新统计数据"""
        for key in self.stats:
            if key in state:
                self.stats[key].append(state[key])

    def save_trajectory(self):
        """将轨迹数据保存到 HDF5 文件"""
        try:
            with h5py.File(self.output_file, "w") as f:
                # 保存轨迹数据
                for i, state in enumerate(self.data):
                    grp = f.create_group(f"step_{i}")
                    for key, value in state.items():
                        if isinstance(value, np.ndarray):
                            grp.create_dataset(key, data=value)
                        else:
                            grp.attrs[key] = value

                # 保存统计数据
                stats_grp = f.create_group("statistics")
                for key, value in self.stats.items():
                    if value:  # 只保存非空数据
                        stats_grp.create_dataset(key, data=np.array(value))

            logger.info(f"Trajectory saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")
            raise


class NeighborList:
    """
    邻居列表类，用于生成和维护原子的邻居列表

    提供高效的邻居搜索和更新功能，支持周期性边界条件。

    Parameters
    ----------
    cutoff : float
        截断半径，单位为 Å
    skin : float, optional
        皮肤厚度，默认为 0.3 Å
    """

    def __init__(self, cutoff: float, skin: float = 0.3):
        if not isinstance(cutoff, int | float) or cutoff <= 0:
            raise ValueError("Cutoff must be a positive number")
        if not isinstance(skin, int | float) or skin < 0:
            raise ValueError("Skin must be non-negative")

        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.cutoff_with_skin = cutoff + skin
        self.neighbor_list = None
        self.last_positions = None
        self.cell = None
        self.last_update_step = 0

        # 性能统计
        self._build_count = 0
        self._update_count = 0
        self._last_build_time = 0

        logger.debug(f"Initialized NeighborList with cutoff={cutoff}, skin={skin}")

    def get_neighbor_stats(self) -> dict:
        """
        返回邻居列表的统计信息

        Returns
        -------
        dict
            包含最小、最大、平均邻居数等统计信息的字典
        """
        if not self.neighbor_list:
            return {}

        neighbor_counts = [len(neighbors) for neighbors in self.neighbor_list]
        return {
            "min_neighbors": min(neighbor_counts),
            "max_neighbors": max(neighbor_counts),
            "avg_neighbors": np.mean(neighbor_counts),
            "std_neighbors": np.std(neighbor_counts),
            "build_count": self._build_count,
            "update_count": self._update_count,
            "last_build_time": self._last_build_time,
        }

    def _validate_cutoff(self, box_size: np.ndarray):
        """验证截断半径的合理性"""
        min_box_length = np.min(box_size)
        if self.cutoff_with_skin > min_box_length / 2:
            logger.warning(
                f"Cutoff radius ({self.cutoff_with_skin:.3f}) is too large "
                f"compared to box size ({min_box_length / 2:.3f})"
            )
            # 禁用自动调整以保持cutoff一致性（修复C44计算问题）
            logger.warning(
                "Cutoff adjustment disabled to maintain consistency with EAM potential. "
                "This may affect neighbor list accuracy but preserves virial theorem."
            )
            # 注释掉自动调整代码
            # self.cutoff = min_box_length / 3
            # self.cutoff_with_skin = self.cutoff + self.skin
            # logger.info(f"Adjusted cutoff radius to {self.cutoff:.3f}")

    def _compute_optimal_grid_size(self, box_size: np.ndarray, num_atoms: int) -> float:
        """计算最优网格大小"""
        volume = np.prod(box_size)
        density = num_atoms / volume
        mean_atomic_spacing = (1 / density) ** (1 / 3)
        return max(self.cutoff + 0.5, mean_atomic_spacing)

    def build(self, cell):
        """构建邻居列表"""
        try:
            import time

            start_time = time.time()

            positions = cell.get_positions()
            num_atoms = cell.num_atoms
            box_size = cell.get_box_lengths()
            cutoff = self.cutoff_with_skin
            self.cell = cell

            self._validate_cutoff(box_size)

            # 根据系统大小选择构建方法
            if num_atoms < 64:
                self._build_brute_force(cell, positions, num_atoms, cutoff)
            else:
                self._build_with_grid(cell, positions, num_atoms, box_size, cutoff)

            self.last_positions = positions.copy()
            self._build_count += 1
            self._last_build_time = time.time() - start_time

            logger.debug(
                f"Built neighbor list for {num_atoms} atoms in {self._last_build_time:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error building neighbor list: {e}")
            raise

    def _build_brute_force(self, cell, positions, num_atoms, cutoff):
        """使用双重循环构建小系统的邻居列表。"""
        self.neighbor_list = [[] for _ in range(num_atoms)]
        cutoff_squared = cutoff**2

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                rij = positions[j] - positions[i]
                # 应用最小镜像原则
                if cell.pbc_enabled:
                    rij = cell.minimum_image(rij)
                distance_squared = np.dot(rij, rij)
                if distance_squared < cutoff_squared:
                    self.neighbor_list[i].append(j)
                    self.neighbor_list[j].append(i)

    def _build_with_grid(self, cell, positions, num_atoms, box_size, cutoff):
        """改进的基于网格的邻居列表构建。"""
        # 验证截断半径
        self._validate_cutoff(box_size)

        # 计算最优网格大小
        grid_size = self._compute_optimal_grid_size(box_size, num_atoms)
        grid_dims = np.maximum(np.floor(box_size / grid_size).astype(int), 1)

        # 初始化网格和邻居列表
        grid = {}
        self.neighbor_list = [[] for _ in range(num_atoms)]
        cutoff_squared = cutoff * cutoff

        # 将原子分配到网格
        for idx, pos in enumerate(positions):
            grid_idx = tuple(((pos / grid_size) % grid_dims).astype(int))
            grid.setdefault(grid_idx, []).append(idx)

        # 构建邻居列表
        for grid_idx, atom_indices in grid.items():
            # 获取相邻网格（考虑周期性边界条件）
            neighbor_cells = self._get_neighbor_cells(grid_idx, grid_dims)

            for i in atom_indices:
                pos_i = positions[i]
                # 检查相邻网格中的原子
                for neighbor_idx in neighbor_cells:
                    for j in grid.get(neighbor_idx, []):
                        if j <= i:
                            continue

                        rij = positions[j] - pos_i
                        if cell.pbc_enabled:
                            rij = cell.minimum_image(rij)

                        dist_squared = np.dot(rij, rij)
                        if dist_squared < cutoff_squared:
                            self.neighbor_list[i].append(j)
                            self.neighbor_list[j].append(i)

    def _get_neighbor_cells(self, grid_idx, grid_dims):
        """获取相邻网格索引（考虑周期性边界条件）"""
        neighbor_cells = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    neighbor_idx = tuple(
                        (np.array(grid_idx) + [dx, dy, dz]) % grid_dims
                    )
                    neighbor_cells.append(neighbor_idx)
        return neighbor_cells

    def need_refresh(self):
        """
        判断是否需要更新邻居列表。

        Returns
        -------
        bool
            如果需要更新，返回 True；否则返回 False。
        """
        if self.last_positions is None:
            return True
        positions = self.cell.get_positions()
        displacements = positions - self.last_positions
        if self.cell.pbc_enabled:
            # 考虑 PBC 下的位移
            displacements = np.array(
                [self.cell.minimum_image(disp) for disp in displacements]
            )
        max_displacement = np.max(np.linalg.norm(displacements, axis=1))
        return max_displacement > (self.skin * 0.5)

    def update(self):
        """更新邻居列表，如果需要的话。"""
        if self.need_refresh():
            self.build(self.cell)

    def get_neighbors(self, atom_index):
        """
        获取指定原子的邻居列表。

        Parameters
        ----------
        atom_index : int
            原子的索引。

        Returns
        -------
        list of int
            邻居原子的索引列表。
        """
        if self.neighbor_list is None:
            self.build(self.cell)
        return self.neighbor_list[atom_index]

    def debug_neighbor_distribution(self):
        """分析和打印邻居分布情况"""
        logger = logging.getLogger(__name__)

        if self.neighbor_list is None:
            logger.warning("Neighbor list not built yet")
            return

        neighbor_counts = [len(neighbors) for neighbors in self.neighbor_list]
        logger.info("Neighbor distribution:")
        logger.info(f"Min neighbors: {min(neighbor_counts)}")
        logger.info(f"Max neighbors: {max(neighbor_counts)}")
        logger.info(f"Average neighbors: {np.mean(neighbor_counts):.2f}")

        # 检查边界原子
        positions = self.cell.get_positions()
        box_lengths = self.cell.get_box_lengths()
        for i, pos in enumerate(positions):
            is_boundary = any(
                abs(p / l - 0.5) > 0.35 for p, l in zip(pos, box_lengths, strict=False)
            )
            if is_boundary:
                logger.info(f"Boundary atom {i}:")
                logger.info(f"  Position: {pos}")
                logger.info(f"  Number of neighbors: {len(self.neighbor_list[i])}")
                logger.info(f"  Neighbors: {self.neighbor_list[i]}")


def get_atomic_mass(symbol: str) -> float:
    """
    根据原子符号获取原子质量（amu）

    Parameters
    ----------
    symbol : str
        原子符号，例如 'Al', 'Si'

    Returns
    -------
    float
        原子质量（amu）

    Raises
    ------
    KeyError
        如果原子符号不存在
    """
    atomic_masses = {
        "Al": 26.9815386,
        "Si": 28.0855,
        "C": 12.0107,
        # ... 其他元素
    }
    try:
        return atomic_masses[symbol]
    except KeyError as e:
        raise KeyError(f"Atomic mass for symbol '{symbol}' not found.") from e
