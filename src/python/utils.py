# 文件名: utils.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现张量转换工具类和数据收集类，并定义了一些常用的单位转换常量。

"""
工具模块

包含 TensorConverter 类用于张量与 Voigt 表示之间的转换，DataCollector 类用于收集模拟过程中的数据，及一些常用的单位转换常量
"""

import numpy as np
import logging


class TensorConverter:
    """
    张量转换工具类，用于 3x3 张量和 Voigt 表示的 6 元素向量之间的转换
    """

    @staticmethod
    def to_voigt(tensor):
        """
        将 3x3 张量转换为 Voigt 表示的 6 元素向量

        Parameters
        ----------
        tensor : numpy.ndarray
            形状为 (3, 3) 的张量

        Returns
        -------
        numpy.ndarray
            形状为 (6,) 的 Voigt 表示向量
        """
        voigt = np.array(
            [
                tensor[0, 0],  # xx
                tensor[1, 1],  # yy
                tensor[2, 2],  # zz
                tensor[1, 2],  # yz
                tensor[0, 2],  # xz
                tensor[0, 1],  # xy
            ]
        )
        return voigt

    @staticmethod
    def from_voigt(voigt):
        """
        将 Voigt 表示的 6 元素向量转换为 3x3 张量

        Parameters
        ----------
        voigt : numpy.ndarray
            形状为 (6,) 的 Voigt 表示向量

        Returns
        -------
        numpy.ndarray
            形状为 (3, 3) 的张量
        """
        tensor = np.array(
            [
                [voigt[0], voigt[5], voigt[4]],  # xx, xy, xz
                [voigt[5], voigt[1], voigt[3]],  # xy, yy, yz
                [voigt[4], voigt[3], voigt[2]],  # xz, yz, zz
            ]
        )
        return tensor


class DataCollector:
    """
    数据收集工具类，用于收集模拟过程中的原子位置和速度数据
    """

    def __init__(self):
        """
        初始化数据收集器
        """
        self.data = []

    def collect(self, cell):
        """
        收集晶胞中的原子位置信息和速度信息

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        """
        positions = [atom.position.copy() for atom in cell.atoms]  # 复制原子位置
        velocities = [atom.velocity.copy() for atom in cell.atoms]  # 复制原子速度
        # 保存到数据字典
        self.data.append({"positions": positions, "velocities": velocities})


class NeighborList:
    """
    邻居列表类，用于生成和维护原子的邻居列表。
    """

    def __init__(self, cutoff, skin=0.3):
        """
        初始化邻居列表。

        Parameters
        ----------
        cutoff : float
            截断半径，单位为 Å。
        skin : float, optional
            皮肤厚度（skin width），默认值为 0.3 Å。
        """
        if cutoff <= 0:
            raise ValueError("Cutoff radius must be positive")
        if skin < 0:
            raise ValueError("Skin distance must be non-negative")

        self.cutoff = cutoff
        self.skin = skin
        self.cutoff_with_skin = cutoff + skin
        self.neighbor_list = None
        self.last_positions = None
        self.cell = None
        self.last_update_step = 0

    def _validate_cutoff(self, box_size):
        """验证截断半径的合理性"""
        min_box_length = np.min(box_size)
        if self.cutoff_with_skin > min_box_length / 2:
            raise ValueError(
                f"Cutoff radius ({self.cutoff_with_skin:.3f}) exceeds half of minimum box length ({min_box_length/2:.3f})"
            )

    def _compute_optimal_grid_size(self, box_size, num_atoms):
        """计算最优的网格大小"""
        # 根据原子密度和截断半径估算最优网格大小
        volume = np.prod(box_size)
        density = num_atoms / volume
        mean_atomic_spacing = (1 / density) ** (1 / 3)

        # 网格大小应该在截断半径和平均原子间距之间
        return min(self.cutoff_with_skin, max(mean_atomic_spacing, self.cutoff))

    def build(self, cell):
        """
        构建邻居列表。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        """
        positions = cell.get_positions()
        num_atoms = cell.num_atoms
        box_size = cell.get_box_lengths()
        cutoff = self.cutoff_with_skin
        self.cell = cell  # 正确设置关联的晶胞对象

        # 对小系统使用简单的双重循环
        if num_atoms < 64:
            self._build_brute_force(cell, positions, num_atoms, cutoff)
        else:
            # 对大系统使用网格划分
            self._build_with_grid(cell, positions, num_atoms, box_size, cutoff)

        self.last_positions = positions.copy()

    def _build_brute_force(self, cell, positions, num_atoms, cutoff):
        """
        使用双重循环构建小系统的邻居列表。
        """
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
        """
        改进的基于网格的邻居列表构建。
        """
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
        """
        更新邻居列表，如果需要的话。
        """
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
        """
        分析和打印邻居分布情况
        """
        logger = logging.getLogger(__name__)

        if self.neighbor_list is None:
            logger.warning("Neighbor list not built yet")
            return

        neighbor_counts = [len(neighbors) for neighbors in self.neighbor_list]
        logger.info(f"Neighbor distribution:")
        logger.info(f"Min neighbors: {min(neighbor_counts)}")
        logger.info(f"Max neighbors: {max(neighbor_counts)}")
        logger.info(f"Average neighbors: {np.mean(neighbor_counts):.2f}")

        # 检查边界原子
        positions = self.cell.get_positions()
        box_lengths = self.cell.get_box_lengths()
        for i, pos in enumerate(positions):
            is_boundary = any(abs(p / l - 0.5) > 0.4 for p, l in zip(pos, box_lengths))
            if is_boundary:
                logger.info(f"Boundary atom {i}:")
                logger.info(f"  Position: {pos}")
                logger.info(f"  Number of neighbors: {len(self.neighbor_list[i])}")
                logger.info(f"  Neighbors: {self.neighbor_list[i]}")


# 单位转换常量
# 定义常见单位转换的常量，用于模拟中单位的转换

# 1 amu = 104.3968445 eV·fs²/Å²
AMU_TO_EVFSA2 = 104.3968445

# 1 eV/Å^3 = 160.21766208 GPa
EV_TO_GPA = 160.2176565
