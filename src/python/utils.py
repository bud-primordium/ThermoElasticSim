# 文件名: utils.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现张量转换工具类和数据收集类，并定义了一些常用的单位转换常量。

"""
工具模块

包含 TensorConverter 类用于张量与 Voigt 表示之间的转换，DataCollector 类用于收集模拟过程中的数据，及一些常用的单位转换常量
"""

import numpy as np
import itertools


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
        self.cutoff = cutoff
        self.skin = skin
        self.cutoff_with_skin = cutoff + skin
        self.neighbor_list = None
        self.last_positions = None
        self.cell = None  # 引用到 Cell 对象

    def build(self, cell):
        positions = cell.get_positions()
        num_atoms = cell.num_atoms
        box_size = cell.get_box_lengths()
        cutoff = self.cutoff_with_skin

        # 定义网格尺寸
        grid_size = cutoff
        grid_dims = np.floor(box_size / grid_size).astype(int)
        grid_dims[grid_dims == 0] = 1  # 防止出现0，至少为1

        # 初始化网格
        grid = {}
        for idx, pos in enumerate(positions):
            grid_idx = tuple(((pos / grid_size) % grid_dims).astype(int))
            grid.setdefault(grid_idx, []).append(idx)

        # 构建邻居列表
        self.neighbor_list = [[] for _ in range(num_atoms)]
        for grid_idx, atom_indices in grid.items():
            # 检查当前网格及其相邻网格
            for offset in itertools.product([-1, 0, 1], repeat=3):
                neighbor_grid_idx = tuple((np.array(grid_idx) + offset) % grid_dims)
                neighbor_atom_indices = grid.get(neighbor_grid_idx, [])
                for i in atom_indices:
                    for j in neighbor_atom_indices:
                        if j > i:
                            rij = positions[j] - positions[i]
                            # 应用最小镜像原则
                            if cell.pbc_enabled:
                                rij = cell.minimum_image(rij)
                            distance_squared = np.dot(rij, rij)
                            if distance_squared < cutoff**2:
                                self.neighbor_list[i].append(j)
                                self.neighbor_list[j].append(i)

        self.last_positions = positions.copy()

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


# 单位转换常量
# 定义常见单位转换的常量，用于模拟中单位的转换

# 1 amu = 104.3968445 eV·fs²/Å²
AMU_TO_EVFSA2 = 104.3968445

# 1 eV/Å^3 = 160.21766208 GPa
EV_TO_GPA = 160.2176565
