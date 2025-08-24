#!/usr/bin/env python3
r"""
晶体结构生成器模块

该模块提供统一的晶体结构生成接口，支持多种晶格类型和材料元素。
主要用于为弹性常数计算和分子动力学模拟创建标准的晶体结构。

支持的晶格类型：
- FCC (面心立方): 如Al、Cu、Au等
- BCC (体心立方): 如Fe、Cr等 (预留)
- HCP (密排六方): 如Zn、Mg等 (预留)

基本使用：
    >>> builder = CrystallineStructureBuilder()
    >>> cell = builder.create_fcc("Al", 4.05, (3, 3, 3))
    >>> cell.num_atoms
    108

.. moduleauthor:: Gilbert Young
.. created:: 2025-08-24
.. version:: 4.0.0
"""

import numpy as np

from thermoelasticsim.core.structure import Atom, Cell


class CrystallineStructureBuilder:
    """
    统一的晶体结构生成器

    提供多种晶格类型的标准化生成方法，支持任意超胞尺寸和材料参数。
    所有生成的结构都经过周期性边界条件优化，适用于分子动力学计算。

    Methods
    -------
    create_fcc(element, lattice_constant, supercell)
        创建面心立方(FCC)结构
    create_bcc(element, lattice_constant, supercell)
        创建体心立方(BCC)结构 (预留)
    create_hcp(element, lattice_constant, supercell)
        创建密排六方(HCP)结构 (预留)

    Examples
    --------
    创建3×3×3的铝FCC超胞:

    >>> builder = CrystallineStructureBuilder()
    >>> al_cell = builder.create_fcc("Al", 4.05, (3, 3, 3))
    >>> print(f"原子数: {al_cell.num_atoms}")
    原子数: 108

    创建不同尺寸的铜结构:

    >>> cu_cell = builder.create_fcc("Cu", 3.615, (4, 4, 4))
    >>> print(f"晶胞体积: {cu_cell.volume:.1f} Å³")
    晶胞体积: 1885.4 Å³
    """

    # 常见元素的原子质量 (amu)
    ATOMIC_MASSES = {
        "Al": 26.9815,
        "Cu": 63.546,
        "Au": 196.966569,
        "Ag": 107.8682,
        "Fe": 55.845,
        "Cr": 51.9961,
        "Ni": 58.6934,
        "Pb": 207.2,
        "Mg": 24.305,
        "Zn": 65.38,
        "Ti": 47.867,
        "V": 50.9415,
    }

    def __init__(self):
        """初始化晶体结构生成器"""
        pass

    def create_fcc(
        self, element: str, lattice_constant: float, supercell: tuple[int, int, int]
    ) -> Cell:
        """
        创建面心立方(FCC)结构

        FCC结构特点：
        - 原胞包含4个原子
        - 基矢：(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        - 配位数：12
        - 致密度：74%

        Parameters
        ----------
        element : str
            元素符号，如"Al"、"Cu"等
        lattice_constant : float
            晶格常数 (Å)
        supercell : tuple of int
            超胞尺寸 (nx, ny, nz)

        Returns
        -------
        Cell
            包含FCC结构的晶胞对象

        Raises
        ------
        ValueError
            如果元素不在支持列表中
        TypeError
            如果参数类型不正确

        Examples
        --------
        创建标准Al FCC结构:

        >>> builder = CrystallineStructureBuilder()
        >>> cell = builder.create_fcc("Al", 4.05, (2, 2, 2))
        >>> cell.num_atoms
        32

        Notes
        -----
        生成的结构特性：

        1. **原子数计算**：总原子数 = 4 × nx × ny × nz
        2. **晶胞矢量**：正交晶胞，a = b = c = lattice_constant
        3. **周期边界**：默认启用，适用于无限系统模拟
        4. **原子编号**：从0开始连续编号
        """
        # 参数验证
        if not isinstance(element, str) or element not in self.ATOMIC_MASSES:
            raise ValueError(
                f"不支持的元素: {element}. "
                f"支持的元素: {list(self.ATOMIC_MASSES.keys())}"
            )

        if not isinstance(lattice_constant, int | float) or lattice_constant <= 0:
            raise ValueError(f"晶格常数必须为正数，得到: {lattice_constant}")

        if (
            not isinstance(supercell, tuple)
            or len(supercell) != 3
            or not all(isinstance(n, int) and n > 0 for n in supercell)
        ):
            raise TypeError(f"超胞尺寸必须为正整数三元组，得到: {supercell}")

        nx, ny, nz = supercell
        mass_amu = self.ATOMIC_MASSES[element]

        # FCC基原子位置 (分数坐标)
        fcc_base_positions = np.array(
            [
                [0.0, 0.0, 0.0],  # 角原子
                [0.5, 0.5, 0.0],  # xy面心
                [0.5, 0.0, 0.5],  # xz面心
                [0.0, 0.5, 0.5],  # yz面心
            ]
        )

        # 构建超胞晶格矢量
        lattice_vectors = np.eye(3) * lattice_constant * np.array([nx, ny, nz])

        # 生成所有原子位置
        atoms = []
        atom_id = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 在当前单胞中放置4个FCC基原子
                    for base_pos in fcc_base_positions:
                        # 相对于超胞的分数坐标：((i, j, k) + base_pos) / (nx, ny, nz)
                        # 注意：不能再除以 (nx, ny, nz) 两次，否则会导致坐标缩放错误
                        fractional_pos = (np.array([i, j, k]) + base_pos) / np.array(
                            [nx, ny, nz]
                        )

                        # 转换为直角坐标
                        cartesian_pos = fractional_pos @ lattice_vectors

                        # 创建原子对象
                        atom = Atom(
                            id=atom_id,
                            symbol=element,
                            mass_amu=mass_amu,
                            position=cartesian_pos,
                        )
                        atoms.append(atom)
                        atom_id += 1

        # 创建晶胞对象
        cell = Cell(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            pbc_enabled=True,  # 启用周期性边界条件
        )

        return cell

    def create_bcc(
        self, element: str, lattice_constant: float, supercell: tuple[int, int, int]
    ) -> Cell:
        """
        创建体心立方(BCC)结构

        BCC结构特点：
        - 原胞包含2个原子
        - 基矢：(0,0,0), (0.5,0.5,0.5)
        - 配位数：8
        - 致密度：68%

        Parameters
        ----------
        element : str
            元素符号，如"Fe"、"Cr"等
        lattice_constant : float
            晶格常数 (Å)
        supercell : tuple of int
            超胞尺寸 (nx, ny, nz)

        Returns
        -------
        Cell
            包含BCC结构的晶胞对象

        Notes
        -----
        此方法为预留接口，完整实现待后续版本。
        当前主要支持FCC结构的弹性常数计算。
        """
        raise NotImplementedError("BCC结构生成将在后续版本中实现")

    def create_hcp(
        self, element: str, lattice_constant: float, supercell: tuple[int, int, int]
    ) -> Cell:
        """
        创建密排六方(HCP)结构

        HCP结构特点：
        - 原胞包含2个原子
        - 六方晶系，c/a ≈ 1.633
        - 配位数：12
        - 致密度：74%

        Parameters
        ----------
        element : str
            元素符号，如"Zn"、"Mg"等
        lattice_constant : float
            a轴晶格常数 (Å)
        supercell : tuple of int
            超胞尺寸 (nx, ny, nz)

        Returns
        -------
        Cell
            包含HCP结构的晶胞对象

        Notes
        -----
        此方法为预留接口，完整实现待后续版本。
        当前主要支持FCC结构的弹性常数计算。
        """
        raise NotImplementedError("HCP结构生成将在后续版本中实现")

    @staticmethod
    def get_supported_elements() -> list[str]:
        """
        获取支持的元素列表

        Returns
        -------
        list of str
            支持的元素符号列表
        """
        return list(CrystallineStructureBuilder.ATOMIC_MASSES.keys())

    @staticmethod
    def get_atomic_mass(element: str) -> float:
        """
        获取指定元素的原子质量

        Parameters
        ----------
        element : str
            元素符号

        Returns
        -------
        float
            原子质量 (amu)

        Raises
        ------
        ValueError
            如果元素不在支持列表中
        """
        if element not in CrystallineStructureBuilder.ATOMIC_MASSES:
            raise ValueError(
                f"不支持的元素: {element}. "
                f"支持的元素: {list(CrystallineStructureBuilder.ATOMIC_MASSES.keys())}"
            )
        return CrystallineStructureBuilder.ATOMIC_MASSES[element]
