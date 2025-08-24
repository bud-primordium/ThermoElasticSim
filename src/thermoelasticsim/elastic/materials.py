#!/usr/bin/env python3
r"""
材料参数配置模块

该模块定义了用于弹性常数计算的标准材料参数，包括晶格常数、原子质量、
文献弹性常数等。支持多种晶体结构和材料元素。

主要组件：
- MaterialParameters: 材料参数数据类
- 预定义材料常量: ALUMINUM_FCC, COPPER_FCC等
- 材料查询和验证工具

基本使用：
    >>> from thermoelasticsim.elastic.materials import ALUMINUM_FCC, COPPER_FCC
    >>> print(f"Al晶格常数: {ALUMINUM_FCC.lattice_constant:.3f} Å")
    Al晶格常数: 4.050 Å

.. moduleauthor:: Gilbert Young
.. created:: 2025-08-24
.. version:: 4.0.0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialParameters:
    """
    材料参数数据类

    封装材料的所有基本物理参数，包括结构、热力学和弹性性质。
    使用frozen=True确保参数不可变，避免意外修改。

    Attributes
    ----------
    name : str
        材料全名，如"Aluminum"
    symbol : str
        化学元素符号，如"Al"
    mass_amu : float
        原子质量 (原子质量单位)
    lattice_constant : float
        晶格常数 (Å)
    structure : str
        晶体结构类型，如"fcc", "bcc", "hcp"
    literature_elastic_constants : dict
        文献弹性常数参考值 (GPa)
        包含键: "C11", "C12", "C44"等
    temperature : float, optional
        参考温度 (K)，默认为0K (零温)
    description : str, optional
        材料描述信息

    Examples
    --------
    创建自定义材料参数:

    >>> gold_fcc = MaterialParameters(
    ...     name="Gold",
    ...     symbol="Au",
    ...     mass_amu=196.966569,
    ...     lattice_constant=4.08,
    ...     structure="fcc",
    ...     literature_elastic_constants={
    ...         "C11": 192.0,
    ...         "C12": 163.0,
    ...         "C44": 41.5
    ...     }
    ... )
    >>> print(f"{gold_fcc.name}: C11 = {gold_fcc.literature_elastic_constants['C11']} GPa")
    Gold: C11 = 192.0 GPa

    Notes
    -----
    弹性常数命名约定：

    - C11, C22, C33: 正应力-正应变常数
    - C12, C13, C23: 泊松效应常数
    - C44, C55, C66: 剪切常数
    - 立方晶系: C11=C22=C33, C44=C55=C66, C12=C13=C23
    """

    name: str
    symbol: str
    mass_amu: float
    lattice_constant: float
    structure: str
    literature_elastic_constants: dict[str, float]
    temperature: float = 0.0  # 默认零温
    description: str = ""

    def __post_init__(self):
        """参数验证"""
        if self.mass_amu <= 0:
            raise ValueError(f"原子质量必须为正数，得到: {self.mass_amu}")

        if self.lattice_constant <= 0:
            raise ValueError(f"晶格常数必须为正数，得到: {self.lattice_constant}")

        if self.structure not in ["fcc", "bcc", "hcp"]:
            raise ValueError(f"不支持的晶体结构: {self.structure}")

        if not isinstance(self.literature_elastic_constants, dict):
            raise TypeError("弹性常数必须为字典类型")

        # 验证弹性常数的键值
        required_keys = {"C11", "C12", "C44"}
        missing_keys = required_keys - set(self.literature_elastic_constants.keys())
        if missing_keys:
            raise ValueError(f"缺少必要的弹性常数: {missing_keys}")

    @property
    def bulk_modulus(self) -> float:
        """
        计算体积模量

        对于立方晶系: K = (C11 + 2*C12) / 3

        Returns
        -------
        float
            体积模量 (GPa)
        """
        return (
            self.literature_elastic_constants["C11"]
            + 2 * self.literature_elastic_constants["C12"]
        ) / 3

    @property
    def shear_modulus(self) -> float:
        """
        计算剪切模量

        对于立方晶系: G = C44 (Voigt平均的一种近似)

        Returns
        -------
        float
            剪切模量 (GPa)
        """
        return self.literature_elastic_constants["C44"]

    @property
    def young_modulus(self) -> float:
        """
        计算杨氏模量

        使用关系: E = 9*K*G / (3*K + G)
        其中K为体积模量，G为剪切模量

        Returns
        -------
        float
            杨氏模量 (GPa)
        """
        K = self.bulk_modulus
        G = self.shear_modulus
        return 9 * K * G / (3 * K + G)

    @property
    def poisson_ratio(self) -> float:
        """
        计算泊松比

        使用关系: ν = (3*K - 2*G) / (6*K + 2*G)

        Returns
        -------
        float
            泊松比 (无量纲)
        """
        K = self.bulk_modulus
        G = self.shear_modulus
        return (3 * K - 2 * G) / (6 * K + 2 * G)


# ==================== 预定义材料参数 ====================

# 铝 (FCC结构) - Mendelev et al. (2008)
ALUMINUM_FCC = MaterialParameters(
    name="Aluminum",
    symbol="Al",
    mass_amu=26.9815,
    lattice_constant=4.045,  # Å, EAM Al1 常用参考晶格常数
    structure="fcc",
    literature_elastic_constants={
        # Mendelev et al. (2008) 文献值 (GPa, 0K)
        "C11": 110,
        "C12": 61,
        "C44": 33,
        # 立方对称性
        "C22": 110,  # C22 = C11
        "C33": 110,  # C33 = C11
        "C55": 33,  # C55 = C44
        "C66": 33,  # C66 = C44
        "C13": 61,  # C13 = C12
        "C23": 61,  # C23 = C12
    },
    temperature=0.0,
    description="EAM Al1势函数参数，基于Mendelev等人2008年工作",
)

# 铜 (FCC结构) - Mendelev et al. (2008)
COPPER_FCC = MaterialParameters(
    name="Copper",
    symbol="Cu",
    mass_amu=63.546,
    lattice_constant=3.639,  # Å, EAM Cu1 常用参考晶格常数
    structure="fcc",
    literature_elastic_constants={
        # Mendelev et al. (2008) 文献值 (GPa, 0K)
        "C11": 175,
        "C12": 128,
        "C44": 84,
        # 立方对称性
        "C22": 175,  # C22 = C11
        "C33": 175,  # C33 = C11
        "C55": 84,  # C55 = C44
        "C66": 84,  # C66 = C44
        "C13": 128,  # C13 = C12
        "C23": 128,  # C23 = C12
    },
    temperature=0.0,
    description="EAM Cu1势函数参数，基于Mendelev等人2008年工作",
)

# 金 (FCC结构) - 预留，待验证
GOLD_FCC = MaterialParameters(
    name="Gold",
    symbol="Au",
    mass_amu=196.966569,
    lattice_constant=4.08,  # Å
    structure="fcc",
    literature_elastic_constants={
        # 实验值 (GPa, 室温)
        "C11": 192.0,
        "C12": 163.0,
        "C44": 41.5,
        # 立方对称性
        "C22": 192.0,
        "C33": 192.0,
        "C55": 41.5,
        "C66": 41.5,
        "C13": 163.0,
        "C23": 163.0,
    },
    temperature=300.0,
    description="实验弹性常数，室温数据",
)


# ==================== 工具函数 ====================


def get_material_by_symbol(symbol: str) -> MaterialParameters | None:
    """
    根据元素符号获取材料参数

    Parameters
    ----------
    symbol : str
        化学元素符号，如"Al", "Cu"

    Returns
    -------
    MaterialParameters or None
        匹配的材料参数，未找到返回None

    Examples
    --------
    >>> al_params = get_material_by_symbol("Al")
    >>> if al_params:
    ...     print(f"找到材料: {al_params.name}")
    找到材料: Aluminum
    """
    material_registry = {
        "Al": ALUMINUM_FCC,
        "Cu": COPPER_FCC,
        "Au": GOLD_FCC,
    }

    return material_registry.get(symbol)


def get_all_materials() -> dict[str, MaterialParameters]:
    """
    获取所有预定义材料参数

    Returns
    -------
    dict
        键为材料名称，值为MaterialParameters对象的字典

    Examples
    --------
    >>> materials = get_all_materials()
    >>> for name, params in materials.items():
    ...     print(f"{name}: {params.structure.upper()}")
    Aluminum: FCC
    Copper: FCC
    Gold: FCC
    """
    return {
        "Aluminum": ALUMINUM_FCC,
        "Copper": COPPER_FCC,
        "Gold": GOLD_FCC,
    }


def compare_elastic_constants(
    material1: MaterialParameters, material2: MaterialParameters
) -> dict[str, dict[str, float]]:
    """
    比较两种材料的弹性常数

    Parameters
    ----------
    material1, material2 : MaterialParameters
        要比较的材料参数

    Returns
    -------
    dict
        包含比较结果的嵌套字典

    Examples
    --------
    >>> al = ALUMINUM_FCC
    >>> cu = COPPER_FCC
    >>> comparison = compare_elastic_constants(al, cu)
    >>> print(f"C11比值: {comparison['ratios']['C11']:.2f}")
    C11比值: 0.65
    """
    # 获取公共的弹性常数键
    common_keys = set(material1.literature_elastic_constants.keys()) & set(
        material2.literature_elastic_constants.keys()
    )

    comparison = {
        "material1": material1.name,
        "material2": material2.name,
        "absolute_differences": {},
        "relative_differences": {},
        "ratios": {},
    }

    for key in common_keys:
        val1 = material1.literature_elastic_constants[key]
        val2 = material2.literature_elastic_constants[key]

        comparison["absolute_differences"][key] = val2 - val1
        comparison["relative_differences"][key] = (val2 - val1) / val1 * 100  # 百分比
        comparison["ratios"][key] = val2 / val1

    return comparison
