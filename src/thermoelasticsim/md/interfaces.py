r"""MD 算符分离架构接口定义

本模块定义了基于刘维尔算符 Trotter 分解的 MD 架构接口；
核心思想是将完整积分步骤拆解为可组合的基础算符。

主要类
------

MDComponent
    MD 组件基类

Propagator
    算符传播子基类，对应 :math:`\exp(iL\,\Delta t)`

IntegrationScheme
    积分方案基类，通过 Trotter 分解组合多个传播子

References
----------
.. [1] Martyna, G. J.; Tuckerman, M. E.; Tobias, D. J.; Klein, M. L. (1996).
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117–1157. https://doi.org/10.1080/00268979600100761
.. [2] Tuckerman, M.; Berne, B. J.; Martyna, G. J. (1992).
       Reversible multiple time scale molecular dynamics.
       The Journal of Chemical Physics, 97(3), 1990–2001. https://doi.org/10.1063/1.463137
.. [3] Martyna, G. J.; Klein, M. L.; Tuckerman, M. E. (1992).
       Nosé–Hoover chains: The canonical ensemble via continuous dynamics.
       The Journal of Chemical Physics, 97(4), 2635–2643. https://doi.org/10.1063/1.463940
.. [4] Yoshida, H. (1990). Construction of higher order symplectic integrators.
       Physics Letters A, 150(5–7), 262–268. https://doi.org/10.1016/0375-9601(90)90092-3
"""

from abc import abstractmethod


class MDComponent:
    """MD组件基类

    所有MD相关组件的基础抽象类，用于统一接口标准。
    """

    pass


class Propagator(MDComponent):
    r"""算符传播子基类

    数值实现 :math:`\exp(iL\,\Delta t)`，在特定物理过程下更新系统状态
    （如位置、速度、热浴变量等）。

    设计原则
    --------
    1. 单一职责：每个传播子仅负责一种物理过程
    2. 无状态：不依赖外部全局状态，信息通过参数传递
    3. 可组合：通过 Trotter 分解自由组合
    4. 可逆性：优先对称分解以提高稳定性

    典型实现
    --------

    :math:`\exp(iL_r\,\Delta t)`
        位置传播（:class:`~thermoelasticsim.md.propagators.PositionPropagator`）

    :math:`\exp(iL_v\,\Delta t)`
        速度传播（:class:`~thermoelasticsim.md.propagators.VelocityPropagator`）

    热浴/压浴变量传播
        如 Nose–Hoover、MTK 等（见相关类）
    """

    @abstractmethod
    def propagate(self, cell, dt: float, **kwargs) -> None:
        r"""执行 :math:`\Delta t` 时间的演化

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含原子位置、速度、力等信息
        dt : float
            时间步长 (fs)
        **kwargs : dict
            额外参数，如 ``potential`` 对象、温度、压力等

        Notes
        -----
        1. 就地修改 `cell` 的相关属性
        2. 保持算符的数学性质（线性、幺正性等）
        3. 不产生副作用（如文件 IO、日志输出等）
        4. 在数值精度范围内保持可逆性
        """
        pass


class IntegrationScheme(MDComponent):
    r"""积分方案基类

    通过 Trotter 分解组合多个传播子，实现完整积分步骤。

    设计原则
    --------
    1. 对称性：优先对称分解以保证时间可逆性
    2. 物理性：遵循目标系综的统计力学原理
    3. 稳定性：控制分解误差在可接受范围
    4. 扩展性：便于添加新的物理过程

    示例（NVE 对称分解）
    --------------------

    .. math::
        \exp(iL\,\Delta t) \approx \exp(iL_v\,\tfrac{\Delta t}{2})\,\exp(iL_r\,\Delta t)\,\exp(iL_v\,\tfrac{\Delta t}{2})
    """

    @abstractmethod
    def step(self, cell, potential, dt: float) -> None:
        r"""执行一个完整积分步

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势函数对象，用于计算原子间相互作用
        dt : float
            时间步长 (fs)

        Notes
        -----
        1. 按既定 Trotter 顺序调用各传播子
        2. 保持对称性与物理正确性
        3. 处理不同物理过程的耦合
        4. 在必要位置计算与更新力
        """
        pass


class ThermostatInterface(MDComponent):
    """恒温器接口

    定义恒温器的标准接口，用于向后兼容现有实现。
    新的Nose-Hoover等实现应当使用Propagator架构。
    """

    @abstractmethod
    def apply(self, cell, dt: float, potential) -> None:
        """应用恒温器控制

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长 (fs)
        potential : Potential
            势函数对象
        """
        pass


class BarostatInterface(MDComponent):
    """恒压器接口

    定义恒压器的标准接口，用于向后兼容现有实现。
    新的Parrinello-Rahman等实现应当使用Propagator架构。
    """

    @abstractmethod
    def apply(self, cell, dt: float, potential) -> None:
        """应用恒压器控制

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长 (fs)
        potential : Potential
            势函数对象
        """
        pass
