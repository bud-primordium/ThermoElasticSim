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

参考文献
--------
Martyna et al., Mol. Phys. 87, 1117 (1996);
Tuckerman et al., J. Chem. Phys. 97, 1990 (1992);
Martyna et al., J. Chem. Phys. 97, 2635 (1992);
Yoshida, Phys. Lett. A 150, 262 (1990).
"""

import abc


class MDComponent:
    """MD组件基类

    所有MD相关组件的基础抽象类，用于统一接口标准。
    """

    pass


class Propagator(MDComponent):
    r"""算符传播子基类。

    在特定物理过程下更新系统状态（如位置、速度、热浴变量等）。
    """

    @abc.abstractmethod
    def propagate(self, cell, dt: float, **kwargs) -> None:
        r"""执行一个时间步长 dt 的演化。"""
        pass


class IntegrationScheme(MDComponent):
    r"""积分方案基类。

    通过 Trotter 分解组合多个传播子，实现完整积分步骤。
    """

    @abc.abstractmethod
    def step(self, cell, potential, dt: float) -> None:
        r"""执行一个完整积分步。"""
        pass


class ThermostatInterface(MDComponent):
    """恒温器接口。"""

    @abc.abstractmethod
    def apply(self, cell, dt: float, potential) -> None:
        """应用恒温器控制。"""
        pass


class BarostatInterface(MDComponent):
    """恒压器接口。"""

    @abc.abstractmethod
    def apply(self, cell, dt: float, potential) -> None:
        """应用恒压器控制。"""
        pass
