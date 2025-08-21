"""MD算符分离架构接口定义

本模块定义了基于刘维尔算符Trotter分解的MD架构接口。
设计思想：将复杂的MD积分分解为可组合的基础算符。

主要类：
- MDComponent: MD组件基类
- Propagator: 算符传播子基类，对应exp(iL*dt)
- IntegrationScheme: 积分方案基类，通过Trotter分解组合Propagator

参考文献：
- Martyna et al., Mol. Phys. 87, 1117 (1996)
- Tuckerman et al., J. Chem. Phys. 97, 1990 (1992)

Created: 2025-08-18
"""

from abc import ABC, abstractmethod
from typing import Optional, Any


class MDComponent(ABC):
    """MD组件基类
    
    所有MD相关组件的基础抽象类，用于统一接口标准。
    """
    pass


class Propagator(MDComponent):
    """算符传播子基类
    
    对应刘维尔算符exp(iL*dt)的数值实现。每个Propagator负责系统状态
    在特定物理过程下的时间演化，如位置更新、速度更新、热浴演化等。
    
    设计原则：
    1. 单一职责：每个Propagator只负责一种物理过程
    2. 无状态：不依赖外部状态，所有信息通过参数传递
    3. 可组合性：可通过Trotter分解自由组合
    4. 时间可逆：支持对称分解保证数值稳定性
    
    典型实现：
    - PositionPropagator: exp(iL_r * dt) → r += v*dt
    - VelocityPropagator: exp(iL_v * dt) → v += F/m*dt  
    - NoseHooverPropagator: 热浴变量演化
    - BarostatPropagator: 压浴变量演化
    """
    
    @abstractmethod
    def propagate(self, cell, dt: float, **kwargs) -> None:
        """执行dt时间的演化
        
        Parameters
        ----------
        cell : Cell
            晶胞对象，包含原子位置、速度、力等信息
        dt : float
            时间步长 (fs)
        **kwargs : dict
            额外参数，如potential对象、温度、压力等
            
        Notes
        -----
        该方法应当：
        1. 就地修改cell中的相关属性
        2. 保持算符的数学性质（线性、幺正性等）
        3. 不产生副作用（如文件IO、日志输出等）
        4. 在数值精度范围内保持可逆性
        """
        pass


class IntegrationScheme(MDComponent):
    """积分方案基类
    
    通过Trotter分解组合多个Propagator实现完整的MD积分步骤。
    负责将物理上耦合的演化过程分解为可计算的序列。
    
    设计原则：
    1. 对称性：优先使用对称分解保证时间可逆性
    2. 物理正确：遵循相应系综的统计力学原理
    3. 数值稳定：控制Trotter分解误差在可接受范围
    4. 可扩展性：易于添加新的物理过程
    
    常见分解模式：
    - NVE: exp(iL*dt) ≈ exp(iL_v*dt/2)·exp(iL_r*dt)·exp(iL_v*dt/2)
    - NVT: 恒温器包裹NVE核心的对称分解
    - NPT: 多层嵌套的复杂对称分解
    """
    
    @abstractmethod
    def step(self, cell, potential, dt: float) -> None:
        """执行一个完整积分步
        
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
        该方法应当：
        1. 按照特定的Trotter分解顺序调用Propagator
        2. 确保分解的对称性和物理正确性
        3. 处理不同物理过程间的耦合
        4. 在必要时计算和更新力
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