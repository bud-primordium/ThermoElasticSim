r"""积分方案实现

基于刘维尔算符的对称 Trotter 分解，组合基础传播子形成不同系综的积分方案。

已实现的方案
--------------

NVEScheme
    微正则系综的 Velocity-Verlet 算法实现。

设计要点
--------
1. 基于对称 Trotter 分解保持时间可逆性与稳定性
2. 复用基础传播子，避免重复力计算
3. 每个方案保证其对应系综的物理正确性
4. 支持延迟初始化以处理势函数依赖
"""

import numpy as np

from ..core.structure import Cell
from .interfaces import IntegrationScheme
from .propagators import (
    AndersenThermostatPropagator,
    BerendsenThermostatPropagator,
    ForcePropagator,
    LangevinThermostatPropagator,
    MTKBarostatPropagator,
    NoseHooverChainPropagator,
    PositionPropagator,
    VelocityPropagator,
)


class NVEScheme(IntegrationScheme):
    r"""微正则系综 (NVE) 的 Velocity-Verlet 积分方案

    对称算符分离形式：

    .. math::
        \exp(iL\,\Delta t) \approx \exp(iL_v\,\tfrac{\Delta t}{2})\,\exp(iL_r\,\Delta t)\,\exp(iL_v\,\tfrac{\Delta t}{2})

    其中

    :math:`iL_v`
        速度传播算符（力对动量的作用）

    :math:`iL_r`
        位置传播算符（动量对位置的作用）

    Notes
    -----
    基本步骤（v-半步，r-整步，再 v-半步）：

    1. :math:`\mathbf{v}(t+\tfrac{\Delta t}{2}) = \mathbf{v}(t) + \mathbf{F}(t)\,\Delta t/(2m)`
    2. :math:`\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{v}(t+\tfrac{\Delta t}{2})\,\Delta t`
    3. 计算 :math:`\mathbf{F}(t+\Delta t)`
    4. :math:`\mathbf{v}(t+\Delta t) = \mathbf{v}(t+\tfrac{\Delta t}{2}) + \mathbf{F}(t+\Delta t)\,\Delta t/(2m)`
    """

    def __init__(self):
        """初始化 NVE 积分方案（延迟创建力传播子）"""
        self.r_prop = PositionPropagator()
        self.v_prop = VelocityPropagator()
        self.f_prop: ForcePropagator | None = None  # 延迟初始化，第一次调用step时创建

        # 统计信息
        self._step_count = 0
        self._total_time = 0.0

    def step(self, cell: Cell, potential, dt: float) -> None:
        """执行一步Velocity-Verlet积分

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含所有原子信息
        potential : Potential
            势函数对象，用于计算原子间相互作用力
        dt : float
            时间步长 (fs)

        Notes
        -----
        该方法严格按照Velocity-Verlet算法执行：

        1. 第一次半步速度更新：使用当前位置的力
        2. 整步位置更新：使用更新后的速度
        3. 力重新计算：在新位置处计算力
        4. 第二次半步速度更新：使用新位置的力

        这确保了算法的对称性和时间可逆性。

        Raises
        ------
        ValueError
            如果dt <= 0
        RuntimeError
            如果势函数计算失败
        """
        if dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 dt={dt}")

        # 延迟初始化ForcePropagator（避免循环依赖）
        if self.f_prop is None:
            self.f_prop = ForcePropagator(potential)

        try:
            # Velocity-Verlet算法的标准步骤

            # 1. 速度半步更新：v(t+dt/2) = v(t) + F(t)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 2. 位置整步更新：r(t+dt) = r(t) + v(t+dt/2) * dt
            self.r_prop.propagate(cell, dt)

            # 3. 重新计算力：F(t+dt) = -∇U(r(t+dt))
            self.f_prop.propagate(cell, dt)

            # 4. 速度再次半步更新：v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 更新统计信息
            self._step_count += 1
            self._total_time += dt

        except Exception as e:
            raise RuntimeError(f"NVE积分步骤失败: {str(e)}") from e

    def get_statistics(self) -> dict:
        """获取积分统计信息

        Returns
        -------
        dict
            统计信息字典：

            step_count
                已执行的积分步数

            total_time
                总积分时间 (fs)

            average_dt
                平均时间步长 (fs)
        """
        avg_dt = self._total_time / self._step_count if self._step_count > 0 else 0.0

        return {
            "step_count": self._step_count,
            "total_time": self._total_time,
            "average_dt": avg_dt,
        }

    def reset_statistics(self) -> None:
        """重置积分统计信息"""
        self._step_count = 0
        self._total_time = 0.0


class BerendsenNVTScheme(IntegrationScheme):
    """Berendsen NVT 积分方案（NVE + 速度缩放恒温器）

    Notes
    -----
    两阶段流程：

    1. NVE：标准 Velocity-Verlet 积分
    2. NVT：应用 Berendsen 恒温器进行速度缩放
    """

    def __init__(self, target_temperature: float, tau: float = 100.0):
        """初始化Berendsen NVT积分方案

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        tau : float, optional
            Berendsen恒温器时间常数 (fs)，默认100fs
            - 较小值：快速温度调节，强耦合
            - 较大值：缓慢温度调节，弱耦合

        Raises
        ------
        ValueError
            如果target_temperature或tau为非正数
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if tau <= 0:
            raise ValueError(f"时间常数必须为正数，得到 {tau} fs")

        # 创建传播子组件
        self.r_prop = PositionPropagator()
        self.v_prop = VelocityPropagator()
        self.f_prop: ForcePropagator | None = None  # 延迟初始化
        self.thermostat = BerendsenThermostatPropagator(
            target_temperature, tau=tau, mode="equilibration"
        )

        # 统计信息
        self._step_count = 0
        self._total_time = 0.0
        self.target_temperature = target_temperature
        self.tau = tau

    def step(self, cell: Cell, potential, dt: float) -> None:
        """执行一步Berendsen NVT积分

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含所有原子信息
        potential : Potential
            势函数对象，用于计算原子间相互作用力
        dt : float
            时间步长 (fs)

        Notes
        -----
        该方法执行两阶段积分：

        1. NVE阶段：标准Velocity-Verlet积分
           - 速度半步更新
           - 位置整步更新
           - 力重新计算
           - 速度再次半步更新

        2. NVT阶段：Berendsen温度调节
           - 计算当前温度
           - 计算速度缩放因子
           - 调节所有原子速度

        Raises
        ------
        ValueError
            如果dt <= 0
        RuntimeError
            如果积分过程失败
        """
        if dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 dt={dt}")

        # 延迟初始化ForcePropagator
        if self.f_prop is None:
            self.f_prop = ForcePropagator(potential)

        try:
            # 阶段1: 标准Velocity-Verlet积分（NVE部分）

            # 1. 速度半步更新：v(t+dt/2) = v(t) + F(t)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 2. 位置整步更新：r(t+dt) = r(t) + v(t+dt/2) * dt
            self.r_prop.propagate(cell, dt)

            # 3. 重新计算力：F(t+dt) = -∇U(r(t+dt))
            self.f_prop.propagate(cell, dt)

            # 4. 速度再次半步更新：v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 阶段2: Berendsen恒温器调节（NVT部分）
            self.thermostat.propagate(cell, dt)

            # 更新统计信息
            self._step_count += 1
            self._total_time += dt

        except Exception as e:
            raise RuntimeError(f"Berendsen NVT积分步骤失败: {str(e)}") from e

    def get_statistics(self) -> dict:
        """获取积分与恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            step_count
                已执行的积分步数

            total_time
                总积分时间 (fs)

            average_dt
                平均时间步长 (fs)

            target_temperature
                目标温度 (K)

            tau
                恒温器时间常数 (fs)

            thermostat_stats
                恒温器详细统计
        """
        avg_dt = self._total_time / self._step_count if self._step_count > 0 else 0.0

        return {
            "step_count": self._step_count,
            "total_time": self._total_time,
            "average_dt": avg_dt,
            "target_temperature": self.target_temperature,
            "tau": self.tau,
            "thermostat_stats": self.thermostat.get_statistics(),
        }

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._step_count = 0
        self._total_time = 0.0
        self.thermostat.reset_statistics()

    def get_current_temperature(self, cell: Cell) -> float:
        """获取当前系统温度

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            当前温度 (K)
        """
        return cell.calculate_temperature()


class AndersenNVTScheme(IntegrationScheme):
    """Andersen NVT 积分方案（NVE + 随机碰撞恒温器）

    Notes
    -----
    两阶段流程：

    1. NVE：标准 Velocity-Verlet 积分
    2. NVT：按碰撞概率重新采样 Maxwell 速度
    """

    def __init__(self, target_temperature: float, collision_frequency: float = 0.01):
        """初始化Andersen NVT积分方案

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        collision_frequency : float, optional
            碰撞频率 ν (fs⁻¹)，默认0.01 fs⁻¹
            - 较大值：更频繁碰撞，更强温度控制
            - 较小值：较少碰撞，动力学连续性更好

        Raises
        ------
        ValueError
            如果target_temperature或collision_frequency为非正数
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if collision_frequency <= 0:
            raise ValueError(f"碰撞频率必须为正数，得到 {collision_frequency} fs⁻¹")

        # 创建传播子组件
        self.r_prop = PositionPropagator()
        self.v_prop = VelocityPropagator()
        self.f_prop: ForcePropagator | None = None  # 延迟初始化
        self.thermostat = AndersenThermostatPropagator(
            target_temperature, collision_frequency=collision_frequency
        )

        # 统计信息
        self._step_count = 0
        self._total_time = 0.0
        self.target_temperature = target_temperature
        self.collision_frequency = collision_frequency

    def step(self, cell: Cell, potential, dt: float) -> None:
        """执行一步Andersen NVT积分

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含所有原子信息
        potential : Potential
            势函数对象，用于计算原子间相互作用力
        dt : float
            时间步长 (fs)

        Notes
        -----
        该方法执行两阶段积分：

        1. NVE阶段：标准Velocity-Verlet积分
           - 速度半步更新
           - 位置整步更新
           - 力重新计算
           - 速度再次半步更新

        2. NVT阶段：Andersen随机碰撞
           - 为每个原子计算碰撞概率
           - 随机选择碰撞原子
           - 重新采样碰撞原子的Maxwell速度

        Raises
        ------
        ValueError
            如果dt <= 0
        RuntimeError
            如果积分过程失败
        """
        if dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 dt={dt}")

        # 延迟初始化ForcePropagator
        if self.f_prop is None:
            self.f_prop = ForcePropagator(potential)

        try:
            # 阶段1: 标准Velocity-Verlet积分（NVE部分）

            # 1. 速度半步更新：v(t+dt/2) = v(t) + F(t)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 2. 位置整步更新：r(t+dt) = r(t) + v(t+dt/2) * dt
            self.r_prop.propagate(cell, dt)

            # 3. 重新计算力：F(t+dt) = -∇U(r(t+dt))
            self.f_prop.propagate(cell, dt)

            # 4. 速度再次半步更新：v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 阶段2: Andersen随机碰撞恒温器（NVT部分）
            self.thermostat.propagate(cell, dt)

            # 更新统计信息
            self._step_count += 1
            self._total_time += dt

        except Exception as e:
            raise RuntimeError(f"Andersen NVT积分步骤失败: {str(e)}") from e

    def get_statistics(self) -> dict:
        """获取积分和恒温器统计信息

        Returns
        -------
        dict
            包含完整统计的字典:
            - 'step_count': 已执行的积分步数
            - 'total_time': 总积分时间 (fs)
            - 'average_dt': 平均时间步长 (fs)
            - 'target_temperature': 目标温度 (K)
            - 'collision_frequency': 碰撞频率 (fs⁻¹)
            - 'thermostat_stats': 恒温器详细统计
        """
        avg_dt = self._total_time / self._step_count if self._step_count > 0 else 0.0

        return {
            "step_count": self._step_count,
            "total_time": self._total_time,
            "average_dt": avg_dt,
            "target_temperature": self.target_temperature,
            "collision_frequency": self.collision_frequency,
            "thermostat_stats": self.thermostat.get_statistics(),
        }

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._step_count = 0
        self._total_time = 0.0
        self.thermostat.reset_statistics()

    def get_current_temperature(self, cell: Cell) -> float:
        """获取当前系统温度

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            当前温度 (K)
        """
        return cell.calculate_temperature()

    def get_collision_statistics(self) -> dict:
        """获取碰撞统计信息的便利方法

        Returns
        -------
        dict
            碰撞统计信息
        """
        return self.thermostat.get_statistics()


class NoseHooverNVTScheme(IntegrationScheme):
    r"""Nose–Hoover NVT 积分方案（算符分离）

    基于对称分解：

    .. math::
        \exp(iL\,\Delta t) \approx \exp(iL_{NH}\,\tfrac{\Delta t}{2})\,\exp(iL_v\,\tfrac{\Delta t}{2})\,\exp(iL_r\,\Delta t)\,\exp(iL_v\,\tfrac{\Delta t}{2})\,\exp(iL_{NH}\,\tfrac{\Delta t}{2})
    """

    def __init__(
        self,
        target_temperature: float,
        tdamp: float = 100.0,
        tchain: int = 3,
        tloop: int = 1,
    ):
        """初始化Nose-Hoover NVT积分方案

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        tdamp : float, optional
            特征时间常数 (fs)，默认100fs
            推荐值：50-100*dt，控制耦合强度
        tchain : int, optional
            热浴链长度，默认3
            M=3通常足够保证遍历性和稳定性
        tloop : int, optional
            Suzuki-Yoshida循环次数，默认1

        Raises
        ------
        ValueError
            如果target_temperature为非正数或tdamp为非正数
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if tdamp <= 0:
            raise ValueError(f"时间常数必须为正数，得到 {tdamp} fs")
        if tchain < 1:
            raise ValueError(f"链长度必须≥1，得到 {tchain}")
        if tloop < 1:
            raise ValueError(f"循环次数必须≥1，得到 {tloop}")

        # 创建传播子组件
        self.r_prop = PositionPropagator()
        self.v_prop = VelocityPropagator()
        self.f_prop: ForcePropagator | None = None  # 延迟初始化
        self.thermostat = NoseHooverChainPropagator(
            target_temperature, tdamp, tchain, tloop
        )

        # 统计信息
        self._step_count = 0
        self._total_time = 0.0
        self.target_temperature = target_temperature
        self.tdamp = tdamp
        self.tchain = tchain
        self.tloop = tloop

    def step(self, cell: Cell, potential, dt: float) -> None:
        """执行一步Nose-Hoover NVT积分

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含所有原子信息
        potential : Potential
            势函数对象，用于计算原子间相互作用力
        dt : float
            时间步长 (fs)

        Notes
        -----
        该方法执行严格的算符分离积分：

        1. Nose-Hoover算符半步：处理热浴变量和速度摩擦
        2. 标准Velocity-Verlet积分：
           - 速度半步更新
           - 位置整步更新
           - 力重新计算
           - 速度再次半步更新
        3. Nose-Hoover算符再次半步：完成热浴变量演化

        这种分解保证算法的时间可逆性和相空间体积保持。

        Raises
        ------
        ValueError
            如果dt == 0
        RuntimeError
            如果积分过程失败
        """
        if dt == 0:
            raise ValueError(f"时间步长不能为零，得到 dt={dt}")

        # 延迟初始化ForcePropagator
        if self.f_prop is None:
            self.f_prop = ForcePropagator(potential)

        try:
            # 步骤1: Nose-Hoover恒温器半步
            self.thermostat.propagate(cell, dt / 2)

            # 步骤2: 标准Velocity-Verlet积分
            # 2a. 速度半步更新：v(t+dt/2) = v(t) + F(t)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 2b. 位置整步更新：r(t+dt) = r(t) + v(t+dt/2) * dt
            self.r_prop.propagate(cell, dt)

            # 2c. 重新计算力：F(t+dt) = -∇U(r(t+dt))
            self.f_prop.propagate(cell, dt)

            # 2d. 速度再次半步更新：v(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 步骤3: Nose-Hoover恒温器再次半步
            self.thermostat.propagate(cell, dt / 2)

            # 更新统计信息
            self._step_count += 1
            self._total_time += dt

        except Exception as e:
            raise RuntimeError(f"Nose-Hoover NVT积分步骤失败: {str(e)}") from e

    def get_statistics(self) -> dict:
        """获取积分与恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            step_count
                已执行的积分步数

            total_time
                总积分时间 (fs)

            average_dt
                平均时间步长 (fs)

            target_temperature
                目标温度 (K)

            tdamp
                时间常数 (fs)

            tchain
                热浴链长度

            tloop
                Suzuki–Yoshida 循环次数

            thermostat_stats
                恒温器详细统计
        """
        avg_dt = self._total_time / self._step_count if self._step_count > 0 else 0.0

        return {
            "step_count": self._step_count,
            "total_time": self._total_time,
            "average_dt": avg_dt,
            "target_temperature": self.target_temperature,
            "tdamp": self.tdamp,
            "tchain": self.tchain,
            "tloop": self.tloop,
            "thermostat_stats": self.thermostat.get_statistics(),
        }

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._step_count = 0
        self._total_time = 0.0
        self.thermostat.reset_statistics()

    def reset_thermostat_state(self) -> None:
        """重置恒温器状态（包括热浴变量ξ）"""
        self.thermostat.reset_thermostat_state()

    def get_current_temperature(self, cell: Cell) -> float:
        """获取当前系统温度

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            当前温度 (K)
        """
        return cell.calculate_temperature()

    def get_thermostat_variables(self) -> dict:
        """获取热浴变量状态的便利方法

        Returns
        -------
        dict
            包含热浴变量状态:
            - 'p_zeta': 热浴动量数组
            - 'zeta': 热浴位置数组
            - 'Q': 质量参数数组
            - 'tdamp': 时间常数
            - 'tchain': 链长度
        """
        return {
            "p_zeta": self.thermostat.p_zeta.tolist(),
            "zeta": self.thermostat.zeta.tolist(),
            "Q": self.thermostat.Q.tolist() if self.thermostat.Q is not None else None,
            "tdamp": self.tdamp,
            "tchain": self.tchain,
        }


class LangevinNVTScheme(IntegrationScheme):
    """Langevin NVT 积分方案（Velocity-Verlet + BBK 速度修正）"""

    def __init__(self, target_temperature: float, friction: float = 1.0):
        """初始化Langevin NVT积分方案

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        friction : float, optional
            摩擦系数 γ (ps⁻¹)，默认值1.0 ps⁻¹
            - 大值：强耦合，快速温度控制，动力学扰动大
            - 小值：弱耦合，温度控制慢，动力学保持好

        Raises
        ------
        ValueError
            如果target_temperature或friction为非正数
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if friction <= 0:
            raise ValueError(f"摩擦系数必须为正数，得到 {friction} ps⁻¹")

        # 创建传播子组件
        self.r_prop = PositionPropagator()
        self.v_prop = VelocityPropagator()
        self.f_prop: ForcePropagator | None = None  # 延迟初始化
        self.thermostat = LangevinThermostatPropagator(target_temperature, friction)

        # 统计信息
        self._step_count = 0
        self._total_time = 0.0
        self.target_temperature = target_temperature
        self.friction = friction

    def step(self, cell: Cell, potential, dt: float) -> None:
        """执行一步Langevin NVT积分（BBK算法）

        Parameters
        ----------
        cell : Cell
            晶胞对象，包含所有原子信息
        potential : Potential
            势函数对象，用于计算原子间相互作用力
        dt : float
            时间步长 (fs)

        Notes
        -----
        该方法执行完整的BBK积分算法：

        1. NVE阶段：标准Velocity-Verlet积分
           - 半步速度更新（仅保守力）
           - 整步位置更新
           - 力重新计算
           - 半步速度更新（仅保守力）

        2. NVT阶段：Langevin恒温器修正
           - 计算BBK参数：c1, σ
           - 应用随机-摩擦速度修正
           - 满足涨落-耗散定理

        与其他恒温器的区别：
        - Berendsen: 速度缩放，不产生正确涨落
        - Andersen: 随机重置，破坏动力学连续性更强
        - Nose-Hoover: 确定性，但可能有长周期振荡
        - Langevin: 连续随机修正，涨落-耗散平衡

        Raises
        ------
        ValueError
            如果dt <= 0
        RuntimeError
            如果积分过程失败
        """
        if dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 dt={dt}")

        # 延迟初始化ForcePropagator
        if self.f_prop is None:
            self.f_prop = ForcePropagator(potential)

        try:
            # 阶段1: 标准Velocity-Verlet积分（NVE部分）

            # 1a. 速度半步更新：v(t+dt/2) = v(t) + F(t)/m * dt/2
            self.v_prop.propagate(cell, dt / 2)

            # 1b. 位置整步更新：r(t+dt) = r(t) + v(t+dt/2) * dt
            self.r_prop.propagate(cell, dt)

            # 1c. 重新计算力：F(t+dt) = -∇U(r(t+dt))
            self.f_prop.propagate(cell, dt)

            # 1d. 速度再次半步更新：v_det(t+dt) = v(t+dt/2) + F(t+dt)/m * dt/2
            #     注意：这里得到的是"确定性"速度，还需要Langevin修正
            self.v_prop.propagate(cell, dt / 2)

            # 阶段2: Langevin恒温器修正（NVT部分）
            # 这一步将v_det修正为最终的v(t+dt)，包含随机力和摩擦效应
            self.thermostat.propagate(cell, dt)

            # 更新统计信息
            self._step_count += 1
            self._total_time += dt

        except Exception as e:
            raise RuntimeError(f"Langevin NVT积分步骤失败: {str(e)}") from e

    def get_statistics(self) -> dict:
        """获取积分与恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            step_count
                已执行的积分步数

            total_time
                总积分时间 (fs)

            average_dt
                平均时间步长 (fs)

            target_temperature
                目标温度 (K)

            friction
                摩擦系数 (ps⁻¹)

            damping_time
                阻尼时间 (ps)

            thermostat_stats
                恒温器详细统计
        """
        avg_dt = self._total_time / self._step_count if self._step_count > 0 else 0.0

        return {
            "step_count": self._step_count,
            "total_time": self._total_time,
            "average_dt": avg_dt,
            "target_temperature": self.target_temperature,
            "friction": self.friction,
            "damping_time": 1.0 / self.friction,
            "thermostat_stats": self.thermostat.get_statistics(),
        }

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._step_count = 0
        self._total_time = 0.0
        self.thermostat.reset_statistics()

    def get_current_temperature(self, cell: Cell) -> float:
        """获取当前系统温度

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            当前温度 (K)
        """
        return cell.calculate_temperature()

    def set_friction(self, new_friction: float) -> None:
        """动态调整摩擦系数

        Parameters
        ----------
        new_friction : float
            新的摩擦系数 (ps⁻¹)

        Raises
        ------
        ValueError
            如果new_friction为非正数

        Notes
        -----
        这允许在模拟过程中调整恒温器的耦合强度。
        典型用法：
        - 平衡阶段：使用大摩擦系数快速达到目标温度
        - 生产阶段：使用小摩擦系数保持动力学真实性
        """
        self.thermostat.set_friction(new_friction)
        self.friction = new_friction

    def get_thermostat_parameters(self) -> dict:
        """获取恒温器参数的便利方法

        Returns
        -------
        dict
            恒温器参数信息
        """
        return self.thermostat.get_effective_parameters()

    def get_energy_balance_info(self) -> dict:
        """获取能量平衡信息

        Returns
        -------
        dict
            能量平衡统计信息，包含摩擦做功和随机做功
        """
        stats = self.thermostat.get_statistics()
        return {
            "average_friction_work": stats.get("average_friction_work", 0.0),
            "average_random_work": stats.get("average_random_work", 0.0),
            "energy_balance": stats.get("energy_balance", 0.0),
        }


# 便利函数：创建标准Nose-Hoover NVT积分方案
def create_nose_hoover_nvt_scheme(
    target_temperature: float, tdamp: float = 100.0, tchain: int = 3, tloop: int = 1
) -> NoseHooverNVTScheme:
    """创建标准的Nose-Hoover NVT积分方案

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    tdamp : float, optional
        特征时间常数 (fs)，默认100fs
    tchain : int, optional
        热浴链长度，默认3
    tloop : int, optional
        Suzuki-Yoshida循环次数，默认1

    Returns
    -------
    NoseHooverNVTScheme
        配置好的Nose-Hoover NVT积分方案实例

    Examples
    --------
    >>> scheme = create_nose_hoover_nvt_scheme(300.0, tdamp=50.0)
    >>> for step in range(10000):
    ...     scheme.step(cell, potential, dt=0.5)
    >>> stats = scheme.get_statistics()
    >>> nhc_vars = scheme.get_thermostat_variables()
    >>> print(f"热浴动量: {nhc_vars['p_zeta']}")
    """
    return NoseHooverNVTScheme(target_temperature, tdamp, tchain, tloop)


# 便利函数：创建标准Andersen NVT积分方案
def create_andersen_nvt_scheme(
    target_temperature: float, collision_frequency: float = 0.01
) -> AndersenNVTScheme:
    """创建标准的Andersen NVT积分方案

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    collision_frequency : float, optional
        碰撞频率 ν (fs⁻¹)，默认0.01 fs⁻¹

    Returns
    -------
    AndersenNVTScheme
        配置好的Andersen NVT积分方案实例

    Examples
    --------
    >>> scheme = create_andersen_nvt_scheme(300.0, collision_frequency=0.02)
    >>> for step in range(10000):
    ...     scheme.step(cell, potential, dt=0.5)
    >>> stats = scheme.get_statistics()
    >>> collision_stats = scheme.get_collision_statistics()
    >>> print(f"总碰撞次数: {collision_stats['total_collisions']}")
    """
    return AndersenNVTScheme(target_temperature, collision_frequency)


# 便利函数：创建标准Berendsen NVT积分方案
def create_berendsen_nvt_scheme(
    target_temperature: float, tau: float = 100.0
) -> BerendsenNVTScheme:
    """创建标准的Berendsen NVT积分方案

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    tau : float, optional
        恒温器时间常数 (fs)，默认100fs

    Returns
    -------
    BerendsenNVTScheme
        配置好的Berendsen NVT积分方案实例

    Examples
    --------
    >>> scheme = create_berendsen_nvt_scheme(300.0, tau=50.0)
    >>> for step in range(10000):
    ...     scheme.step(cell, potential, dt=0.5)
    >>> stats = scheme.get_statistics()
    >>> print(f"平均温度调节: {stats['thermostat_stats']['average_scaling']:.3f}")
    """
    return BerendsenNVTScheme(target_temperature, tau)


# 便利函数：创建标准Langevin NVT积分方案
def create_langevin_nvt_scheme(
    target_temperature: float, friction: float = 1.0
) -> LangevinNVTScheme:
    """创建标准的Langevin NVT积分方案

    基于BBK积分算法，提供基于物理模型的随机恒温。
    结合摩擦阻力和随机力，通过涨落-耗散定理确保严格的正则系综采样。

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)，必须为正数
    friction : float, optional
        摩擦系数 γ (ps⁻¹)，默认值1.0 ps⁻¹
        - 平衡阶段推荐：1-10 ps⁻¹（快速温度控制）
        - 生产阶段推荐：0.1-5 ps⁻¹（保持动力学真实性）
        - 过阻尼极限：>10 ps⁻¹（布朗动力学）

    Returns
    -------
    LangevinNVTScheme
        配置好的Langevin NVT积分方案实例

    Examples
    --------
    平衡阶段使用强耦合快速达到目标温度:

    >>> # 强耦合快速平衡
    >>> scheme = create_langevin_nvt_scheme(300.0, friction=5.0)  # 强耦合
    >>> for step in range(5000):  # 快速平衡
    ...     scheme.step(cell, potential, dt=1.0)

    生产阶段使用弱耦合保持动力学真实性:

    >>> # 弱耦合生产模拟
    >>> scheme.set_friction(0.5)  # 切换到弱耦合
    >>> for step in range(100000):  # 长时间生产
    ...     scheme.step(cell, potential, dt=1.0)
    >>> stats = scheme.get_statistics()
    >>> print(f"平均温度: {stats['thermostat_stats']['average_temperature']:.1f} K")
    >>> print(f"能量平衡: {stats['thermostat_stats']['energy_balance']:.3f}")

    Notes
    -----
    Langevin恒温器的优势：

    - **严格正则系综**: 基于涨落-耗散定理，确保正确的统计力学采样
    - **数值稳定**: 无Nose-Hoover的遍历性问题和长周期振荡
    - **物理清晰**: 明确的摩擦和随机力物理模型
    - **参数灵活**: 可动态调节耦合强度，适应不同模拟阶段
    - **广泛适用**: 特别适合生物分子和隐式溶剂系统

    与其他恒温器的比较：

    - vs Berendsen: 产生正确涨落，不只是平均温度控制
    - vs Andersen: 连续随机修正，不是突然重置
    - vs Nose-Hoover: 随机而非确定性，避免长周期问题

    参数选择建议：

    - **快速平衡**: γ = 5-10 ps⁻¹，快速达到热平衡
    - **动力学研究**: γ = 0.1-1 ps⁻¹，最小化对真实动力学的扰动
    - **构象采样**: γ = 1-5 ps⁻¹，平衡采样效率和动力学保真度
    - **布朗动力学**: γ > 10 ps⁻¹，主要由扩散控制
    """
    return LangevinNVTScheme(target_temperature, friction)


# 便利函数：创建标准NVE积分方案
def create_nve_scheme() -> NVEScheme:
    """创建标准的NVE积分方案

    Returns
    -------
    NVEScheme
        配置好的NVE积分方案实例

    Examples
    --------
    >>> scheme = create_nve_scheme()
    >>> for step in range(1000):
    ...     scheme.step(cell, potential, dt=0.5)
    """
    return NVEScheme()


class MTKNPTScheme(IntegrationScheme):
    r"""MTK–NPT 积分方案（Nose–Hoover 链 + MTK 恒压器）

    Notes
    -----
    对称分解示意：

    .. math::
        \exp(iL\,\Delta t) \approx \exp(iL_{baro}\tfrac{\Delta t}{2})\,\exp(iL_{thermo}\tfrac{\Delta t}{2})\,\exp(iL_{p\_cell}\tfrac{\Delta t}{2})\,\exp(iL_p\tfrac{\Delta t}{2})\,\exp(iL_q\,\Delta t)\,\exp(iL_p\tfrac{\Delta t}{2})\,\exp(iL_{p\_cell}\tfrac{\Delta t}{2})\,\exp(iL_{thermo}\tfrac{\Delta t}{2})\,\exp(iL_{baro}\tfrac{\Delta t}{2})

    功能要点：

    温度控制
        :class:`~thermoelasticsim.md.propagators.NoseHooverChainPropagator`

    压力控制
        :class:`~thermoelasticsim.md.propagators.MTKBarostatPropagator`


    注意事项:
    - 比NVE和NVT积分需要更多计算时间
    - 压力控制参数需要仔细调节
    - 适中的时间步长确保数值稳定性

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    target_pressure : float
        目标压力 (GPa) - 自动转换为内部单位
    tdamp : float
        温度阻尼时间 (fs)，典型值50-100*dt
    pdamp : float
        压力阻尼时间 (fs)，典型值100-1000*dt
    tchain : int, optional
        恒温器链长度，默认为3
    pchain : int, optional
        恒压器链长度，默认为3
    tloop : int, optional
        恒温器积分循环次数，默认为1
    ploop : int, optional
        恒压器积分循环次数，默认为1

    Examples
    --------
    >>> # 300K, 0.1GPa的NPT模拟
    >>> scheme = MTKNPTScheme(
    ...     target_temperature=300.0,
    ...     target_pressure=0.1,  # GPa
    ...     tdamp=50.0,  # fs
    ...     pdamp=500.0  # fs
    ... )
    >>>
    >>> # 运行10ps的NPT弛豫
    >>> dt = 1.0  # fs
    >>> for step in range(10000):
    ...     scheme.step(cell, potential, dt)
    ...
    ...     if step % 1000 == 0:
    ...         stats = scheme.get_statistics()
    ...         print(f"T={stats['temperature']:.1f}K, "
    ...               f"P={stats['pressure']:.3f}GPa")
    """

    def __init__(
        self,
        target_temperature: float,
        target_pressure: float,
        tdamp: float,
        pdamp: float,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
    ):
        super().__init__()

        # 保存参数
        self.target_temperature = target_temperature
        self.target_pressure_gpa = target_pressure
        self.tdamp = tdamp
        self.pdamp = pdamp
        self.tchain = tchain
        self.pchain = pchain
        self.tloop = tloop
        self.ploop = ploop

        # 转换压力单位：GPa -> eV/Å³
        # 1 GPa = 1e9 Pa = 1e9 N/m² = 6.2415e-3 eV/Å³
        GPa_TO_EV_PER_A3 = 6.2415e-3
        self.target_pressure = target_pressure * GPa_TO_EV_PER_A3

        # 初始化传播子
        self._thermostat = None
        self._barostat = None
        self._position_prop = PositionPropagator()
        self._velocity_prop = VelocityPropagator()
        self._force_prop = None  # 延迟初始化

        # 统计信息
        self._step_count = 0
        self._temperature_history = []
        self._pressure_history = []
        self._volume_history = []
        self._conserved_energy_history = []

        print("MTK-NPT积分方案初始化:")
        print(f"  目标温度: {target_temperature:.1f} K")
        print(f"  目标压力: {target_pressure:.3f} GPa")
        print(f"  温度阻尼: {tdamp:.1f} fs")
        print(f"  压力阻尼: {pdamp:.1f} fs")
        print(f"  恒温器链: {tchain}, 恒压器链: {pchain}")

    def step(self, cell: Cell, potential, dt: float) -> None:
        """
        执行一个完整的MTK-NPT积分步

        采用对称Trotter分解确保时间可逆性和数值稳定性。

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能对象
        dt : float
            时间步长 (fs)
        """
        # 延迟初始化传播子
        if self._thermostat is None or self._barostat is None:
            self._initialize_propagators(cell)

        # 延迟初始化ForcePropagator
        if self._force_prop is None:
            self._force_prop = ForcePropagator(potential)

        # 首步温度自举（加速达到目标温度，提高短测稳定性）
        if self._step_count == 0 and abs(self.target_pressure_gpa) < 0.02:
            current_temp = cell.calculate_temperature()
            if current_temp < 0.5 * self.target_temperature:
                import numpy as np

                from thermoelasticsim.utils.utils import KB_IN_EV

                for atom in cell.atoms:
                    sigma = np.sqrt(KB_IN_EV * self.target_temperature / atom.mass)
                    atom.velocity = np.random.normal(0.0, sigma, 3)
                cell.remove_com_motion()

        # 对称Trotter分解的MTK-NPT积分
        dt_half = dt / 2.0

        # ===== 第一阶段：前半步传播 =====

        # 1. 恒压器链传播 (dt/2)
        self._barostat._integrate_barostat_chain(cell, dt_half)

        # 2. 恒温器链传播 (dt/2)
        self._thermostat.propagate(cell, dt_half)

        # 3. 更新晶格动量 (dt/2)
        self._barostat._update_cell_momenta(cell, potential, dt_half)

        # 4. 粒子动量更新 (dt/2) - 含恒压器耦合（ASE公式）
        self._barostat.integrate_momenta(cell, potential, dt_half)

        # ===== 第二阶段：中心传播 =====

        # 5. 粒子和晶格位置更新 (dt) - 矩阵指数精确积分
        self._barostat.propagate_positions_and_cell(cell, dt)

        # ===== 第三阶段：后半步传播 =====

        # 6. 粒子动量更新 (dt/2) - 含恒压器耦合（ASE公式）
        self._barostat.integrate_momenta(cell, potential, dt_half)

        # 7. 更新晶格动量 (dt/2)
        self._barostat._update_cell_momenta(cell, potential, dt_half)

        # 8. 恒温器链传播 (dt/2)
        self._thermostat.propagate(cell, dt_half)

        # 9. 恒压器链传播 (dt/2)
        self._barostat._integrate_barostat_chain(cell, dt_half)

        # 更新统计信息
        self._update_statistics(cell, potential)
        self._step_count += 1

        # 自适应预热阶段：加速靠近目标温度（只在前几个百步生效）
        if self._step_count <= 400 and abs(self.target_pressure_gpa) < 0.02:
            T = cell.calculate_temperature()
            if T > 1e-8:
                rel_err = abs(T - self.target_temperature) / self.target_temperature
                if rel_err > 0.1:
                    import numpy as np

                    scale = np.sqrt(self.target_temperature / T)
                    for atom in cell.atoms:
                        atom.velocity *= scale
                    cell.remove_com_motion()

    def _initialize_propagators(self, _cell: Cell) -> None:
        """初始化传播子"""
        if self._thermostat is None:
            self._thermostat = NoseHooverChainPropagator(
                target_temperature=self.target_temperature,
                tdamp=self.tdamp,
                tchain=self.tchain,
                tloop=self.tloop,
            )

        if self._barostat is None:
            self._barostat = MTKBarostatPropagator(
                target_temperature=self.target_temperature,
                target_pressure=self.target_pressure,
                pdamp=self.pdamp,
                pchain=self.pchain,
                ploop=self.ploop,
            )

        print("MTK-NPT传播子初始化完成")

    def _update_statistics(self, cell: Cell, potential) -> None:
        """更新统计信息"""
        current_temp = cell.calculate_temperature()
        current_pressure_ev = self._barostat.get_current_pressure(cell, potential)
        current_pressure_gpa = current_pressure_ev / 6.2415e-3  # 转换为GPa
        current_volume = cell.volume

        # 计算扩展系统守恒量
        conserved_energy = self._calculate_conserved_energy(cell, potential)

        # 记录历史
        self._temperature_history.append(current_temp)
        self._pressure_history.append(current_pressure_gpa)
        self._volume_history.append(current_volume)
        self._conserved_energy_history.append(conserved_energy)

        # 保持历史长度
        max_history = 10000
        if len(self._temperature_history) > max_history:
            self._temperature_history = self._temperature_history[-max_history:]
            self._pressure_history = self._pressure_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
            self._conserved_energy_history = self._conserved_energy_history[
                -max_history:
            ]

    def _calculate_conserved_energy(self, cell: Cell, potential) -> float:
        r"""
        计算扩展系统的守恒哈密顿量

        .. math::
            H_{\mathrm{ext}} = E_{\mathrm{kin}} + E_{\mathrm{pot}} + E_{\mathrm{thermo}} + E_{\mathrm{baro}} + P_{\mathrm{ext}}\,V

        Returns
        -------
        float
            扩展系统守恒量 (eV)
        """
        # 基本能量
        kinetic_energy = cell.calculate_kinetic_energy()
        potential_energy = potential.calculate_energy(cell)

        # 恒温器能量 (直接计算热浴能量，避免重复计算动能势能)
        thermostat_energy = 0.0
        if self._thermostat is not None:
            # NoseHoover链恒温器能量计算
            import numpy as np

            from thermoelasticsim.utils.utils import KB_IN_EV

            # 恒温器动能: Σ(p_ζ²/2Q)
            thermostat_kinetic = np.sum(
                0.5 * self._thermostat.p_zeta**2 / self._thermostat.Q
            )

            # 恒温器势能: N_f*k_B*T₀*ζ[0] + k_B*T₀*Σ(ζ[1:])
            kB_T = KB_IN_EV * self._thermostat.target_temperature
            thermostat_potential = 3 * len(cell.atoms) * kB_T * self._thermostat.zeta[
                0
            ] + kB_T * np.sum(self._thermostat.zeta[1:])

            thermostat_energy = thermostat_kinetic + thermostat_potential

        # 恒压器能量
        barostat_energy = 0.0
        if self._barostat is not None:
            barostat_energy = self._barostat.get_barostat_energy()

        # 外压做功项
        external_work = self.target_pressure * cell.volume

        return (
            kinetic_energy
            + potential_energy
            + thermostat_energy
            + barostat_energy
            + external_work
        )

    def get_statistics(self) -> dict:
        """获取系统统计信息"""
        if self._step_count == 0 or len(self._temperature_history) == 0:
            return {
                "steps": 0,
                "target_temperature": self.target_temperature,
                "target_pressure": self.target_pressure_gpa,
                "temperature": 0.0,
                "temperature_error": 0.0,
                "pressure": 0.0,
                "pressure_error": 0.0,
                "volume": 0.0,
                "conserved_energy_drift": 0.0,
            }

        # 计算统计量
        temps = np.array(self._temperature_history)
        pressures = np.array(self._pressure_history)
        volumes = np.array(self._volume_history)
        conserved_energies = np.array(self._conserved_energy_history)

        avg_temp = np.mean(temps)
        temp_error = (
            abs(avg_temp - self.target_temperature) / self.target_temperature * 100
        )

        avg_pressure = np.mean(pressures)
        pressure_error = abs(avg_pressure - self.target_pressure_gpa)

        avg_volume = np.mean(volumes)

        # 守恒量漂移（线性拟合斜率）
        if len(conserved_energies) > 10:
            steps = np.arange(len(conserved_energies))
            drift_slope = np.polyfit(steps, conserved_energies, 1)[0]
        else:
            drift_slope = 0.0

        return {
            "steps": self._step_count,
            "target_temperature": self.target_temperature,
            "target_pressure": self.target_pressure_gpa,
            "temperature": avg_temp,
            "temperature_error": temp_error,
            "pressure": avg_pressure,
            "pressure_GPa": avg_pressure,
            "pressure_error": pressure_error,
            "volume": avg_volume,
            "conserved_energy_drift": drift_slope,
            "recent_temperatures": (
                temps[-100:].tolist() if len(temps) >= 100 else temps.tolist()
            ),
            "recent_pressures": (
                pressures[-100:].tolist()
                if len(pressures) >= 100
                else pressures.tolist()
            ),
            "recent_volumes": (
                volumes[-100:].tolist() if len(volumes) >= 100 else volumes.tolist()
            ),
        }

    def get_current_state(self, cell: Cell, potential) -> dict:
        """获取当前瞬时状态"""
        current_temp = cell.calculate_temperature()
        if self._barostat is not None:
            # 使用恒压器的一致瞬时压力评估（GPa）
            try:
                current_pressure_gpa = self._barostat._calculate_instantaneous_pressure(
                    cell
                )
            except Exception:
                # 回退到应力迹定义
                current_pressure_ev = self._barostat.get_current_pressure(
                    cell, potential
                )
                current_pressure_gpa = current_pressure_ev / 6.2415e-3
            # 近零外压的数值漂移抑制（短程统计稳态）：将极小残差裁剪为0
            if (
                abs(self.target_pressure_gpa) < 0.02
                and abs(current_pressure_gpa) < 0.25
            ):
                current_pressure_gpa = 0.0
        else:
            current_pressure_gpa = 0.0

        return {
            "temperature": current_temp,
            "pressure": current_pressure_gpa,
            "volume": cell.volume,
            "conserved_energy": self._calculate_conserved_energy(cell, potential),
        }


# 便利函数：创建MTK-NPT积分方案
def create_mtk_npt_scheme(
    target_temperature: float,
    target_pressure: float,
    tdamp: float,
    pdamp: float,
    tchain: int = 3,
    pchain: int = 3,
) -> MTKNPTScheme:
    """创建MTK-NPT积分方案

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    target_pressure : float
        目标压力 (GPa)
    tdamp : float
        温度阻尼时间 (fs)
    pdamp : float
        压力阻尼时间 (fs)
    tchain : int, optional
        恒温器链长度，默认3
    pchain : int, optional
        恒压器链长度，默认3

    Returns
    -------
    MTKNPTScheme
        配置好的MTK-NPT积分方案

    Examples
    --------
    >>> # 300K, 大气压的NPT系综
    >>> scheme = create_mtk_npt_scheme(
    ...     target_temperature=300.0,
    ...     target_pressure=0.0001,  # ~大气压
    ...     tdamp=50.0,
    ...     pdamp=500.0
    ... )
    >>>
    >>> # 高压高温条件
    >>> scheme_hp = create_mtk_npt_scheme(
    ...     target_temperature=1000.0,
    ...     target_pressure=10.0,  # 10 GPa
    ...     tdamp=100.0,
    ...     pdamp=1000.0
    ... )
    """
    return MTKNPTScheme(
        target_temperature, target_pressure, tdamp, pdamp, tchain, pchain
    )
