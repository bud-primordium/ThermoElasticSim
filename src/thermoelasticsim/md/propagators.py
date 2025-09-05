r"""基础算符传播子实现

每个传播子对应一个基本物理过程，可通过对称 Trotter 分解组合为完整积分方案。

已实现的传播子
--------------

PositionPropagator
    位置演化：:math:`\mathbf{r} \leftarrow \mathbf{r} + \mathbf{v}\,\Delta t`

VelocityPropagator
    速度演化：:math:`\mathbf{v} \leftarrow \mathbf{v} + \mathbf{F}\,\Delta t/m`

ForcePropagator
    力计算：:math:`\mathbf{F} = -\nabla U(\mathbf{r})`

设计要点
--------
1. 单一职责：每个传播子仅负责一种过程
2. 与 Velocity-Verlet 分解一致，减少重复力计算
3. 数值稳定、支持时间可逆的对称分解
"""

from typing import Any

import numpy as np

from ..core.structure import Cell
from ..elastic.mechanics import StressCalculator
from ..utils.utils import EV_TO_GPA, KB_IN_EV
from .interfaces import Propagator

# ====================================================================
# 四阶Suzuki-Yoshida分解系数 (High-Precision Integration)
# ====================================================================
# Ref: H. Yoshida, Phys. Lett. A 150, 262-268 (1990)
#      https://doi.org/10.1016/0375-9601(90)90092-3
#
# 这些系数用于高精度的扩展系统积分，特别是Nose-Hoover链恒温器。
# 四阶方案相比二阶具有显著更好的长期稳定性和精度。
FOURTH_ORDER_COEFFS = [
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),  # w1 = 1.351207191959657...
    -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),  # w2 = -1.702414383919315...
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),  # w3 = 1.351207191959657...
]

# 验证系数：w1 + w2 + w3 = 1.0 (必须严格满足)
_coeff_sum = sum(FOURTH_ORDER_COEFFS)
if abs(_coeff_sum - 1.0) > 1e-14:
    raise ValueError(f"Suzuki-Yoshida系数和验证失败: {_coeff_sum} != 1.0")

# 精确值记录（用于文档和验证）
_W1 = _W3 = 1.3512071919596578  # 1/(2-2^(1/3))
_W2 = -1.7024143839193153  # -2^(1/3)/(2-2^(1/3))


class PositionPropagator(Propagator):
    r"""位置传播子：:math:`\exp(iL_r\,\Delta t)`

    实现位置的时间演化：:math:`\mathbf{r}(t+\Delta t)=\mathbf{r}(t)+\mathbf{v}(t)\,\Delta t`。

    Notes
    -----
    处理周期性边界；不改变速度与力；与势能无关。
    """

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        """执行位置更新

        Parameters
        ----------
        cell : Cell
            晶胞对象，将直接修改其中原子的位置
        dt : float
            时间步长 (fs)
        **kwargs : Any
            未使用，保持接口一致性
        """
        for atom in cell.atoms:
            # 位置更新：r(t+dt) = r(t) + v(t)*dt
            atom.position += atom.velocity * dt

            # 应用周期性边界条件
            atom.position = cell.apply_periodic_boundary(atom.position)


class VelocityPropagator(Propagator):
    r"""速度传播子：:math:`\exp(iL_v\,\Delta t)`

    实现速度的时间演化：:math:`\mathbf{v}(t+\Delta t)=\mathbf{v}(t)+\mathbf{F}(t)\,\Delta t/m`。

    Notes
    -----
    假设 :code:`atom.force` 已计算；不在此处重新计算力。
    """

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        """执行速度更新

        Parameters
        ----------
        cell : Cell
            晶胞对象，将直接修改其中原子的速度
        dt : float
            时间步长 (fs)
        **kwargs : Any
            未使用，保持接口一致性

        Notes
        -----
        该方法假设所有原子的力(atom.force)已经正确计算。
        如果力未计算或已过期，结果将不正确。
        """
        for atom in cell.atoms:
            # 速度更新：v(t+dt) = v(t) + F(t)/m*dt
            atom.velocity += atom.force / atom.mass * dt


class ForcePropagator(Propagator):
    r"""力计算传播子：:math:`\mathbf{F} = -\nabla U(\mathbf{r})`

    封装势函数的力计算调用，控制力计算时机，避免重复计算。
    """

    def __init__(self, potential):
        """初始化力传播子

        Parameters
        ----------
        potential : Potential
            势函数对象，必须实现calculate_forces方法
        """
        self.potential = potential

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        """计算并更新所有原子的力

        Parameters
        ----------
        cell : Cell
            晶胞对象，将更新其中所有原子的force属性
        dt : float
            时间步长 (fs)，在此方法中未使用，仅为接口一致性
        **kwargs : Any
            额外参数，可能包含势函数参数等

        Notes
        -----
        该方法将：
        1. 调用势函数计算所有原子间相互作用
        2. 更新每个原子的force属性
        3. 确保力的单位和符号正确

        如果势函数计算失败，将抛出相应异常。
        """
        # dt参数在此处未使用，仅为保持接口一致性
        # 调用势函数计算力
        self.potential.calculate_forces(cell)


class BerendsenThermostatPropagator(Propagator):
    r"""Berendsen 恒温器传播子（速度缩放）

    缩放因子 :math:`\lambda`：

    .. math::
        \lambda = \sqrt{1 + \frac{\Delta t}{\tau}\left(\frac{T_0}{T}-1\right)}

    Notes
    -----
    后处理式恒温；不严格产生正则分布，常用于快速平衡。

    References
    ----------
    Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R. (1984).
    Molecular dynamics with coupling to an external bath. The Journal of Chemical Physics, 81(8), 3684–3690.
    https://doi.org/10.1063/1.448118
    """

    def __init__(
        self, target_temperature: float, tau: float = None, mode: str = "equilibration"
    ):
        r"""初始化 Berendsen 恒温器

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        tau : float, optional
            时间常数 :math:`\tau_T` (fs)。未给定时按 `mode` 选择：

            500.0
                平衡模式（中等耦合，收敛较快）

            2000.0
                生产模式（弱耦合，扰动更小）
        mode : str, optional
            运行模式，'equilibration'(平衡) 或 'production'(生产)
            影响默认τ_T值的选择

        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if mode not in ["equilibration", "production"]:
            raise ValueError(f"mode必须是'equilibration'或'production'，得到 {mode}")

        self.target_temperature = target_temperature
        self.mode = mode

        # 默认 τ_T 参数选择
        if tau is None:
            if mode == "equilibration":
                self.tau = 500.0  # fs，快速平衡
            else:  # production
                self.tau = 2000.0  # fs，弱耦合生产
        else:
            if tau <= 0:
                raise ValueError(f"时间常数必须为正数，得到 {tau} fs")
            self.tau = tau

        print(
            f"Berendsen恒温器初始化: T={target_temperature}K, τ_T={self.tau}fs, mode={mode}"
        )

        # 统计信息
        self._total_scaling_steps = 0
        self._cumulative_scaling = 0.0
        self._max_scaling_factor = 1.0
        self._min_scaling_factor = 1.0
        self._limited_steps = 0  # 被限制的步数
        self._upper_limited = 0  # 上限限制次数
        self._lower_limited = 0  # 下限限制次数

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        r"""执行 Berendsen 温度调节

        Parameters
        ----------
        cell : Cell
            晶胞对象，将修改其中所有原子的速度
        dt : float
            时间步长 (fs)
        **kwargs : Any
            额外参数，未使用

        Notes
        -----
        标准流程：

        1. 计算瞬时温度 :math:`T`
        2. 计算缩放因子 :math:`\lambda = \sqrt{1 + \tfrac{\Delta t}{\tau_T}(\tfrac{T_0}{T}-1)}`
        3. 对所有原子速度缩放 :math:`\mathbf{v} \leftarrow \lambda\,\mathbf{v}`
        4. 可选：数值稳定性限制与统计监控
        """
        # 根据研究报告算法步骤：计算当前系统瞬时温度
        current_temp = cell.calculate_temperature()

        # 特殊情况处理：如果温度接近0（无运动），初始化Maxwell分布
        if current_temp < 1e-6:  # 接近0K
            print("⚠️  检测到零温度，初始化Maxwell分布速度")
            self._initialize_maxwell_velocities(cell)
            current_temp = cell.calculate_temperature()

        # 数值稳定性保护：避免除零错误，设置最小温度阈值
        # 根据研究报告，这是必要的数值保护措施
        if current_temp < 0.1:  # 最小0.1K
            current_temp = 0.1

        # 根据研究报告的Berendsen温度弛豫方程：
        # dT/dt = (T_0 - T)/τ_T
        # 速度缩放因子: λ = sqrt(1 + dt/τ_T * (T_0/T_current - 1))
        raw_scaling_factor = np.sqrt(
            1.0 + (dt / self.tau) * (self.target_temperature / current_temp - 1.0)
        )

        # ASE风格数值稳定性保护：限制缩放因子在合理范围内
        # 根据研究报告，这防止了过度的速度缩放，提供数值稳定性
        # 放宽限制从0.9-1.1到0.8-1.2，允许更大的温度调节
        scaling_factor = raw_scaling_factor
        limited = False

        if raw_scaling_factor > 1.2:
            scaling_factor = 1.2
            limited = True
            self._upper_limited += 1
        elif raw_scaling_factor < 0.8:
            scaling_factor = 0.8
            limited = True
            self._lower_limited += 1

        # 更新限制统计
        if limited:
            self._limited_steps += 1

        # 增强的监控和警告机制
        if self._total_scaling_steps % 500 == 0:  # 每500步输出统计
            avg_scaling = (
                self._cumulative_scaling / self._total_scaling_steps
                if self._total_scaling_steps > 0
                else 1.0
            )
            limitation_rate = (
                (self._limited_steps / self._total_scaling_steps * 100)
                if self._total_scaling_steps > 0
                else 0.0
            )
            temp_error = (
                abs(current_temp - self.target_temperature)
                / self.target_temperature
                * 100
            )

            print(
                f"Berendsen统计 (#{self._total_scaling_steps}步): T={current_temp:.1f}K, 误差={temp_error:.2f}%, 平均缩放={avg_scaling:.3f}"
            )
            print(
                f"  限制率: {limitation_rate:.2f}% (上限:{self._upper_limited}, 下限:{self._lower_limited})"
            )

        if limited and self._total_scaling_steps % 100 == 0:
            print(
                f"⚠️  Berendsen缩放因子被限制: 原始={raw_scaling_factor:.3f}, 限制后={scaling_factor:.3f}"
            )
            print(
                f"   当前温度={current_temp:.1f}K, 目标温度={self.target_temperature}K, τ_T={self.tau}fs"
            )

        # 应用速度缩放： v_new = λ * v_old
        for atom in cell.atoms:
            atom.velocity *= scaling_factor

        # 更新统计信息
        self._total_scaling_steps += 1
        self._cumulative_scaling += scaling_factor
        self._max_scaling_factor = max(self._max_scaling_factor, scaling_factor)
        self._min_scaling_factor = min(self._min_scaling_factor, scaling_factor)

    def _initialize_maxwell_velocities(self, cell: Cell) -> None:
        """初始化Maxwell分布速度"""
        for atom in cell.atoms:
            # Maxwell分布标准差: σ = sqrt(kB*T/m)
            sigma = np.sqrt(KB_IN_EV * self.target_temperature / atom.mass)
            atom.velocity = np.random.normal(0, sigma, 3)

        # 移除质心运动
        cell.remove_com_motion()

    def get_statistics(self) -> dict:
        r"""获取 Berendsen 恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            total_steps
                总调节步数

            average_scaling
                平均缩放因子

            max_scaling
                最大缩放因子

            min_scaling
                最小缩放因子

            target_temperature
                目标温度 (K)

            tau
                时间常数 :math:`\tau_T` (fs)

            mode
                运行模式（equilibration/production）

            limited_steps
                被限制的步数

            upper_limited / lower_limited
                上/下限限制次数

            limitation_rate
                限制比例 (%)

            effective_coupling
                有效耦合强度评估
        """
        if self._total_scaling_steps == 0:
            return {
                "total_steps": 0,
                "average_scaling": 1.0,
                "max_scaling": 1.0,
                "min_scaling": 1.0,
                "target_temperature": self.target_temperature,
                "tau": self.tau,
                "mode": self.mode,
                "limited_steps": 0,
                "upper_limited": 0,
                "lower_limited": 0,
                "limitation_rate": 0.0,
                "effective_coupling": "unknown",
            }

        avg_scaling = self._cumulative_scaling / self._total_scaling_steps
        limitation_rate = self._limited_steps / self._total_scaling_steps * 100

        # 评估有效耦合强度
        scaling_deviation = abs(avg_scaling - 1.0)
        if scaling_deviation < 0.01:
            effective_coupling = "weak"  # 接近NVE
        elif scaling_deviation < 0.05:
            effective_coupling = "moderate"  # 中等耦合
        else:
            effective_coupling = "strong"  # 强耦合

        return {
            "total_steps": self._total_scaling_steps,
            "average_scaling": avg_scaling,
            "max_scaling": self._max_scaling_factor,
            "min_scaling": self._min_scaling_factor,
            "target_temperature": self.target_temperature,
            "tau": self.tau,
            "mode": self.mode,
            "limited_steps": self._limited_steps,
            "upper_limited": self._upper_limited,
            "lower_limited": self._lower_limited,
            "limitation_rate": limitation_rate,
            "effective_coupling": effective_coupling,
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self._total_scaling_steps = 0
        self._cumulative_scaling = 0.0
        self._max_scaling_factor = 1.0
        self._min_scaling_factor = 1.0
        self._limited_steps = 0
        self._upper_limited = 0
        self._lower_limited = 0


class AndersenThermostatPropagator(Propagator):
    r"""Andersen 随机碰撞恒温器传播子

    每个原子在每个时间步以概率 :math:`p=\nu\,\Delta t` 与热浴碰撞，
    并从 Maxwell–Boltzmann 分布重新采样速度。

    Notes
    -----
    产生正确的正则 (NVT) 分布，但会打断动力学连续性。

    References
    ----------
    Andersen, H. C. (1980). Molecular dynamics simulations at constant pressure and/or temperature.
    The Journal of Chemical Physics, 72(4), 2384–2393. https://doi.org/10.1063/1.439486
    """

    def __init__(self, target_temperature: float, collision_frequency: float = None):
        r"""初始化 Andersen 恒温器

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)
        collision_frequency : float, optional
            碰撞频率 :math:`\nu` (fs⁻¹)。未给定时按系统尺寸选取：

            小系统 (≤ 50 原子)
                0.005 fs⁻¹

            中系统 (50–150 原子)
                0.03 fs⁻¹

            大系统 (> 150 原子)
                0.08 fs⁻¹

        Raises
        ------
        ValueError
            如果target_temperature或collision_frequency为非正数
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if collision_frequency is not None and collision_frequency <= 0:
            raise ValueError(f"碰撞频率必须为正数，得到 {collision_frequency} fs⁻¹")

        self.target_temperature = target_temperature
        self.base_collision_frequency = collision_frequency  # 保存用户设置的基础频率

        # 延迟初始化实际使用的频率（需要知道系统尺寸）
        self._effective_collision_frequency = collision_frequency

        # 统计信息
        self._total_steps = 0
        self._total_collisions = 0
        self._collision_history = []
        self._system_size_initialized = False

    def _initialize_system_size_adaptive_frequency(self, num_atoms: int) -> float:
        """根据系统尺寸初始化适应性频率

        根据研究报告的经验公式和实际测试结果，
        不同尺寸系统需要不同的碰撞频率以保持有效控温。

        Parameters
        ----------
        num_atoms : int
            系统中的原子数

        Returns
        -------
        float
            适合该系统尺寸的碰撞频率 (fs⁻¹)
        """
        if self.base_collision_frequency is not None:
            # 用户指定了频率，但仍需要根据系统尺寸调整
            base_freq = self.base_collision_frequency
        else:
            # 根据系统尺寸选择基础频率
            if num_atoms <= 50:
                base_freq = 0.005  # 小系统
            elif num_atoms <= 150:
                base_freq = 0.03  # 中系统
            else:
                base_freq = 0.08  # 大系统

        print(
            f"Andersen恒温器初始化: {num_atoms}原子系统, 选择频率={base_freq:.5f} fs⁻¹"
        )
        self._system_size_initialized = True
        return base_freq

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        r"""执行 Andersen 随机碰撞

        Parameters
        ----------
        cell : Cell
            晶胞对象，将修改其中部分原子的速度
        dt : float
            时间步长 (fs)
        **kwargs : Any
            额外参数，未使用

        Notes
        -----
        1. 计算系统级碰撞概率 :math:`P=\nu\,\Delta t`
        2. 逐原子以概率 :math:`P/N` 判定是否碰撞
        3. 若碰撞，从 Maxwell–Boltzmann 分布采样新速度
        """
        num_atoms = len(cell.atoms)

        # 延迟初始化：根据系统尺寸确定碰撞频率
        if not self._system_size_initialized:
            self._effective_collision_frequency = (
                self._initialize_system_size_adaptive_frequency(num_atoms)
            )

        # 计算当前系统温度
        current_temp = cell.calculate_temperature()

        # 系统级碰撞概率计算 (关键修复!)
        # 根据研究报告: P_collision = ν × Δt
        # 这是整个系统的碰撞概率，不是单个原子的
        system_collision_probability = self._effective_collision_frequency * dt

        # 限制碰撞概率在合理范围内
        system_collision_probability = min(system_collision_probability, 0.5)  # 最多50%

        # 向量化判定每个原子是否发生碰撞
        per_atom_collision_probability = system_collision_probability / num_atoms
        if per_atom_collision_probability < 0:
            per_atom_collision_probability = 0.0
        elif per_atom_collision_probability > 1.0:
            per_atom_collision_probability = 1.0

        # 生成碰撞掩码
        rand = np.random.random(num_atoms)
        mask = rand < per_atom_collision_probability
        step_collisions = int(np.count_nonzero(mask))

        if step_collisions > 0:
            # 为被选中的原子采样Maxwell-Boltzmann速度
            v = cell.get_velocities()
            masses = np.array([a.mass for a in cell.atoms], dtype=np.float64)
            sigmas = np.sqrt(KB_IN_EV * self.target_temperature / masses)
            # 只更新 mask 对应的行
            k = step_collisions
            new_v = np.random.normal(0.0, 1.0, (k, 3)) * sigmas[mask][:, None]
            v[mask, :] = new_v
            # 回写到原子对象
            for i, m in enumerate(mask):
                if m:
                    cell.atoms[i].velocity = v[i]

        # 定期移除质心运动（维持系统稳定性）
        if step_collisions > 0 or self._total_steps % 20 == 0:  # 有碰撞时或每20步
            cell.remove_com_motion()

        # 统计信息监控
        self._total_steps += 1
        self._total_collisions += step_collisions
        self._collision_history.append(step_collisions)

        # 保持历史记录在合理长度
        if len(self._collision_history) > 1000:
            self._collision_history.pop(0)

        # 定期输出统计信息
        if self._total_steps % 1000 == 0:
            actual_collision_rate = self._total_collisions / self._total_steps
            expected_rate = self._effective_collision_frequency
            rate_ratio = (
                (actual_collision_rate / expected_rate) if expected_rate > 0 else 0
            )
            temp_error = (
                abs(current_temp - self.target_temperature)
                / self.target_temperature
                * 100
            )

            print(
                f"Andersen统计 (#{self._total_steps}步): T={current_temp:.1f}K, 误差={temp_error:.2f}%"
            )
            print(
                f"  碰撞率: 实际={actual_collision_rate:.6f}, 期望={expected_rate:.6f}, 比值={rate_ratio:.3f}"
            )

    def _sample_maxwell_boltzmann_velocity(self, atom) -> None:
        """为单个原子采样Maxwell-Boltzmann分布速度（标准算法）

        根据研究报告(Lines 198-202)的标准Maxwell-Boltzmann分布:
        p(v_α) = sqrt(m_i/(2π*k_B*T_0)) * exp(-m_i*v_α²/(2*k_B*T_0))

        每个速度分量是独立的高斯分布，标准差: σ = sqrt(k_B*T/m)

        Parameters
        ----------
        atom : Atom
            要重新采样速度的原子
        """
        # Maxwell分布的标准差: σ = sqrt(k_B*T_0/m)
        # 使用目标温度，不做缓变处理（符合研究报告的标准算法）
        sigma = np.sqrt(KB_IN_EV * self.target_temperature / atom.mass)

        # 从高斯分布采样三维速度
        # 每个分量独立采样，符合Maxwell-Boltzmann理论
        atom.velocity = np.random.normal(0, sigma, 3)

    def get_statistics(self) -> dict:
        """获取 Andersen 恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            total_steps
                总步数

            total_collisions
                总碰撞次数

            collision_rate
                平均碰撞率 (次/步)

            expected_rate
                理论碰撞率 (fs⁻¹)

            target_temperature
                目标温度 (K)

            effective_collision_frequency
                有效碰撞频率 (fs⁻¹)

            base_collision_frequency
                基础频率 (fs⁻¹)

            recent_collisions
                最近 100 步的碰撞数

            system_size_initialized
                是否已根据系统尺寸初始化
        """
        if self._total_steps == 0:
            return {
                "total_steps": 0,
                "total_collisions": 0,
                "collision_rate": 0.0,
                "expected_rate": 0.0,
                "target_temperature": self.target_temperature,
                "effective_collision_frequency": self._effective_collision_frequency
                or 0.0,
                "base_collision_frequency": self.base_collision_frequency or 0.0,
                "recent_collisions": 0,
                "system_size_initialized": self._system_size_initialized,
            }

        collision_rate = self._total_collisions / self._total_steps
        recent_collisions = (
            sum(self._collision_history[-100:]) if self._collision_history else 0
        )

        return {
            "total_steps": self._total_steps,
            "total_collisions": self._total_collisions,
            "collision_rate": collision_rate,
            "expected_rate": self._effective_collision_frequency or 0.0,
            "target_temperature": self.target_temperature,
            "effective_collision_frequency": self._effective_collision_frequency or 0.0,
            "base_collision_frequency": self.base_collision_frequency or 0.0,
            "recent_collisions": recent_collisions,
            "system_size_initialized": self._system_size_initialized,
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self._total_steps = 0
        self._total_collisions = 0
        self._collision_history = []


class NoseHooverChainPropagator(Propagator):
    r"""Nose–Hoover 链恒温器传播子（四阶 Suzuki–Yoshida 分解）

    Notes
    -----
    链式热浴变量自洽演化，时间可逆对称分解，保证正则系综采样质量。

    参考文献：
        Martyna et al., J. Chem. Phys. 97, 2635 (1992); Yoshida, Phys. Lett. A 150, 262 (1990).
    """

    def __init__(
        self,
        target_temperature: float,
        tdamp: float = 100.0,
        tchain: int = 3,
        tloop: int = 1,
    ):
        """初始化Nose-Hoover链恒温器

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)，必须为正数
        tdamp : float, optional
            特征时间常数 (fs)，默认100fs
            推荐值：50-100*dt，控制耦合强度
            过小值(如10*dt)会导致数值不稳定
        tchain : int, optional
            热浴链长度，默认3
            M=3通常足够保证遍历性和稳定性
        tloop : int, optional
            Suzuki-Yoshida循环次数，默认1
            通常1次循环已足够，增加会提升精度但增大计算量

        Raises
        ------
        ValueError
            如果参数设置不合理
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if tdamp <= 0:
            raise ValueError(f"时间常数必须为正数，得到 {tdamp} fs")
        if tchain < 1:
            raise ValueError(f"链长度必须≥1，得到 {tchain}")
        if tloop < 1:
            raise ValueError(f"循环次数必须≥1，得到 {tloop}")

        self.target_temperature = target_temperature
        self.tdamp = tdamp
        self.tchain = tchain
        self.tloop = tloop

        # 质量参数Q矩阵 - 延迟初始化（需要系统信息）
        self.Q = None
        self._num_atoms_global = None
        self._dof = None  # 体系用于恒温器的自由度（与温度统计口径一致）
        self._masses = None  # 原子质量数组（向量化计算用）
        self._initialized = False

        # 热浴变量状态
        self.p_zeta = np.zeros(self.tchain)  # 热浴动量
        self.zeta = np.zeros(self.tchain)  # 热浴位置（用于能量计算）

        # 统计信息
        self._total_steps = 0
        self._temp_history = []
        self._conserved_energy_history = []

    def _initialize_Q_parameters(self, cell: Cell) -> None:
        """初始化质量参数Q矩阵

        标准实现：
        Q[0] = N_f * k_B * T₀ * τ²   (第一个热浴)
        Q[j] = k_B * T₀ * τ²         (后续热浴)

        其中N_f是系统自由度，τ是特征时间常数。

        Parameters
        ----------
        cell : Cell
            晶胞对象，用于计算系统自由度
        """
        self._num_atoms_global = len(cell.atoms)

        # 计算用于恒温器的一致自由度口径：
        # - 单原子: 3
        # - 多原子: 3N-3（扣除质心平动），与 Cell.calculate_temperature 一致
        N_f = 3 if self._num_atoms_global <= 1 else 3 * self._num_atoms_global - 3
        self._dof = N_f
        # 缓存质量数组以便后续向量化计算
        try:
            self._masses = np.array([a.mass for a in cell.atoms], dtype=np.float64)
        except Exception:
            self._masses = None

        # 温度单位转换
        kB_T = KB_IN_EV * self.target_temperature

        # Q参数计算 (ASE公式)
        self.Q = np.zeros(self.tchain)
        self.Q[0] = N_f * kB_T * self.tdamp**2  # 第一个热浴（按一致自由度）
        self.Q[1:] = kB_T * self.tdamp**2  # 后续热浴

        self._initialized = True

        print(
            f"NHC初始化: N_atoms={self._num_atoms_global}, N_f={N_f}, "
            f"T={self.target_temperature}K, τ={self.tdamp}fs"
        )
        print(f"Q参数: Q[0]={self.Q[0]:.3e}, Q[1:]={self.Q[1]:.3e}")

    def _calculate_instantaneous_kinetic_energy(self, cell: Cell) -> float:
        """计算瞬时动能

        计算Σ(p_i²/2m_i)，不移除质心运动。
        在算符分离中间步骤，保持所有动量信息的完整性。

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            总动能 (eV)
        """
        v = cell.get_velocities()  # (N,3)
        m = (
            self._masses
            if self._masses is not None
            else np.array([a.mass for a in cell.atoms], dtype=np.float64)
        )
        vsq = np.einsum("ij,ij->i", v, v)
        return float(0.5 * np.dot(m, vsq))

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        r"""执行 Nose–Hoover 链传播（四阶 Suzuki–Yoshida）

        实现完整的 NHC 算符：

        .. math::
            \exp\big(iL_{\mathrm{NHC}}\,\Delta t\big)

        采用三步四阶 Suzuki–Yoshida 分解保证高精度和长期稳定性。

        算法流程：

        1. 遍历每个 SY 系数 :math:`w_k`
        2. 对每个系数执行 :math:`t_{loop}` 次内循环
        3. 调用 :code:`_nhc_integration_loop(w_k \cdot \Delta t / t_{loop})`

        Parameters
        ----------
        cell : Cell
            晶胞对象，将修改原子速度
        dt : float
            传播时间步长 (fs)
        **kwargs : Any
            额外参数（当前未使用，保留接口一致性）

        Notes
        -----
        与 NVE 的对称包裹关系：

        .. math::
            \exp\big(iL_{\mathrm{NHC}}\tfrac{\Delta t}{2}\big)\,\exp\big(iL_{\mathrm{NVE}}\,\Delta t\big)\,\exp\big(iL_{\mathrm{NHC}}\tfrac{\Delta t}{2}\big)
        """
        # 延迟初始化Q参数
        if not self._initialized:
            self._initialize_Q_parameters(cell)

        # 四阶Suzuki-Yoshida主循环
        for _ in range(self.tloop):
            for coeff in FOURTH_ORDER_COEFFS:
                # 每个系数对应的子步长
                sub_delta = coeff * dt / self.tloop
                # 执行单次NHC积分循环
                self._nhc_integration_loop(cell, sub_delta)

        # 更新统计信息
        self._update_statistics(cell)

    def _nhc_integration_loop(self, cell: Cell, delta: float) -> None:
        r"""单次 NHC 积分循环（对称回文序列）

        实现 MTK 方法的核心：对称回文更新保证时间可逆性。

        步骤：

        1. 从后向前传播热浴链（:math:`M-1\to 0`）
        2. 更新热浴位置并缩放粒子速度（热浴耦合）
        3. 从前向后传播热浴链（:math:`0\to M-1`）

        Parameters
        ----------
        cell : Cell
            晶胞对象
        delta : float
            积分子步长
        """
        delta2 = delta / 2.0  # 半步长
        delta4 = delta / 4.0  # 四分之一步长

        # 步骤1: "向前"传播热浴链（M-1, M-2, ..., 0）
        for j in range(self.tchain):
            idx = self.tchain - 1 - j  # 从链末端开始
            self._update_single_thermostat(cell, idx, delta2, delta4)

        # 步骤2: 更新热浴位置（积分p_ζ/Q）
        self.zeta += delta * self.p_zeta / self.Q

        # 步骤3: 缩放粒子速度（关键的热浴耦合）
        # v *= exp(-delta * p_ζ[0] / Q[0])
        scaling_factor = np.exp(-delta * self.p_zeta[0] / self.Q[0])
        for atom in cell.atoms:
            atom.velocity *= scaling_factor

        # 步骤4: "向后"传播热浴链（0, 1, ..., M-1）
        for j in range(self.tchain):
            self._update_single_thermostat(cell, j, delta2, delta4)

    def _update_single_thermostat(
        self, cell: Cell, j: int, delta2: float, delta4: float
    ) -> None:
        """更新单个热浴变量

        实现单个热浴p_ζ[j]的Trotter分解更新，包含：
        1. 邻居热浴的耦合作用（指数缩放）
        2. "力"G_j的作用（线性更新）
        3. 再次邻居热浴的耦合作用

        这个三步对称分解保证算法的时间可逆性。

        Parameters
        ----------
        cell : Cell
            晶胞对象
        j : int
            热浴索引 (0 ≤ j < tchain)
        delta2 : float
            半步长
        delta4 : float
            四分之一步长
        """
        # 第一部分：来自后续热浴的耦合（如果存在）
        if j < self.tchain - 1:
            coupling_factor = np.exp(-delta4 * self.p_zeta[j + 1] / self.Q[j + 1])
            self.p_zeta[j] *= coupling_factor

        # 计算热浴"力" G_j
        if j == 0:
            # 第一个热浴：与物理系统耦合（向量化实现，口径一致）
            v = cell.get_velocities()  # (N,3)
            m = (
                self._masses
                if self._masses is not None
                else np.array([a.mass for a in cell.atoms], dtype=np.float64)
            )
            if self._num_atoms_global > 1:
                total_mass = float(np.sum(m))
                com_velocity = (m[:, None] * v).sum(axis=0) / total_mass
                dv = v - com_velocity[None, :]
                momentum_squared_over_mass = float(
                    np.sum(m * np.einsum("ij,ij->i", dv, dv))
                )
            else:
                momentum_squared_over_mass = float(
                    np.sum(m * np.einsum("ij,ij->i", v, v))
                )

            G_j = (
                momentum_squared_over_mass
                - self._dof * KB_IN_EV * self.target_temperature
            )
        else:
            # 后续热浴：与前一个热浴耦合
            G_j = (
                self.p_zeta[j - 1] ** 2 / self.Q[j - 1]
            ) - KB_IN_EV * self.target_temperature

        # 应用"力"（中心更新）
        self.p_zeta[j] += delta2 * G_j

        # 第二部分：再次来自后续热浴的耦合
        if j < self.tchain - 1:
            coupling_factor = np.exp(-delta4 * self.p_zeta[j + 1] / self.Q[j + 1])
            self.p_zeta[j] *= coupling_factor

    def _update_statistics(self, cell: Cell) -> None:
        """更新统计信息"""
        self._total_steps += 1

        # 记录温度
        current_temp = cell.calculate_temperature()
        self._temp_history.append(current_temp)

        # 记录守恒量（扩展哈密顿量）
        try:
            conserved_energy = self.get_conserved_energy(cell)
            self._conserved_energy_history.append(conserved_energy)
        except Exception:
            # 如果计算失败，记录NaN
            self._conserved_energy_history.append(float("nan"))

        # 限制历史长度
        max_history = 1000
        if len(self._temp_history) > max_history:
            self._temp_history.pop(0)
            self._conserved_energy_history.pop(0)

    def get_conserved_energy(self, cell: Cell) -> float:
        r"""计算 NHC 扩展哈密顿量（守恒量）

        .. math::
            H' = E_{\mathrm{kin}} + E_{\mathrm{pot}} + E_{\mathrm{thermo}}

        其中热浴能量 :math:`E_{\mathrm{thermo}}` 为：

        .. math::
            E_{\mathrm{thermo}} = \sum_{j=0}^{M-1} \frac{p_{\zeta_j}^2}{2 Q_j}
            + N_f k_B T_0\, \zeta_0
            + k_B T_0 \sum_{j=1}^{M-1} \zeta_j

        Returns
        -------
        float
            扩展哈密顿量 (eV)
        """
        if not self._initialized:
            return float("nan")

        # 动能和势能
        kinetic = self._calculate_instantaneous_kinetic_energy(cell)
        potential = cell.calculate_potential_energy()

        # 热浴动能
        thermostat_kinetic = np.sum(0.5 * self.p_zeta**2 / self.Q)

        # 热浴势能
        kB_T = KB_IN_EV * self.target_temperature
        thermostat_potential = self._dof * kB_T * self.zeta[0] + kB_T * np.sum(
            self.zeta[1:]
        )

        return kinetic + potential + thermostat_kinetic + thermostat_potential

    def get_statistics(self) -> dict:
        """获取详细统计信息"""
        if self._total_steps == 0:
            return {
                "total_steps": 0,
                "target_temperature": self.target_temperature,
                "tdamp": self.tdamp,
                "tchain": self.tchain,
                "tloop": self.tloop,
                "average_temperature": 0.0,
                "temperature_std": 0.0,
                "current_p_zeta": self.p_zeta.tolist(),
                "current_zeta": self.zeta.tolist(),
                "Q_parameters": self.Q.tolist() if self.Q is not None else None,
                "conserved_energy_drift": 0.0,
            }

        # 温度统计
        temps = np.array(self._temp_history)
        avg_temp = np.mean(temps)
        temp_std = np.std(temps)

        # 守恒量漂移（最近500步的线性拟合斜率）
        conserved_drift = 0.0
        if len(self._conserved_energy_history) > 100:
            recent_energies = self._conserved_energy_history[-500:]
            if not any(np.isnan(recent_energies)):
                x = np.arange(len(recent_energies))
                slope, _ = np.polyfit(x, recent_energies, 1)
                conserved_drift = slope

        return {
            "total_steps": self._total_steps,
            "target_temperature": self.target_temperature,
            "tdamp": self.tdamp,
            "tchain": self.tchain,
            "tloop": self.tloop,
            "average_temperature": float(avg_temp),
            "temperature_std": float(temp_std),
            "current_p_zeta": self.p_zeta.tolist(),
            "current_zeta": self.zeta.tolist(),
            "Q_parameters": self.Q.tolist() if self.Q is not None else None,
            "conserved_energy_drift": float(conserved_drift),
        }

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self._total_steps = 0
        self._temp_history = []
        self._conserved_energy_history = []

    def reset_thermostat_state(self) -> None:
        """完全重置热浴状态"""
        self.p_zeta.fill(0.0)
        self.zeta.fill(0.0)
        self.reset_statistics()


# 为了方便导入，提供一个创建基础NVE传播子的工厂函数
def create_nve_propagators(potential):
    r"""创建 NVE 积分所需的基础传播子

    Parameters
    ----------
    potential : Potential
        势函数对象

    Returns
    -------
    dict
        传播子字典：

        ``'position'``
            :class:`~thermoelasticsim.md.propagators.PositionPropagator` 实例

        ``'velocity'``
            :class:`~thermoelasticsim.md.propagators.VelocityPropagator` 实例

        ``'force'``
            :class:`~thermoelasticsim.md.propagators.ForcePropagator` 实例

    Examples
    --------
    >>> propagators = create_nve_propagators(eam_potential)
    >>> pos_prop = propagators['position']
    >>> vel_prop = propagators['velocity']
    >>> force_prop = propagators['force']
    """
    return {
        "position": PositionPropagator(),
        "velocity": VelocityPropagator(),
        "force": ForcePropagator(potential),
    }


class LangevinThermostatPropagator(Propagator):
    r"""Langevin 动力学恒温器传播子（BBK 积分）

    通过阻尼力和随机力模拟热浴作用，实现温度控制。

    Notes
    -----
    Langevin 运动方程：

    .. math::
        m_i \frac{d^2\mathbf{r}_i}{dt^2} = \mathbf{F}_i(\mathbf{r}) - \gamma m_i \frac{d\mathbf{r}_i}{dt} + \mathbf{R}_i(t)

    其中：

    - :math:`\mathbf{F}_i(\mathbf{r})` - 保守力
    - :math:`\gamma` - 摩擦系数（friction coefficient）
    - :math:`\mathbf{R}_i(t)` - 随机力，满足涨落–耗散定理

    BBK 积分常数：

    - :math:`c_1 = \exp(-\gamma\,\Delta t)` - 速度阻尼因子
    - :math:`\sigma = \sqrt{k_B T\,(1-c_1^2)/m}` - 随机力标准差

    参考文献：
        Brünger, Brooks & Karplus, Chem. Phys. Lett. 105, 495 (1984).
    """

    def __init__(self, target_temperature: float, friction: float = 1.0):
        r"""初始化 Langevin 恒温器。

        Parameters
        ----------
        target_temperature : float
            目标温度 (K)，必须为正数。
        friction : float, optional
            摩擦系数 gamma (ps⁻¹)，默认值 1.0 ps⁻¹。

        Raises
        ------
        ValueError
            如果 target_temperature 或 friction 为非正数。
        """
        if target_temperature <= 0:
            raise ValueError(f"目标温度必须为正数，得到 {target_temperature} K")
        if friction <= 0:
            raise ValueError(f"摩擦系数必须为正数，得到 {friction} ps⁻¹")

        self.target_temperature = target_temperature
        self.friction = friction  # γ in ps⁻¹

        # 统计信息
        self._total_steps = 0
        self._temperature_history = []
        self._friction_work_history = []  # 摩擦做功历史
        self._random_work_history = []  # 随机力做功历史

        print(f"Langevin恒温器初始化: T={target_temperature}K, γ={friction:.3f} ps⁻¹")
        print(f"  阻尼时间: τ_damp = {1 / friction:.3f} ps")
        print(
            f"  预期特性: {'强耦合' if friction >= 5.0 else '弱耦合' if friction <= 0.5 else '中等耦合'}"
        )

    def propagate(self, cell: Cell, dt: float, **kwargs: Any) -> None:
        r"""执行 Langevin 恒温器传播（BBK 速度更新）。

        Parameters
        ----------
        cell : Cell
            晶胞对象，将修改其中所有原子的速度。
        dt : float
            时间步长 (fs)。
        **kwargs : Any
            额外参数，当前版本未使用。

        Raises
        ------
        ValueError
            如果 dt <= 0。
        RuntimeError
            如果温度计算或随机数生成失败。
        """
        if dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 dt={dt} fs")

        try:
            # 转换时间单位：dt从fs转换为ps（因为friction单位是ps⁻¹）
            dt_ps = dt / 1000.0  # fs -> ps

            # 计算当前温度用于监控
            current_temp = self._calculate_temperature(cell)

            # BBK算法常数
            gamma_dt = self.friction * dt_ps
            c1 = np.exp(-gamma_dt)  # 速度衰减因子

            # 向量化更新
            V = cell.get_velocities()  # (N,3)
            M = np.array([a.mass for a in cell.atoms], dtype=np.float64)

            # 记录摩擦与随机做功
            V_old = V.copy()

            # σ_i = sqrt(kB*T*(1-c1^2)/m_i)
            variance = KB_IN_EV * self.target_temperature * (1.0 - c1 * c1) / M
            variance = np.maximum(variance, 0.0)
            sigma = np.sqrt(variance)

            R = np.random.normal(0.0, 1.0, V.shape)
            V_new = c1 * V + sigma[:, None] * R

            # 做功统计（近似 m * v · dv）
            dv_friction = (c1 - 1.0) * V_old
            dv_random = sigma[:, None] * R
            total_friction_work = float(
                np.sum(M * np.einsum("ij,ij->i", V_old, dv_friction))
            )
            total_random_work = float(
                np.sum(M * np.einsum("ij,ij->i", V_old, dv_random))
            )

            # 回写速度
            for i, atom in enumerate(cell.atoms):
                atom.velocity = V_new[i]

            # 移除质心运动（保持动量守恒）
            # 注意：在Langevin动力学中，随机力可能会产生净动量
            cell.remove_com_motion()

            # 更新统计信息
            self._total_steps += 1
            self._temperature_history.append(current_temp)
            self._friction_work_history.append(total_friction_work)
            self._random_work_history.append(total_random_work)

            # 定期输出统计信息（每1000步）
            if self._total_steps % 1000 == 0:
                final_temp = self._calculate_temperature(cell)
                temp_error = (
                    abs(final_temp - self.target_temperature)
                    / self.target_temperature
                    * 100
                )
                avg_friction_work = np.mean(self._friction_work_history[-1000:])
                avg_random_work = np.mean(self._random_work_history[-1000:])

                print(
                    f"Langevin统计 (#{self._total_steps}步): T={final_temp:.1f}K, 误差={temp_error:.2f}%"
                )
                print(
                    f"  能量平衡: 摩擦做功={avg_friction_work:.3e} eV, 随机做功={avg_random_work:.3e} eV"
                )

        except Exception as e:
            raise RuntimeError(f"Langevin恒温器传播失败: {str(e)}") from e

    def _calculate_temperature(self, cell: Cell) -> float:
        """计算当前系统温度（与structure.py保持一致）

        Parameters
        ----------
        cell : Cell
            晶胞对象

        Returns
        -------
        float
            当前温度 (K)
        """
        num_atoms = len(cell.atoms)
        if num_atoms == 0:
            return 0.0

        if num_atoms == 1:
            # 单原子系统
            atom = cell.atoms[0]
            kinetic = 0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
            dof = 3
        else:
            # 多原子系统：扣除质心运动
            total_mass = sum(atom.mass for atom in cell.atoms)
            total_momentum = sum(atom.mass * atom.velocity for atom in cell.atoms)
            com_velocity = total_momentum / total_mass

            kinetic = sum(
                0.5
                * atom.mass
                * np.dot(atom.velocity - com_velocity, atom.velocity - com_velocity)
                for atom in cell.atoms
            )
            dof = 3 * num_atoms - 3

        if dof <= 0:
            return 0.0

        temperature = 2.0 * kinetic / (dof * KB_IN_EV)
        return temperature

    def get_statistics(self) -> dict:
        """获取 Langevin 恒温器统计信息

        Returns
        -------
        dict
            统计信息字典：

            total_steps
                总传播步数

            target_temperature
                目标温度 (K)

            friction
                摩擦系数 (ps⁻¹)

            damping_time
                阻尼时间 (ps)

            coupling_strength
                耦合强度描述

            average_temperature
                平均温度 (K)

            temperature_std
                温度标准差 (K)

            temperature_error
                平均温度误差 (%)

            average_friction_work
                平均摩擦做功 (eV)

            average_random_work
                平均随机做功 (eV)

            energy_balance
                摩擦做功与随机做功比值

            recent_temperatures
                最近 100 步的温度历史
        """
        if self._total_steps == 0:
            return {
                "total_steps": 0,
                "target_temperature": self.target_temperature,
                "friction": self.friction,
                "damping_time": 1.0 / self.friction,
                "coupling_strength": self._get_coupling_strength_description(),
                "average_temperature": 0.0,
                "temperature_std": 0.0,
                "temperature_error": 0.0,
                "average_friction_work": 0.0,
                "average_random_work": 0.0,
                "energy_balance": 0.0,
                "recent_temperatures": [],
            }

        temps = np.array(self._temperature_history)
        friction_works = np.array(self._friction_work_history)
        random_works = np.array(self._random_work_history)

        avg_temp = np.mean(temps)
        temp_std = np.std(temps)
        temp_error = (
            abs(avg_temp - self.target_temperature) / self.target_temperature * 100
        )

        avg_friction_work = np.mean(friction_works) if len(friction_works) > 0 else 0.0
        avg_random_work = np.mean(random_works) if len(random_works) > 0 else 0.0

        # 能量平衡：理想情况下摩擦做功与随机做功应该基本抵消
        energy_balance = (avg_friction_work + avg_random_work) / max(
            abs(avg_friction_work), 1e-10
        )

        return {
            "total_steps": self._total_steps,
            "target_temperature": self.target_temperature,
            "friction": self.friction,
            "damping_time": 1.0 / self.friction,
            "coupling_strength": self._get_coupling_strength_description(),
            "average_temperature": avg_temp,
            "temperature_std": temp_std,
            "temperature_error": temp_error,
            "average_friction_work": avg_friction_work,
            "average_random_work": avg_random_work,
            "energy_balance": energy_balance,
            "recent_temperatures": (
                temps[-100:].tolist() if len(temps) >= 100 else temps.tolist()
            ),
        }

    def _get_coupling_strength_description(self) -> str:
        """获取耦合强度的描述性字符串

        Returns
        -------
        str
            耦合强度描述
        """
        if self.friction >= 5.0:
            return "strong"  # 强耦合：快速温度控制，动力学扰动大
        elif self.friction <= 0.5:
            return "weak"  # 弱耦合：温度控制慢，动力学保持好
        else:
            return "moderate"  # 中等耦合：平衡控温速度和动力学保持

    def reset_statistics(self) -> None:
        """重置所有统计信息"""
        self._total_steps = 0
        self._temperature_history.clear()
        self._friction_work_history.clear()
        self._random_work_history.clear()

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
        这允许在模拟过程中调整恒温器的耦合强度，
        例如在平衡阶段使用大摩擦系数，在生产阶段使用小摩擦系数。
        """
        if new_friction <= 0:
            raise ValueError(f"摩擦系数必须为正数，得到 {new_friction} ps⁻¹")

        old_friction = self.friction
        self.friction = new_friction

        print(f"摩擦系数已更新: {old_friction:.3f} -> {new_friction:.3f} ps⁻¹")
        print(f"阻尼时间已更新: {1 / old_friction:.3f} -> {1 / new_friction:.3f} ps")

    def get_effective_parameters(self) -> dict:
        r"""获取当前有效参数

        Returns
        -------
        dict
            参数字典：

            friction
                摩擦系数 (ps⁻¹)

            damping_time
                阻尼时间 (ps)，:math:`\tau_{\mathrm{damp}} = 1/\gamma`

            target_temperature
                目标温度 (K)

            coupling_description
                耦合强度描述（weak/moderate/strong）
        """
        return {
            "friction": self.friction,
            "damping_time": 1.0 / self.friction,
            "target_temperature": self.target_temperature,
            "coupling_description": self._get_coupling_strength_description(),
        }


# ====================================================================
# MTK-NPT 恒压器算符 (Martyna-Tobias-Klein Barostat)
# ====================================================================


def exprel(x: np.ndarray) -> np.ndarray:
    """
    计算 (exp(x) - 1) / x，避免x接近0时的数值误差

    Parameters
    ----------
    x : np.ndarray
        输入数组

    Returns
    -------
    np.ndarray
        计算结果
    """
    # 对于小值使用Taylor展开：exprel(x) = 1 + x/2 + x²/6 + x³/24 + ...
    mask_small = np.abs(x) < 1e-6
    result = np.zeros_like(x)

    # 大值情况：直接计算
    mask_large = ~mask_small
    result[mask_large] = (np.exp(x[mask_large]) - 1) / x[mask_large]

    # 小值情况：Taylor展开到4阶
    x_small = x[mask_small]
    result[mask_small] = 1 + x_small / 2 + x_small**2 / 6 + x_small**3 / 24

    return result


class MTKBarostatPropagator(Propagator):
    r"""MTK（Martyna–Tobias–Klein）恒压器传播子

    Notes
    -----
    对称分解 + 矩阵指数实现：

    - 晶格共轭动量 :math:`\mathbf{P}_g` 的演化由应力与外压驱动
    - 在 :math:`\mathbf{P}_g` 的特征基中进行精确积分（矩阵指数）

    核心关系：

    .. math::
        d\mathbf{P}_g/dt = V\,(\boldsymbol{\sigma}_{\mathrm{int}} - P_{\mathrm{ext}}\,\mathbf{I}) + \text{kinetic correction}

    References
    ----------
    Martyna, G. J.; Tobias, D. J.; Klein, M. L. (1994). Constant pressure molecular dynamics algorithms.
    The Journal of Chemical Physics, 101(5), 4177–4189. https://doi.org/10.1063/1.467468

    Parameters
    ----------
    target_temperature : float
        目标温度 (K)
    target_pressure : float
        目标压力 (eV/Å³)
    pdamp : float
        压力阻尼时间 (fs)，典型值约 :math:`100\,\times\,dt`
    pchain : int, optional
        恒压器链长度，默认 3
    ploop : int, optional
        恒压器链积分循环次数，默认 1
    """

    def __init__(
        self,
        target_temperature: float,
        target_pressure: float,
        pdamp: float,
        pchain: int = 3,
        ploop: int = 1,
    ):
        super().__init__()

        # 基本参数
        self.target_temperature = target_temperature
        self.target_pressure = target_pressure  # eV/Å³
        self.pdamp = pdamp  # fs
        self.pchain = pchain
        self.ploop = ploop

        # 热力学常数
        self.kT = KB_IN_EV * target_temperature  # eV

        # 应力计算器
        self.stress_calculator = StressCalculator()

        # 初始化链变量
        self._xi = np.zeros(self.pchain)  # 恒压器坐标
        self._p_xi = np.zeros(self.pchain)  # 恒压器动量

        # 晶格共轭动量 (3x3矩阵)
        self._P_g = np.zeros((3, 3))

        # 初始化标记
        self._initialized = False

        # 统计信息
        self._total_steps = 0
        self._pressure_history = []
        self._volume_history = []
        self._barostat_energy_history = []

        print("MTK恒压器初始化:")
        print(f"  目标温度: {target_temperature:.1f} K")
        print(f"  目标压力: {target_pressure:.6f} eV/Å³")
        print(f"  压力阻尼: {pdamp:.1f} fs")
        print(f"  恒压器链: {pchain} 变量")

    def _initialize_parameters(self, cell: Cell) -> None:
        """
        初始化质量参数和内部变量

        Parameters
        ----------
        cell : Cell
            晶胞对象
        """
        if self._initialized:
            return

        N_atoms = len(cell.atoms)

        # 晶格质量参数 W (关键参数)
        # 经典选择：W = (N + 1) * kB * T * τ_p²
        self.W = (N_atoms + 1) * self.kT * (self.pdamp**2)

        # 恒压器链质量参数 R_j
        self._R = np.zeros(self.pchain)
        cell_dof = 9  # 3x3晶格矩阵的自由度

        # R[0]对应晶格自由度，R[j>0]对应链变量
        self._R[0] = cell_dof * self.kT * (self.pdamp**2)
        for j in range(1, self.pchain):
            self._R[j] = self.kT * (self.pdamp**2)

        self._initialized = True

        print("MTK参数初始化完成:")
        print(f"  晶格质量 W: {self.W:.2e}")
        print(f"  链质量 R[0]: {self._R[0]:.2e}")
        print(f"  链质量 R[1:]: {self._R[1]:.2e}")

    def propagate(self, cell: Cell, dt: float, potential=None) -> None:
        """
        执行恒压器传播一个时间步dt

        基于对称Trotter分解的恒压器链积分，包含：
        1. 恒压器链传播 (dt/2)
        2. 晶格动量更新
        3. 恒压器链传播 (dt/2)

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长 (fs)
        potential : Potential, optional
            势能对象，用于应力计算
        """
        if not self._initialized:
            self._initialize_parameters(cell)

        if potential is None:
            print("警告: MTK恒压器需要势能对象计算应力")
            return

        # 对称分解：前半步恒压器链传播
        dt_half = dt / 2.0
        self._integrate_barostat_chain(cell, dt_half)

        # 更新晶格动量（受应力驱动）
        self._update_cell_momenta(cell, potential, dt_half)

        # 对称分解：后半步恒压器链传播
        self._integrate_barostat_chain(cell, dt_half)

        # 更新统计
        self._update_statistics(cell, potential)
        self._total_steps += 1

    def _integrate_barostat_chain(self, cell: Cell, dt: float) -> None:
        """
        积分恒压器Nose-Hoover链

        实现与NHC恒温器类似的链式积分，但"力"来源于晶格动能。
        使用四阶Suzuki-Yoshida分解确保高精度。

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长
        """
        # 确保参数已初始化
        if not self._initialized:
            self._initialize_parameters(cell)

        # 四阶Suzuki-Yoshida系数
        dt_suzuki = [w * dt for w in FOURTH_ORDER_COEFFS] * self.ploop

        for dt_sub in dt_suzuki:
            self._single_barostat_chain_step(cell, dt_sub)

    def _single_barostat_chain_step(self, cell: Cell, dt: float) -> None:
        """
        执行单步恒压器链积分

        算法流程（对称回文）:
        1. 更新链尾变量p_xi[M-1] (dt/2)
        2. 依次更新p_xi[j], xi[j] (j从M-1到0)
        3. 缩放晶格动量P_g
        4. 反向更新xi[j], p_xi[j] (j从0到M-1)
        5. 更新链尾变量p_xi[M-1] (dt/2)

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长
        """
        dt_half = dt / 2.0
        dt_quarter = dt / 4.0

        # 1. 更新链尾 (M-1) 动量
        if self.pchain > 1:
            G_M_minus_1 = (self._p_xi[-2] ** 2 / self._R[-2] - self.kT) / self._R[-1]
            self._p_xi[-1] += G_M_minus_1 * dt_quarter

        # 2. 前向更新链 (从M-1到1)
        for j in range(self.pchain - 1, 0, -1):
            # 计算"力" G_j
            if j == self.pchain - 1:
                if j == 1:
                    # 第一个链变量的力来自晶格动能
                    trace_P2 = np.trace(self._P_g.T @ self._P_g)
                    G_j = (trace_P2 / self.W - 9 * self.kT) / self._R[j]
                else:
                    G_j = (self._p_xi[j - 1] ** 2 / self._R[j - 1] - self.kT) / self._R[
                        j
                    ]
            else:
                if j == 1:
                    trace_P2 = np.trace(self._P_g.T @ self._P_g)
                    G_j = (trace_P2 / self.W - 9 * self.kT) / self._R[j]
                else:
                    G_j = (self._p_xi[j - 1] ** 2 / self._R[j - 1] - self.kT) / self._R[
                        j
                    ]

            # 更新动量和坐标
            self._p_xi[j] += G_j * dt_half
            self._xi[j] += (self._p_xi[j] / self._R[j]) * dt_half

        # 3. 更新第一个链变量 (j=0)
        if self.pchain > 0:
            # 第0个变量的"力"来自晶格动能
            trace_P2 = np.trace(self._P_g.T @ self._P_g)
            G_0 = (trace_P2 / self.W - 9 * self.kT) / self._R[0]

            self._p_xi[0] += G_0 * dt_half
            self._xi[0] += (self._p_xi[0] / self._R[0]) * dt_half

            # 缩放晶格动量
            scale_factor = np.exp(-self._p_xi[0] / self._R[0] * dt)
            self._P_g *= scale_factor

        # 4. 反向更新链 (从0到M-1) - 时间可逆性
        for j in range(self.pchain):
            if j == 0:
                trace_P2 = np.trace(self._P_g.T @ self._P_g)
                G_0 = (trace_P2 / self.W - 9 * self.kT) / self._R[0]
                self._p_xi[0] += G_0 * dt_half
            else:
                if j == 1:
                    trace_P2 = np.trace(self._P_g.T @ self._P_g)
                    G_j = (trace_P2 / self.W - 9 * self.kT) / self._R[j]
                else:
                    G_j = (self._p_xi[j - 1] ** 2 / self._R[j - 1] - self.kT) / self._R[
                        j
                    ]

                self._p_xi[j] += G_j * dt_half

            self._xi[j] += (self._p_xi[j] / self._R[j]) * dt_half

        # 5. 最终更新链尾动量
        if self.pchain > 1:
            G_M_minus_1 = (self._p_xi[-2] ** 2 / self._R[-2] - self.kT) / self._R[-1]
            self._p_xi[-1] += G_M_minus_1 * dt_quarter

    def _update_cell_momenta(self, cell: Cell, potential, dt: float) -> None:
        """
        更新晶格共轭动量P_g

        基于内外应力差驱动晶格动量变化:
        dP_g/dt = V * (σ_internal - P_external * I) + kinetic_correction

        Parameters
        ----------
        cell : Cell
            晶胞对象
        potential : Potential
            势能对象
        dt : float
            时间步长
        """
        # 确保参数已初始化
        if not self._initialized:
            self._initialize_parameters(cell)

        # 计算应力张量 (eV/Å³)
        stress_tensor = self.stress_calculator.calculate_total_stress(cell, potential)

        # 计算体积和动能修正项
        volume = cell.volume
        N_atoms = len(cell.atoms)

        # 动能修正项：Tr(p²/m) / (3*N) * I
        total_kinetic = 0.0
        for atom in cell.atoms:
            velocity = atom.velocity  # Å/fs
            mass = atom.mass  # 原子质量单位
            total_kinetic += 0.5 * mass * np.dot(velocity, velocity)

        # 转换动能到正确单位并计算修正项
        kinetic_correction = (2.0 * total_kinetic) / (3.0 * N_atoms) * np.eye(3)

        # 应力驱动项: 使用 "pressure-like" 张量 -σ
        # G = V * ( -σ - P_ext * I ) + kinetic_correction
        pressure_tensor_ext = self.target_pressure * np.eye(3)
        internal_pressure_tensor = -stress_tensor
        stress_drive = volume * (internal_pressure_tensor - pressure_tensor_ext)

        # 总驱动力
        G = stress_drive + kinetic_correction

        # 更新晶格动量
        self._P_g += G * dt

    def integrate_momenta(self, cell: Cell, potential, dt: float) -> None:
        r"""
        更新粒子动量（与恒压器耦合的半步，MTK 公式）

        在 :math:`\mathbf{P}_g` 的特征基下：

        .. math::
            \mathbf{p}' = \mathbf{p}\,\exp\!\Big(-\frac{\boldsymbol{\kappa}\,\Delta t}{W}\Big)
            + \Delta t\,\mathbf{F}\,\operatorname{exprel}\!\Big(-\frac{\boldsymbol{\kappa}\,\Delta t}{W}\Big)

        其中 :math:`\boldsymbol{\kappa} = \boldsymbol{\lambda} + \operatorname{Tr}(\mathbf{P}_g)/(3N)`。

        Notes
        -----
        - 此方法已包含力贡献，无需额外的 NVE 速度半步
        - 调用前后需配合 ``_update_cell_momenta(dt/2)`` 与位置/晶格传播
        """
        if not self._initialized:
            self._initialize_parameters(cell)

        # 确保力为最新
        potential.calculate_forces(cell)

        # 特征分解 P_g = U diag(lambda) U^T
        try:
            eigvals, U = np.linalg.eigh(self._P_g)
        except np.linalg.LinAlgError:
            eigvals = np.zeros(3)
            U = np.eye(3)

        N = len(cell.atoms)
        if N == 0:
            return

        masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)  # (N,)
        velocities = cell.get_velocities()  # (N,3)
        momenta = velocities * masses[:, None]  # p = m v
        forces = cell.get_forces()  # (N,3)

        # 变换到特征坐标
        y = momenta @ U  # (N,3)
        fU = forces @ U  # (N,3)

        # 计算kappa和缩放系数
        kappa = eigvals + np.trace(self._P_g) / (3.0 * N)  # (3,)
        x = -kappa * (dt / self.W)  # (3,)
        scale = np.exp(x)  # (3,)
        exr = exprel(x)  # (3,)

        y_new = y * scale[None, :] + dt * fU * exr[None, :]
        p_new = y_new @ U.T

        # 回写速度
        v_new = p_new / masses[:, None]
        for i, atom in enumerate(cell.atoms):
            atom.velocity = v_new[i]

    def propagate_positions_and_cell(self, cell: Cell, dt: float) -> None:
        r"""
        使用矩阵指数精确传播粒子位置与晶格（MTK 核心步骤）

        流程：

        1. 特征分解 :math:`\mathbf{P}_g = \mathbf{U}\,\operatorname{diag}(\boldsymbol{\lambda})\,\mathbf{U}^\top`
        2. 在特征坐标系中用 :math:`\exp(\boldsymbol{\lambda}\,\Delta t/W)` 精确积分
        3. 变换回原坐标系

        Parameters
        ----------
        cell : Cell
            晶胞对象
        dt : float
            时间步长 (fs)
        """
        if not self._initialized:
            self._initialize_parameters(cell)

        # 对P_g进行特征值分解
        try:
            eigvals, U = np.linalg.eigh(self._P_g)
        except np.linalg.LinAlgError:
            print("警告: P_g矩阵特征值分解失败，使用单位矩阵")
            eigvals = np.zeros(3)
            U = np.eye(3)

        # 1. 更新晶格矩阵 h
        h_matrix = cell.lattice_vectors.T  # 转换为列向量形式
        h_transformed = h_matrix @ U  # 变换到特征坐标系

        # 在特征坐标系中精确积分
        exp_factors = np.exp(eigvals * dt / self.W)
        h_evolved = h_transformed * exp_factors[None, :]  # 广播

        # 变换回原坐标系
        h_new = h_evolved @ U.T
        # 使用接口以确保体积和逆矩阵更新
        cell.set_lattice_vectors(h_new.T)  # 转换回行向量形式

        # 2. 更新原子位置与速度（仅恒压器耦合的部分）
        positions = cell.get_positions()  # (N, 3)
        velocities = cell.get_velocities()  # (N, 3)

        # 在特征坐标系中变换
        pos_transformed = positions @ U  # (N, 3) @ (3, 3) = (N, 3)
        vel_transformed = velocities @ U

        # 精确积分位置 - 使用exprel避免数值误差
        dt_over_W = dt / self.W
        lambda_dt = eigvals * dt_over_W  # (3,)

        # ✅ 修复：正确的MTK位置更新公式
        # ASE公式: r_new = r * exp(λt) + v * dt * exprel(λt)
        # 其中exprel项已经包含dt因子，不需要额外乘dt
        exp_factors = np.exp(lambda_dt)
        exprel_factors = exprel(lambda_dt) * dt  # dt因子在这里

        pos_evolved = (
            pos_transformed * exp_factors[None, :]
            + vel_transformed * exprel_factors[None, :]
        )  # 移除多余的dt

        # 变换回原坐标系
        positions_new = pos_evolved @ U.T
        cell.set_positions(positions_new)

        # 注意：此处不更新原子速度。动量的指数缩放与力贡献由
        # integrate_momenta() 两个半步负责，避免重复缩放。

    def _update_statistics(self, cell: Cell, potential) -> None:
        """更新统计信息"""
        # 计算当前压力（取迹的1/3）
        stress_tensor = self.stress_calculator.calculate_total_stress(cell, potential)
        current_pressure = -np.trace(stress_tensor) / 3.0  # 负号：拉伸为正压

        # 记录统计量
        self._pressure_history.append(current_pressure)
        self._volume_history.append(cell.volume)

        # 计算恒压器能量
        barostat_energy = self.get_barostat_energy()
        self._barostat_energy_history.append(barostat_energy)

        # 保持历史长度
        max_history = 10000
        if len(self._pressure_history) > max_history:
            self._pressure_history = self._pressure_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
            self._barostat_energy_history = self._barostat_energy_history[-max_history:]

    def get_barostat_energy(self) -> float:
        r"""
        计算恒压器贡献的能量

        .. math::
            E_{\mathrm{baro}} = \sum_j \frac{p_{\xi_j}^2}{2 R_j}
            + \frac{\operatorname{Tr}(\mathbf{P}_g^{\top}\mathbf{P}_g)}{2W}
            + \text{external work}

        Returns
        -------
        float
            恒压器能量 (eV)
        """
        if not self._initialized:
            return 0.0

        # 恒压器链动能
        chain_kinetic = 0.0
        for j in range(self.pchain):
            chain_kinetic += 0.5 * self._p_xi[j] ** 2 / self._R[j]

        # 晶格动能
        lattice_kinetic = 0.5 * np.trace(self._P_g.T @ self._P_g) / self.W

        # 外压做功项 (暂时忽略，因为需要积分P*dV)
        external_work = 0.0

        return chain_kinetic + lattice_kinetic + external_work

    def get_current_pressure(self, cell: Cell, potential) -> float:
        """
        获取当前系统压力

        Returns
        -------
        float
            当前压力 (eV/Å³)
        """
        if not self._initialized:
            return 0.0

        stress_tensor = self.stress_calculator.calculate_total_stress(cell, potential)
        return -np.trace(stress_tensor) / 3.0

    def get_statistics(self) -> dict:
        """获取恒压器统计信息"""
        if self._total_steps == 0 or len(self._pressure_history) == 0:
            return {
                "total_steps": self._total_steps,
                "target_pressure": self.target_pressure,
                "target_pressure_GPa": self.target_pressure * EV_TO_GPA,
                "average_pressure": 0.0,
                "pressure_error": 0.0,
                "average_volume": 0.0,
                "volume_std": 0.0,
                "average_barostat_energy": 0.0,
            }

        pressures = np.array(self._pressure_history)
        volumes = np.array(self._volume_history)
        barostat_energies = np.array(self._barostat_energy_history)

        avg_pressure = np.mean(pressures)
        pressure_error = abs(avg_pressure - self.target_pressure)
        avg_volume = np.mean(volumes)
        volume_std = np.std(volumes)
        avg_barostat_energy = np.mean(barostat_energies)

        return {
            "total_steps": self._total_steps,
            "target_pressure": self.target_pressure,
            "target_pressure_GPa": self.target_pressure * EV_TO_GPA,
            "average_pressure": avg_pressure,
            "average_pressure_GPa": avg_pressure * EV_TO_GPA,
            "pressure_error": pressure_error,
            "pressure_error_GPa": pressure_error * EV_TO_GPA,
            "average_volume": avg_volume,
            "volume_std": volume_std,
            "average_barostat_energy": avg_barostat_energy,
            "recent_pressures": (
                pressures[-100:].tolist()
                if len(pressures) >= 100
                else pressures.tolist()
            ),
            "recent_volumes": (
                volumes[-100:].tolist() if len(volumes) >= 100 else volumes.tolist()
            ),
        }
