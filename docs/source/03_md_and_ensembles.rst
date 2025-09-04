.. currentmodule:: thermoelasticsim.md

========================
分子动力学与统计系综
========================

本章介绍分子动力学模拟的理论基础、统计系综概念以及ThermoElasticSim中的具体实现。重点在于如何使用schemes架构实现不同系综的模拟，解决"如何控制温度和压力"的核心问题。

MD理论基础
==============

哈密顿力学框架
--------------

N粒子系统的哈密顿量：

.. math::
   H(\mathbf{r}, \mathbf{p}) = \sum_{i=1}^N \frac{\mathbf{p}_i^2}{2m_i} + V(\mathbf{r}_1, ..., \mathbf{r}_N)

运动方程（哈密顿正则方程）：

.. math::
   \dot{\mathbf{r}}_i = \frac{\partial H}{\partial \mathbf{p}_i} = \frac{\mathbf{p}_i}{m_i}, \quad
   \dot{\mathbf{p}}_i = -\frac{\partial H}{\partial \mathbf{r}_i} = \mathbf{F}_i

刘维尔算符与时间演化
--------------------

系统的时间演化由刘维尔算符描述：

.. math::
   iL = \sum_{i=1}^N \left( \dot{\mathbf{r}}_i \cdot \frac{\partial}{\partial \mathbf{r}_i} + \dot{\mathbf{p}}_i \cdot \frac{\partial}{\partial \mathbf{p}_i} \right)

形式解为：

.. math::
   \Gamma(t) = e^{iLt} \Gamma(0)

Trotter分解与算符分离
---------------------

将刘维尔算符分解为可单独求解的部分：

.. math::
   iL = iL_r + iL_v + iL_F

其中：
    - :math:`iL_r = \sum_i \mathbf{v}_i \cdot \nabla_{\mathbf{r}_i}`：位置演化
    - :math:`iL_v = \sum_i \mathbf{F}_i/m_i \cdot \nabla_{\mathbf{v}_i}`：速度演化
    - :math:`iL_F`：力更新（瞬时）

Velocity-Verlet算法对应的对称分解：

.. math::
   e^{iL\Delta t} \approx e^{iL_v\Delta t/2} e^{iL_r\Delta t} e^{iL_F} e^{iL_v\Delta t/2}

关键API接口
================

ThermoElasticSim 的 MD 模拟基于统一的 schemes 架构。以下是核心接口：

**系综积分方案 (Integration Schemes)**
  - :class:`~thermoelasticsim.md.schemes.NVEScheme` - 微正则系综（NVE）
  - :class:`~thermoelasticsim.md.schemes.LangevinNVTScheme` - Langevin 恒温器（NVT）
  - :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme` - Nosé–Hoover 链（NVT）
  - :class:`~thermoelasticsim.md.schemes.MTKNPTScheme` - MTK 等温等压系综（NPT）

**传播子组件 (Propagators)**
  - :class:`~thermoelasticsim.md.propagators.PositionPropagator` - 位置更新算符
  - :class:`~thermoelasticsim.md.propagators.VelocityPropagator` - 速度更新算符
  - :class:`~thermoelasticsim.md.propagators.ForcePropagator` - 力计算算符

**系统状态计算**
  - :meth:`~thermoelasticsim.core.structure.Cell.calculate_stress_tensor` - 应力张量计算
  - :data:`~thermoelasticsim.utils.utils.EV_TO_GPA` - 单位转换常数（160.2176634）

统计系综概念与对照
==========================

系综定义与物理意义
------------------

**微正则系综（NVE）**
  - 守恒量：粒子数N、体积V、总能量E
  - 物理系统：孤立系统
  - 温度定义：通过动能的时间平均

**正则系综（NVT）**
  - 固定量：粒子数N、体积V、温度T
  - 物理系统：与热浴接触
  - 实现方式：恒温器调控动能

**等温等压系综（NPT）**
  - 固定量：粒子数N、压强P、温度T
  - 物理系统：与热浴和压浴接触
  - 晶胞体积可变

瞬时温度和压强
--------------

瞬时温度（基于能均分定理）：

.. math::
   T(t) = \frac{2K(t)}{3Nk_B} = \frac{1}{3Nk_B} \sum_{i=1}^N m_i v_i^2

瞬时压强（virial定理）：

.. math::
   P(t) = \frac{1}{V} \left( Nk_BT - \frac{1}{3}\sum_{i<j} \mathbf{r}_{ij} \cdot \mathbf{F}_{ij} \right)

恒温器实现
==============

Berendsen 恒温器（教学）
------------------------------

速度重缩放方法，简单但不严格满足正则分布：

.. math::
   \lambda = \sqrt{1 + \frac{\Delta t}{\tau_T}\left(\frac{T_0}{T} - 1\right)}

实现：:class:`~thermoelasticsim.md.schemes.BerendsenNVTScheme`

说明：仅作教学用途，生产计算请使用 :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme` 或 :class:`~thermoelasticsim.md.schemes.LangevinNVTScheme`。

Andersen 恒温器（教学）
----------------------------

随机碰撞方法，严格正则但可能影响动力学：

实现：:class:`~thermoelasticsim.md.schemes.AndersenNVTScheme`

说明：仅作教学用途，示例与推荐实践请参考 :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme` 与 :class:`~thermoelasticsim.md.schemes.LangevinNVTScheme`。

Nose-Hoover链恒温器
-------------------

扩展系统方法，保持确定性动力学：

链变量的运动方程：

.. math::
   \dot{\xi}_1 = \frac{G_1}{Q_1}, \quad G_1 = \sum_i m_i v_i^2 - 3Nk_BT

实现：:class:`~thermoelasticsim.md.propagators.NoseHooverChainPropagator` 和 :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme`

.. code-block:: python

    from thermoelasticsim.md.schemes import NoseHooverNVTScheme

    scheme = NoseHooverNVTScheme(
        target_temperature=300.0,
        chain_length=3,      # 链长度
        tau=50.0,           # 特征时间(fs)
        integration_order=4  # Suzuki-Yoshida阶数
    )

Langevin恒温器
--------------

随机-摩擦动力学：

.. math::
   m\ddot{\mathbf{r}} = \mathbf{F} - \gamma m\dot{\mathbf{r}} + \mathbf{R}(t)

其中随机力满足涨落-耗散定理：

.. math::
   \langle R_i(t)R_j(t') \rangle = 2m\gamma k_BT\delta_{ij}\delta(t-t')

实现：:class:`~thermoelasticsim.md.schemes.LangevinNVTScheme`

.. code-block:: python

    from thermoelasticsim.md.schemes import LangevinNVTScheme

    scheme = LangevinNVTScheme(
        target_temperature=300.0,
        friction=0.01  # ps⁻¹（注意单位）
    )

恒压器实现
==============

MTK (Martyna-Tobias-Klein) 算法
--------------------------------

扩展系统包含晶胞自由度，实现NPT系综：

晶胞演化方程：

.. math::
   \dot{\mathbf{h}} = \frac{p_\mathbf{h}}{W} \mathbf{h}

其中 :math:`W` 是晶胞"质量"参数。

实现：:class:`~thermoelasticsim.md.propagators.MTKBarostatPropagator` 和 :class:`~thermoelasticsim.md.schemes.MTKNPTScheme`

.. code-block:: python

    # 自包含最小示例：构建系统 + 运行 NPT
    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import MTKNPTScheme

    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )
    potential = EAMAl1Potential()

    scheme = MTKNPTScheme(
        target_temperature=300.0,  # K
        target_pressure=0.0,       # GPa
        tdamp=100.0,              # fs，温度阻尼时间
        pdamp=1000.0,             # fs，压力阻尼时间
        tchain=3
    )

    # NPT模拟
    for step in range(10000):
        scheme.step(cell, potential, dt=1.0)
        if step % 100 == 0:
            from thermoelasticsim.utils.utils import EV_TO_GPA
            T = cell.calculate_temperature()
            stress = cell.calculate_stress_tensor(potential)
            P = np.trace(stress) / 3.0 * EV_TO_GPA
            V = cell.volume
            print(f"Step {step}: T={T:.1f}K, P={P:.3f}GPa, V={V:.1f}Å³")

压力控制参数选择
----------------

典型参数建议：

- **tau_p**：1000-5000 fs（压力弛豫时间）
- **bulk_modulus**：用于估算晶胞质量
- **各向异性**：可选择各向同性或完全各向异性

可复现实例
================

Propagator-Scheme模式
----------------------

ThermoElasticSim采用模块化的Propagator-Scheme架构：

**Propagator（传播子）**：单一物理过程的时间演化
  - :class:`~thermoelasticsim.md.propagators.PositionPropagator`
  - :class:`~thermoelasticsim.md.propagators.VelocityPropagator`
  - :class:`~thermoelasticsim.md.propagators.ForcePropagator`
  - :class:`~thermoelasticsim.md.propagators.NoseHooverChainPropagator`

**Scheme（积分方案）**：组合Propagator实现完整算法
  - :class:`~thermoelasticsim.md.schemes.NVEScheme`
  - :class:`~thermoelasticsim.md.schemes.BerendsenNVTScheme`
  - :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme`
  - :class:`~thermoelasticsim.md.schemes.MTKNPTScheme`

NVE 微正则系综示例
===================

以下是 NVE 系综的基本使用方法，对 3×3×3 铝超胞进行 200 步积分：

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import NVEScheme

    # 创建系统
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )
    potential = EAMAl1Potential()

    # 初始化速度（300K Maxwell 分布）
    for atom in cell.atoms:
        # Maxwell分布标准差: σ = sqrt(kB*T/m)
        sigma = np.sqrt(8.617e-5 * 300.0 / atom.mass)  # eV单位
        atom.velocity = np.random.normal(0, sigma, 3)

    # 移除质心运动
    cell.remove_com_motion()

    # NVE 模拟
    scheme = NVEScheme()
    temperatures = []
    energies = []

    for step in range(200):
        scheme.step(cell, potential, dt=1.0)

        # 记录状态
        T_current = cell.calculate_temperature()
        E_kinetic = cell.calculate_kinetic_energy()
        E_potential = potential.calculate_energy(cell)

        temperatures.append(T_current)
        energies.append(E_kinetic + E_potential)

    # 统计结果
    T_mean = np.mean(temperatures[50:])  # 跳过前50步平衡
    E_drift = (energies[-1] - energies[0]) / energies[0]
    print(f"平均温度: {T_mean:.1f} K")
    print(f"能量漂移: {E_drift:.2e}")

NVT 恒温系综示例
===================

Langevin 恒温器的温度控制演示：

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import LangevinNVTScheme

    # 创建新系统（避免父本状态影响）
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )
    potential = EAMAl1Potential()

    # 初始化为 400K（高于目标）
    for atom in cell.atoms:
        sigma = np.sqrt(8.617e-5 * 400.0 / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
    cell.remove_com_motion()

    # Langevin NVT 模拟
    scheme = LangevinNVTScheme(
        target_temperature=300.0,  # K
        friction=1.0               # ps⁻¹
    )

    temperatures = []
    for step in range(1000):
        scheme.step(cell, potential, dt=1.0)

        if step % 10 == 0:  # 每10步记录
            T_current = cell.calculate_temperature()
            temperatures.append(T_current)

    # 温度收敛分析
    T_final = np.mean(temperatures[-20:])  # 最后20个点平均
    print(f"初始温度: 400K")
    print(f"目标温度: 300K")
    print(f"最终温度: {T_final:.1f}K")
    print(f"温度偏差: {abs(T_final - 300)/300*100:.1f}%")

Andersen 恒温器（教学最小示例）
--------------------------------

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import AndersenNVTScheme

    # 系统构建（3×3×3）
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )
    potential = EAMAl1Potential()

    # 初始化速度为 350K
    for atom in cell.atoms:
        sigma = np.sqrt(8.617e-5 * 350.0 / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
    cell.remove_com_motion()

    # 教学用随机碰撞恒温器（小系统，适度频率）
    scheme = AndersenNVTScheme(target_temperature=300.0, collision_frequency=0.01)

    # 短程演示
    for step in range(500):
        scheme.step(cell, potential, dt=1.0)
    print(f"T≈{cell.calculate_temperature():.1f} K (Andersen 教学示例)")

说明：Andersen 会破坏动力学连续性，适合教学或快速热化；生产计算建议使用 Langevin 或 Nose–Hoover 链。

NPT 等温等压系综示例
==========================

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import NVEScheme

    # 创建系统
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc('Al', 4.05, (3, 3, 3))
    potential = EAMAl1Potential()

.. raw:: html

    <!-- 重复示例移除：避免赘述与潜在冲突，保留上面的 Langevin 示例即可 -->

NPT 等温等压系综示例
====================

MTK 算法的压力和体积控制演示：

.. code-block:: python

    from thermoelasticsim.md.schemes import MTKNPTScheme
    from thermoelasticsim.utils.utils import EV_TO_GPA

    # 创建新系统
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )

    # 初始化300K速度
    for atom in cell.atoms:
        sigma = np.sqrt(8.617e-5 * 300.0 / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
    cell.remove_com_motion()

    # MTK NPT 模拟
    scheme = MTKNPTScheme(
        target_temperature=300.0,  # K
        target_pressure=0.0,       # GPa
        tau_t=100.0,              # 温度耦合时间 (fs)
        tau_p=1000.0,             # 压力耦合时间 (fs)
        chain_length=3            # Nose-Hoover链长度
    )

    # 平衡阶段（500步）
    for step in range(500):
        scheme.step(cell, potential, dt=1.0)

    # 生产阶段（记录数据）
    volumes = []
    pressures = []
    temperatures = []

    for step in range(1000):
        scheme.step(cell, potential, dt=1.0)

        if step % 10 == 0:
            # 计算瞬时状态
            stress_tensor = cell.calculate_stress_tensor(potential)
            pressure_gpa = np.trace(stress_tensor) / 3.0 * EV_TO_GPA

            volumes.append(cell.volume)
            pressures.append(pressure_gpa)
            temperatures.append(cell.calculate_temperature())

    # 统计分析
    V_mean = np.mean(volumes)
    V_std = np.std(volumes)
    P_mean = np.mean(pressures)
    P_std = np.std(pressures)
    T_mean = np.mean(temperatures)

    print(f"平均体积: {V_mean:.1f} ± {V_std:.1f} Å³")
    print(f"平均压力: {P_mean:.3f} ± {P_std:.3f} GPa")
    print(f"平均温度: {T_mean:.1f} K")

实现架构
============

Propagator-Scheme模式
---------------------

ThermoElasticSim采用模块化的Propagator-Scheme架构：

**Propagator（传播子）**：单一物理过程的时间演化
  - :class:`~propagators.PositionPropagator` - 位置更新算符
  - :class:`~propagators.VelocityPropagator` - 速度更新算符
  - :class:`~propagators.ForcePropagator` - 力计算算符
  - :class:`~propagators.NoseHooverChainPropagator` - Nose-Hoover链恒温器

**Scheme（积分方案）**：组合Propagator实现完整算法
  - :class:`~schemes.NVEScheme` - 微正则系综方案
  - :class:`~schemes.AndersenNVTScheme` - Andersen恒温系综方案
  - :class:`~schemes.LangevinNVTScheme` - Langevin恒温系综方案
  - :class:`~schemes.MTKNPTScheme` - MTK等温等压系综方案

最佳实践建议
============

系综选择指南
------------

- **结构优化**：使用NVE或简单的Berendsen NVT
- **平衡性质**：Nose-Hoover NVT/NPT提供严格系综
- **输运性质**：NVE或弱耦合NVT保持动力学
- **相变研究**：NPT允许体积变化

参数调节建议
------------

1. **时间步长稳定性**
   - 金属EAM势：1-2 fs（本章示例使用1 fs）
   - 共价键系统：0.5-1 fs
   - 验证标准：NVE能量守恒至机器精度

2. **恒温器耦合时间**
   - Andersen碰撞频率：小系统0.01 fs⁻¹，大系统0.05-0.08 fs⁻¹
   - Langevin摩擦系数：0.01 ps⁻¹（温度控制~2%精度；注意单位为ps⁻¹）
   - Nose-Hoover τ：50-100×dt

3. **恒压器参数**
   - MTK压力耦合时间：1000-5000×dt
   - 系统尺寸要求：>100原子保证统计意义

小结
====

本章完整展示了MD模拟的理论基础和ThermoElasticSim实现：

- **理论框架**：哈密顿力学、刘维尔算符、算符分离原理
- **系综概念**：NVE/NVT/NPT的控制量差异与适用场景
- **恒温恒压**：四种恒温器和MTK恒压器的算法原理
- **API实践**：基于schemes架构的可复现示例
- **参数指南**：时间步长、耦合参数的实用建议

所有示例均可直接运行，展示真实API用法，为后续弹性常数计算奠定基础。
