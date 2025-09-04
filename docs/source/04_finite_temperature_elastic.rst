.. currentmodule:: thermoelasticsim.elastic

========================
有限温度弹性常数
========================

本章系统化阐明有限温度弹性常数的计算方法，解决"如何在有限温度下准确测量弹性常数"的核心问题。采用"微小形变 + MD 采样应力"的等温刚度求解策略，结合统计误差分析。

计算工作流概览
==================

有限温度弹性常数计算的标准流程
------------------------------

与零温度方法的关键差异在于需要通过 MD 采样来获得时间平均应力响应：

**第一步：NPT 平衡获取热膨胀结构**
  在目标温度 T 和压力 P 下运行 NPT 系综，获得热膨胀后的平衡体积和晶格常数

**第二步：施加微小应变**
  对平衡结构施加小应变 ε，构造形变梯度矩阵 :math:`\mathbf{F} = \mathbf{I} + \boldsymbol{\varepsilon}`

**第三步：NVT 应力采样**
  在固定温度 T 下运行 NVT 系综，对形变系统进行 MD 采样

**第四步：时间平均应力**
  计算应力张量的时间平均值 :math:`\langle\boldsymbol{\sigma}\rangle_t`

**第五步：线性拟合弹性常数**
  从应力-应变关系 :math:`\langle\sigma_{ij}\rangle = C_{ijkl}(T) \varepsilon_{kl}` 拟合温度相关的弹性常数 :math:`C_{ijkl}(T)`

理论要点与近似
--------------

**小应变线性化假设**
  形变梯度矩阵采用 :math:`\mathbf{F} = \mathbf{I} + \boldsymbol{\varepsilon}`，忽略几何非线性项

**等温刚度定义**
  有限温度弹性常数为等温刚度：:math:`\sigma_{ij}(T) = C_{ijkl}(T) \varepsilon_{kl}`

**时间平均的物理意义**
  通过 MD 采样获得热力学平均，等效于系综平均：:math:`\langle\boldsymbol{\sigma}\rangle_{MD} \approx \langle\boldsymbol{\sigma}\rangle_{ensemble}`

关键API接口
===============

有限温度弹性常数计算涉及以下核心接口：

**晶体结构与材料**
  - :meth:`~thermoelasticsim.core.crystalline_structures.CrystallineStructureBuilder.create_fcc` - FCC晶体构建方法
  - :data:`~thermoelasticsim.elastic.materials.ALUMINUM_FCC` - 铝材料参数常数

**形变控制**
  - :class:`~thermoelasticsim.elastic.deformation.Deformer` - 应变施加器
  - :meth:`~thermoelasticsim.elastic.deformation.Deformer.apply_deformation` - 形变矩阵应用

**MD系综模拟**
  - :class:`~thermoelasticsim.md.schemes.MTKNPTScheme` - 等温等压系综
  - :class:`~thermoelasticsim.md.schemes.NoseHooverNVTScheme` - Nosé–Hoover 链（NVT）
  - :class:`~thermoelasticsim.md.schemes.LangevinNVTScheme` - Langevin 恒温系综

**系统状态计算**
  - :meth:`~thermoelasticsim.core.structure.Cell.calculate_stress_tensor` - 应力张量计算
  - :meth:`~thermoelasticsim.core.structure.Cell.calculate_temperature` - 瞬时温度计算
  - :data:`~thermoelasticsim.utils.utils.EV_TO_GPA` - 单位转换常数（160.2176634）

NPT平衡 + NVT采样示例
==========================

C44剪切弹性常数计算
-------------------

以下演示在300K下计算铝的C44弹性常数，展示完整的NPT平衡+NVT采样工作流：

.. code-block:: python

    import numpy as np
    # 线性拟合可用 numpy.polyfit，避免SciPy依赖
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.md.schemes import create_mtk_npt_scheme, NoseHooverNVTScheme
    from thermoelasticsim.elastic.deformation import Deformer
    from thermoelasticsim.utils.utils import EV_TO_GPA

    # 第一步：创建初始系统
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)  # 108原子系统；示例足够，耗时更小
    )
    potential = EAMAl1Potential()

    # 初始化300K速度
    for atom in cell.atoms:
        sigma = np.sqrt(8.617e-5 * 300.0 / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
    cell.remove_com_motion()

    # 第二步：NPT平衡获得热膨胀结构
    npt_scheme = create_mtk_npt_scheme(
        target_temperature=300.0,  # K
        target_pressure=0.0,       # GPa
        tdamp=100.0,              # fs，温度阻尼时间
        pdamp=1000.0              # fs，压力阻尼时间
    )

    print("NPT平衡阶段...")
    for step in range(10000):
        npt_scheme.step(cell, potential, dt=1.0)
        if step % 2000 == 0:
            T = cell.calculate_temperature()
            stress = cell.calculate_stress_tensor(potential)
            P = np.trace(stress) / 3.0 * EV_TO_GPA
            print(f"  步 {step}: T={T:.1f}K, P={P:.3f}GPa, V={cell.volume:.1f}Å³")

    # 保存平衡构型作为参考
    equilibrium_cell = cell.copy()
    V0 = equilibrium_cell.volume
    print(f"平衡体积: {V0:.1f} Å³")

    # 第三步：C44剪切形变采样
    print("\nC44剪切形变采样...")
    strains_c44 = np.array([-0.004, -0.002, 0.0, 0.002, 0.004])
    stress_c44_avg = []

    for strain in strains_c44:
        # 构造剪切形变矩阵（yz剪切）
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, strain],
                      [0.0, strain, 1.0]])

        # 创建形变后的晶胞
        deformed_cell = equilibrium_cell.copy()
        deformer = Deformer()
        deformer.apply_deformation(deformed_cell, F)

        # 初始化相同温度速度
        for atom in deformed_cell.atoms:
            sigma = np.sqrt(8.617e-5 * 300.0 / atom.mass)
            atom.velocity = np.random.normal(0, sigma, 3)
        deformed_cell.remove_com_motion()

        # NVT 采样（使用 Nose–Hoover 链，参考示例脚本）
        nvt_scheme = NoseHooverNVTScheme(
            target_temperature=300.0,
            tdamp=50.0,  # fs
            tchain=3,
            tloop=1,
        )

        # 短期平衡
        for _ in range(2000):
            nvt_scheme.step(deformed_cell, potential, dt=1.0)

        # 生产采样
        stress_yz_data = []
        for step in range(10000):
            nvt_scheme.step(deformed_cell, potential, dt=1.0)
            if step % 5 == 0:  # 每5步采样一次
                stress_tensor = deformed_cell.calculate_stress_tensor(potential)
                stress_yz_data.append(stress_tensor[1, 2])  # σ_yz分量

        # 时间平均
        avg_stress = np.mean(stress_yz_data)
        std_stress = np.std(stress_yz_data)
        stress_c44_avg.append(avg_stress)

        print(f"  γ_yz={strain:+.3f}: ⟨σ_yz⟩={avg_stress:.4f}±{std_stress/np.sqrt(len(stress_yz_data)):.4f} eV/Å³")

    # 第四步：线性拟合获得 C44
    # 注意：Voigt 表示中剪切应变为工程剪切应变（2×张量剪切应变）
    engineering_strains = 2 * strains_c44  # γ = 2ε_yz
    # 使用 numpy.polyfit 进行一次线性拟合
    coeffs = np.polyfit(engineering_strains, stress_c44_avg, 1)
    slope = coeffs[0]
    C44_finite_T = slope * EV_TO_GPA  # 转换为 GPa
    # 计算 R² 与简易误差估计
    ypred = np.polyval(coeffs, engineering_strains)
    residuals = stress_c44_avg - ypred
    dof = max(len(engineering_strains) - 2, 1)
    C44_error = np.sqrt(np.sum(residuals**2) / dof) / np.sqrt(len(engineering_strains)) * EV_TO_GPA

    print(f"\n有限温度弹性常数结果:")
    print(f"C44(300K) = {C44_finite_T:.1f} ± {C44_error:.1f} GPa")
    ss_res = np.sum((np.array(stress_c44_avg) - ypred) ** 2)
    ss_tot = np.sum((np.array(stress_c44_avg) - np.mean(stress_c44_avg)) ** 2)
    r2_c44 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0
    print(f"拟合相关系数 R² = {r2_c44:.4f}")

C11 与 C12（单轴）
------------------

单轴应变模式的完整演示（同时拟合 C11 与 C12）：

.. code-block:: python

    print("\n单轴形变采样（C11 与 C12）...")
    strains_uniaxial = np.array([-0.003, -0.0015, 0.0, 0.0015, 0.003])
    stress_uniaxial_avg = []

    for strain in strains_uniaxial:
        # 构造单轴形变矩阵（x方向拉伸）
        F = np.array([[1.0 + strain, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

        # 应用形变
        deformed_cell = equilibrium_cell.copy()
        deformer = Deformer()
        deformer.apply_deformation(deformed_cell, F)

        # 重新初始化速度
        for atom in deformed_cell.atoms:
            sigma = np.sqrt(8.617e-5 * 300.0 / atom.mass)
            atom.velocity = np.random.normal(0, sigma, 3)
        deformed_cell.remove_com_motion()

        # 使用 Langevin 恒温器（示例）
        nvt_scheme = LangevinNVTScheme(
            target_temperature=300.0,
            friction=1.0  # ps⁻¹
        )

        # 平衡
        for _ in range(2000):
            nvt_scheme.step(deformed_cell, potential, dt=1.0)

        # 采样 σ_xx 与 σ_yy 分量
        stress_xx_data = []
        stress_yy_data = []
        for step in range(8000):
            nvt_scheme.step(deformed_cell, potential, dt=1.0)
            if step % 4 == 0:
                stress_tensor = deformed_cell.calculate_stress_tensor(potential)
                stress_xx_data.append(stress_tensor[0, 0])
                stress_yy_data.append(stress_tensor[1, 1])

        avg_stress = np.mean(stress_xx_data)
        std_stress = np.std(stress_xx_data)
        stress_uniaxial_avg.append(avg_stress)

        print(f"  ε_xx={strain:+.3f}: ⟨σ_xx⟩={avg_stress:.4f}±{std_stress/np.sqrt(len(stress_xx_data)):.4f} eV/Å³")

    # 线性拟合：σ_xx/ε→C11, σ_yy/ε→C12
    coeffs_c11 = np.polyfit(strains_uniaxial, stress_uniaxial_avg, 1)
    C11 = coeffs_c11[0] * EV_TO_GPA
    ypred_c11 = np.polyval(coeffs_c11, strains_uniaxial)
    ss_res_c11 = np.sum((np.array(stress_uniaxial_avg) - ypred_c11) ** 2)
    ss_tot_c11 = np.sum((np.array(stress_uniaxial_avg) - np.mean(stress_uniaxial_avg)) ** 2)
    r2_c11 = 1.0 - ss_res_c11 / ss_tot_c11 if ss_tot_c11 != 0 else 1.0

    coeffs_c12 = np.polyfit(strains_uniaxial, stress_yy_data, 1)
    C12 = coeffs_c12[0] * EV_TO_GPA
    ypred_c12 = np.polyval(coeffs_c12, strains_uniaxial)
    ss_res_c12 = np.sum((np.array(stress_yy_data) - ypred_c12) ** 2)
    ss_tot_c12 = np.sum((np.array(stress_yy_data) - np.mean(stress_yy_data)) ** 2)
    r2_c12 = 1.0 - ss_res_c12 / ss_tot_c12 if ss_tot_c12 != 0 else 1.0

    print(f"\nC11 = {C11:.1f} GPa (R²={r2_c11:.4f})")
    print(f"C12 = {C12:.1f} GPa (R²={r2_c12:.4f})")

涨落法理论概述
==================

NVT应力涨落关系
---------------

基于统计力学的涨落-响应定理，弹性常数可通过应力协方差计算：

.. math::
   C_{ijkl}^{Born} = \frac{V}{k_B T} \left( \langle\sigma_{ij}\sigma_{kl}\rangle - \langle\sigma_{ij}\rangle\langle\sigma_{kl}\rangle \right)

**优点**：
- 单次NVT模拟获得完整弹性常数张量
- 无需人为施加应变
- 自然包含非谐效应和温度依赖性

**挑战**：
- 收敛性差，需要极长模拟时间（>10⁶步）
- 强烈的有限尺寸效应
- 数值精度要求高，涨落量为小量差值

NPT应变涨落关系
---------------

在NPT系综中，也可通过应变涨落计算：

.. math::
   S_{ijkl} = \frac{k_B T}{V} \left( \langle\varepsilon_{ij}\varepsilon_{kl}\rangle - \langle\varepsilon_{ij}\rangle\langle\varepsilon_{kl}\rangle \right)

其中 :math:`S_{ijkl}` 为柔度张量，通过矩阵求逆得到刚度张量。

当前实现状态
------------

**注意**：ThermoElasticSim 中涨落法相关模块 :mod:`~thermoelasticsim.elastic.fluctuation_method` 当前为占位模块，尚未实现完整功能。

建议使用本章介绍的应力-应变法进行有限温度弹性常数计算，该方法稳定可靠且已充分验证。

统计误差与收敛性
====================

块平均误差估计
--------------

使用块平均法估计时间平均的统计误差：

.. code-block:: python

    def estimate_statistical_error(data, block_sizes=[50, 100, 200]):
        """估计应力时间序列的统计误差"""
        results = {}

        for block_size in block_sizes:
            n_blocks = len(data) // block_size
            if n_blocks < 5:
                continue

            # 计算块平均
            block_means = []
            for i in range(n_blocks):
                block_data = data[i*block_size:(i+1)*block_size]
                block_means.append(np.mean(block_data))

            # 块间标准误差
            block_std = np.std(block_means)
            block_error = block_std / np.sqrt(n_blocks)

            results[block_size] = {
                'mean': np.mean(block_means),
                'error': block_error,
                'n_blocks': n_blocks
            }

        return results

    # 应用到应力数据
    error_analysis = estimate_statistical_error(stress_yz_data)
    print("块平均误差分析:")
    for bs, result in error_analysis.items():
        print(f"  块大小{bs}: {result['mean']:.4f}±{result['error']:.4f} eV/Å³ ({result['n_blocks']}块)")

相关时间分析
------------

计算自相关时间，确定独立样本数量：

.. code-block:: python

    def calculate_correlation_time(data, max_lag=None):
        """计算应力时间序列的相关时间"""
        if max_lag is None:
            max_lag = len(data) // 4

        data = np.array(data)
        data_centered = data - np.mean(data)

        # 自相关函数
        autocorr = []
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(data_centered[:-lag], data_centered[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)

        # 找到降到1/e的时间
        autocorr = np.array(autocorr)
        tau_indices = np.where(autocorr < 1/np.e)[0]
        tau_c = tau_indices[0] if len(tau_indices) > 0 else max_lag

        return tau_c, autocorr[:tau_c+10]

    # 分析相关性
    tau_c, acf = calculate_correlation_time(stress_yz_data)
    n_independent = len(stress_yz_data) / max(tau_c, 1)

    print(f"\n相关时间分析:")
    print(f"相关时间: {tau_c} 步 (约 {tau_c*1.0:.1f} fs)")
    print(f"总样本数: {len(stress_yz_data)}")
    print(f"独立样本数: {n_independent:.0f}")

参数指导与最佳实践
======================

系统尺寸选择
------------

**最小系统**：3×3×3超胞（108原子）
  - 用途：快速测试和参数调试
  - 精度：定性结果，误差可能较大

**标准系统**：4×4×4超胞（256原子）
  - 用途：常规定量计算
  - 精度：典型误差5-10%

**高精度系统**：4×4×4超胞（256原子）
  - 用途：收敛性验证与更佳稳定性（含截断势通常 ≥4×4×4 即可）
  - 精度：误差通常 <5%

时间步长与模拟长度
------------------

**时间步长**：
  - 推荐值：1.0 fs（金属EAM势函数）
  - 验证标准：NVE能量守恒检查

**平衡阶段**：
  - NPT平衡：5000-20000步
  - 形变后平衡：1000-5000步
  - 高温需要更长平衡时间

**生产阶段**：
  - 最小：5000步（快速测试）
  - 标准：10000-20000步（定量计算）
  - 高精度：>50000步（研究级精度）

恒温器参数调节
--------------

**Andersen恒温器**：
  - 小系统：collision_frequency = 0.01 fs⁻¹
  - 大系统：collision_frequency = 0.05-0.08 fs⁻¹

**Langevin恒温器**：
  - 推荐值：friction = 0.01 fs⁻¹
  - 温度控制精度：~2%

采样策略
--------

**采样频率**：
  - 推荐：每3-5步采样一次
  - 原因：平衡计算成本与相关性

**数据质量检查**：
  - 温度稳定性：变异系数<5%
  - 应力收敛性：块平均误差分析
  - 相关时间：确保足够独立样本

小结
====

本章完整阐述了有限温度弹性常数的计算方法和ThermoElasticSim实现：

- **工作流设计**：NPT平衡→施加应变→NVT采样→线性拟合的系统化流程
- **API实践**：基于schemes架构的可复现示例，展示 C44 和 C11/C12 计算
- **统计分析**：块平均法和相关时间分析确保结果可靠性
- **涨落法概述**：理论公式与实现挑战，当前推荐应力-应变法
- **最佳实践**：系统尺寸、时间步长、采样策略的实用指导

有限温度计算的关键在于充分的统计采样和严格的误差分析，确保物理结果的准确性和可重复性。
