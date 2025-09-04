========================
零温弹性常数计算
========================

本章介绍零温下弹性常数的具体计算工作流。我们将从实际问题出发：如何通过应力-应变拟合得到立方晶系的三个独立弹性常数 C₁₁、C₁₂、C₄₄。

.. currentmodule:: thermoelasticsim.elastic

工作流概述
==========

核心问题：如何计算弹性常数
--------------------------

立方晶系材料有三个独立弹性常数：C₁₁、C₁₂、C₄₄。从热力学角度，零温弹性常数定义为内能对应变的二阶导数 :cite:`Wallace1972`：

.. math::
   C_{ijkl} = \frac{1}{V_0} \frac{\partial^2 E}{\partial \varepsilon_{ij} \partial \varepsilon_{kl}} \bigg|_{\varepsilon=0}

这一定义揭示了弹性常数的能量本质：它描述了晶格形变引起的能量曲率。对于谐振子模型，这一曲率直接决定了振动频率，因此弹性常数与晶格动力学密切相关。

我们通过以下工作流获得：

**零温弹性常数计算工作流**：

1. **构建晶体结构（超胞）** - 使用标准晶格常数创建FCC超胞
2. **基态制备** - 等比例缩放/完全弛豫获得平衡晶格常数和近零应力基态
3. **施加已知形变** - 对平衡结构施加一组小应变 :math:`\{\varepsilon^{(i)}\}`
4. **内部坐标松弛** - 每个形变结构进行力学平衡优化
5. **应力张量计算** - 计算相应的应力响应 :math:`\{\sigma^{(i)}\}`
6. **线性拟合** - 从应力-应变关系 :math:`\sigma = f(\varepsilon)` 提取弹性常数组合
7. **求解弹性常数** - 线性拟合得到 :math:`C_{11}、C_{12}、C_{44}`

形变模式设计
------------

为了独立提取三个弹性常数，本章采用两类形变模式（与 examples 一致）。这些形变模式的设计基于晶体对称性和应力响应的独立性原理：

.. math::
   \begin{aligned}
   \text{单轴形变} & \quad F = I + \varepsilon \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \quad \Rightarrow \quad \sigma_{xx} = C_{11}\,\varepsilon,\; \sigma_{yy}=\sigma_{zz}=C_{12}\,\varepsilon \\
   \text{剪切形变} & \quad F = I + \varepsilon \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \quad \Rightarrow \quad \sigma_{xy} = 2C_{44}\varepsilon
   \end{aligned}

通过求解这三个线性关系即可得到所有弹性常数。值得注意的是，除了这些基本形变，还可以使用更复杂的形变模式：

**正交形变（Orthorhombic distortion）**：保持体积不变的形变，可以直接提取 :math:`C_{11} - C_{12}`：

.. math::
   F_{\text{orth}} = \begin{bmatrix}
   1+\delta & 0 & 0 \\
   0 & 1-\delta & 0 \\
   0 & 0 & 1/(1-\delta^2)
   \end{bmatrix}

这种形变的优势在于消除了体积变化的影响，使得到的弹性常数组合更加精确。

关键API接口
============

零温弹性常数计算涉及以下核心组件：

结构构建
--------

:py:meth:`~thermoelasticsim.core.crystalline_structures.CrystallineStructureBuilder.create_fcc` 创建 FCC 晶体结构：

.. code-block:: python

    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC

    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)  # 108原子超胞
    )

形变施加
--------

:py:class:`~thermoelasticsim.elastic.deformation.Deformer` 管理晶格形变：

.. code-block:: python

    # 前提：已按"结构构建"创建变量 cell
    import numpy as np
    from thermoelasticsim.elastic.deformation import Deformer

    deformer = Deformer(delta=0.005, num_steps=5)

    # 构造形变梯度矩阵 F = I + ε
    strain_tensor = np.array([[0.005, 0, 0], [0, 0, 0], [0, 0, 0]])
    F = np.eye(3) + strain_tensor

    deformed_cell = cell.copy()
    deformer.apply_deformation(deformed_cell, F)

结构优化
--------

:py:class:`~thermoelasticsim.elastic.deformation_method.zero_temp.StructureRelaxer` 执行内部坐标松弛。这一步至关重要，因为施加形变后，原子的内部位置需要重新优化以达到力平衡。从能量角度看，这对应于在固定应变下寻找能量最小值：

.. math::
   \min_{\{\vec{u}_i\}} E(\mathbf{F}; \{\vec{u}_i\})

其中 :math:`\{\vec{u}_i\}` 是原子的内部位移。优化算法的选择影响计算效率：

- **L-BFGS**：利用Hessian近似，收敛快，适合光滑势能面
- **共轭梯度（CG）**：稳定可靠，但收敛较慢
- **FIRE**：对噪声鲁棒，适合复杂势能面

.. code-block:: python

    # 前提：已得到 deformed_cell（见上节"形变施加"和"结构构建"）
    from thermoelasticsim.elastic.deformation_method.zero_temp import StructureRelaxer
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    relaxer = StructureRelaxer(
        optimizer_type='L-BFGS',
        optimizer_params={'gtol': 1e-6, 'maxiter': 5000},
        supercell_dims=(3, 3, 3)
    )
    potential = EAMAl1Potential()
    # 形变后内部弛豫（固定晶格，优化内部坐标）
    relaxed_cell = relaxer.internal_relax(deformed_cell, potential)

应力计算
--------

通过 :meth:`~thermoelasticsim.core.structure.Cell.calculate_stress_tensor` 统一接口计算应力。在原子尺度，应力张量通过维里定理计算：

.. math::
   \sigma_{\alpha\beta} = -\frac{1}{V} \left( \sum_i m_i v_{i,\alpha} v_{i,\beta} + \sum_i r_{i,\alpha} F_{i,\beta} \right)

零温下动能项消失，应力完全由势能贡献（维里应力）：

.. math::
   \sigma_{\alpha\beta} = -\frac{1}{V} \sum_i r_{i,\alpha} F_{i,\beta} = \frac{1}{V} \sum_i r_{i,\alpha} \frac{\partial U}{\partial r_{i,\beta}}

对于两体势，这可以进一步简化为成对相互作用的贡献。对于多体势（如 Tersoff），我们仍使用
:math:`\sigma = -V^{-1}\sum r\otimes F` 的统一定义，但在实现上采用三元簇分解进行维里记账：

- 两体斥力项：:math:`(r_j-r_i)\otimes F_{ij}`
- 键序配对项（由 :math:`b_{ij}` 引起的配对力）：附带负号
- 三体吸引项：:math:`(r_j-r_i)\otimes F_j + (r_k-r_i)\otimes F_k`

最终将所有贡献相加并除以体积得到张拉为正的应力。本项目内所有公式与符号约定均与代码实现保持一致。

.. code-block:: python

    # 统一应力计算入口（本段自包含，可直接运行）
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.utils.utils import EV_TO_GPA

    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )
    potential = EAMAl1Potential()

    stress_tensor = cell.calculate_stress_tensor(potential)
    stress_GPa = stress_tensor * EV_TO_GPA  # 转换为 GPa
    print("Stress Tensor (GPa):")
    print(stress_GPa)

弹性常数求解
--------------

:class:`~thermoelasticsim.elastic.deformation_method.zero_temp.ElasticConstantsSolver` 执行应力-应变拟合。应力-应变数据的线性拟合采用最小二乘法：

.. math::
   C = \frac{\sum_i (\varepsilon_i - \bar{\varepsilon})(\sigma_i - \bar{\sigma})}{\sum_i (\varepsilon_i - \bar{\varepsilon})^2}

拟合质量通过决定系数 :math:`R^2` 评估，理想的线性响应应有 :math:`R^2 > 0.99`。

.. code-block:: python

    from thermoelasticsim.elastic.deformation_method.zero_temp import ElasticConstantsSolver

    solver = ElasticConstantsSolver()
    elastic_constants = solver.solve(strain_list, stress_list)

    C11 = elastic_constants['C11']
    C12 = elastic_constants['C12']
    C44 = elastic_constants['C44']

一键基准测试
============

最简单的方法：run_zero_temp_benchmark
----------------------------------------

:py:func:`~thermoelasticsim.elastic.benchmark.run_zero_temp_benchmark` 提供完整的一键计算：

.. code-block:: python

    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC, COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential, EAMCu1Potential

    # 铝的弹性常数（标准精度）
    al_results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=(3, 3, 3),
        precision=False
    )

    print(f"铝弹性常数 (GPa):")
    print(f"C11 = {al_results['elastic_constants']['C11']:.1f}")
    print(f"C12 = {al_results['elastic_constants']['C12']:.1f}")
    print(f"C44 = {al_results['elastic_constants']['C44']:.1f}")

    # 铜的弹性常数（高精度模式）
    cu_results = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=(3, 3, 3),
        precision=True  # 使用1e-5级微小应变
    )

    print(f"\n铜弹性常数 (GPa):")
    print(f"C11 = {cu_results['elastic_constants']['C11']:.1f}")
    print(f"C12 = {cu_results['elastic_constants']['C12']:.1f}")
    print(f"C44 = {cu_results['elastic_constants']['C44']:.1f}")

这个函数封装了完整工作流，自动处理形变模式选择、应力计算和线性拟合。

手动工作流实现
==============

从晶体构建到弹性常数的完整步骤
------------------------------

以下示例演示完整的手动工作流，展示每个步骤的具体实现：

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential
    from thermoelasticsim.elastic.deformation import Deformer
    from thermoelasticsim.elastic.deformation_method.zero_temp import StructureRelaxer, ElasticConstantsSolver
    from thermoelasticsim.utils.utils import EV_TO_GPA

    # 步骤1：构建晶体结构
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(
        element=ALUMINUM_FCC.symbol,
        lattice_constant=ALUMINUM_FCC.lattice_constant,
        supercell=(3, 3, 3)
    )

    # 步骤2：初始化工具
    potential = EAMAl1Potential()
    deformer = Deformer(delta=0.005, num_steps=5)
    relaxer = StructureRelaxer(
        optimizer_type='L-BFGS',
        optimizer_params={'ftol': 1e-6, 'gtol': 1e-6},
        supercell_dims=(3, 3, 3)
    )

    # 步骤3：手动施加两种形变模式演示
    strains_and_stresses = []

    # 单轴形变（提取 C11 与 C12）
    for strain in [-0.005, 0.0, 0.005]:
        strain_tensor = np.array([[strain, 0, 0], [0, 0, 0], [0, 0, 0]])
        F = np.eye(3) + strain_tensor

        deformed_cell = cell.copy()
        deformer.apply_deformation(deformed_cell, F)

        # 步骤4：内部坐标松弛（原位修改deformed_cell）
        relaxer.internal_relax(deformed_cell, potential)

        # 步骤5：计算应力
        stress_tensor = deformed_cell.calculate_stress_tensor(potential)
        stress_xx_GPa = stress_tensor[0, 0] * EV_TO_GPA  # C11 对应
        stress_yy_GPa = stress_tensor[1, 1] * EV_TO_GPA  # C12 对应

        strains_and_stresses.append((strain, stress_xx_GPa, stress_yy_GPa))
        print(f"单轴应变 ε={strain:+.3f}, σxx={stress_xx_GPa:.2f} GPa, σyy={stress_yy_GPa:.2f} GPa")

    # 剪切形变（提取 C44）
    shear_data = []
    for strain in [-0.005, 0.0, 0.005]:
        strain_tensor = np.array([[0, strain, 0], [strain, 0, 0], [0, 0, 0]])
        F = np.eye(3) + strain_tensor

        deformed_cell = cell.copy()
        deformer.apply_deformation(deformed_cell, F)
        relaxer.internal_relax(deformed_cell, potential)

        stress_tensor = deformed_cell.calculate_stress_tensor(potential)
        stress_xy_GPa = stress_tensor[0, 1] * EV_TO_GPA

        shear_data.append((strain, stress_xy_GPa))
        print(f"剪切应变 γ={strain:+.3f}, 应力 τxy={stress_xy_GPa:.2f} GPa")

    # 步骤6：线性拟合提取弹性常数（简化演示）
    # 线性拟合可用 numpy.polyfit，避免外部依赖

    # 单轴拟合：σxx vs ε 斜率 = C11；σyy vs ε 斜率 = C12
    strains = [item[0] for item in strains_and_stresses]
    stresses_xx = [item[1] for item in strains_and_stresses]
    stresses_yy = [item[2] for item in strains_and_stresses]
    C11_slope = np.polyfit(strains, stresses_xx, 1)[0]
    C12_slope = np.polyfit(strains, stresses_yy, 1)[0]
    print(f"\nC11 (σxx/ε): {C11_slope:.1f} GPa")
    print(f"C12 (σyy/ε): {C12_slope:.1f} GPa")

    # 剪切拟合：斜率 = 2*C44
    shear_strains = [item[0] for item in shear_data]
    shear_stresses = [item[1] for item in shear_data]
    coeffs_s = np.polyfit(shear_strains, shear_stresses, 1)
    slope_shear = coeffs_s[0]
    C44_manual = slope_shear / 2  # σxy = 2*C44*εxy
    print(f"C44: {C44_manual:.1f} GPa")

（可选）说明：模块中提供了 ElasticConstantsSolver 原型，但出于稳健性考虑，本文档推荐使用上面的"单轴 + 剪切"手动拟合流程。

单位转换
--------

应力计算结果需要从 eV/Å³ 转换为 GPa，使用 :data:`~thermoelasticsim.utils.utils.EV_TO_GPA` 常数：

.. code-block:: python

    # 前提：已有 eV/Å³ 单位的应力张量变量 stress_tensor（3×3）
    from thermoelasticsim.utils.utils import EV_TO_GPA
    stress_GPa = stress_tensor * EV_TO_GPA
    print(f"转换系数: {EV_TO_GPA:.7f}")
    print(f"应力 (GPa): {stress_GPa[0,0]:.2f}")

验证与误差分析
==============

超胞尺寸收敛性测试
------------------

验证计算结果的系统尺寸依赖性是确保精度的关键步骤。周期性边界条件下的有限尺寸效应可表示为：

.. math::
   C(L) = C_{\infty} + \frac{A}{L^{\alpha}}

其中 :math:`L` 是系统尺寸，指数 :math:`\alpha` 依赖于相互作用的衰减特性：

- 短程相互作用：:math:`\alpha \approx 3` （体积效应）
- 长程库仑相互作用：:math:`\alpha \approx 1` （表面效应）

通过不同尺寸的计算可外推到热力学极限：

.. code-block:: python

    sizes = [(2,2,2), (3,3,3), (4,4,4)]
    convergence_data = []

    for size in sizes:
        result = run_zero_temp_benchmark(
            material_params=ALUMINUM_FCC,
            potential=EAMAl1Potential(),
            supercell_size=size,
            precision=False
        )

        convergence_data.append({
            'size': f"{size[0]}×{size[1]}×{size[2]}",
            'atoms': np.prod(size) * 4,
            'C11': result['elastic_constants']['C11'],
            'C12': result['elastic_constants']['C12'],
            'C44': result['elastic_constants']['C44']
        })

    print("尺寸收敛性分析:")
    for data in convergence_data:
        print(f"{data['size']} ({data['atoms']}原子): "
              f"C11={data['C11']:.1f}, C12={data['C12']:.1f}, C44={data['C44']:.1f} GPa | "
              f"文献: C11={ALUMINUM_FCC.literature_elastic_constants['C11']:.1f}, "
              f"C12={ALUMINUM_FCC.literature_elastic_constants['C12']:.1f}, "
              f"C44={ALUMINUM_FCC.literature_elastic_constants['C44']:.1f} GPa")

实际计算中，当超胞包含 > 100 个原子时，有限尺寸误差通常 < 10%。

精度模式对比
---------------

比较标准精度与高精度模式。应变幅度的选择需要平衡两个因素：

1. **线性响应区域**：应变必须足够小以保持在线性区域内
2. **数值精度**：应变必须足够大以避免数值噪声


.. code-block:: python

    from thermoelasticsim.elastic.materials import COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMCu1Potential

    # 标准精度（δ=0.005）— Cu
    result_standard = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=(3, 3, 3),
        precision=False
    )

    # 高精度（δ=1e-5级别）— Cu
    result_precise = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=(3, 3, 3),
        precision=True
    )

    print("精度对比分析 (Cu):")
    print(f"标准: C11={result_standard['elastic_constants']['C11']:.2f} GPa")
    print(f"高精度: C11={result_precise['elastic_constants']['C11']:.2f} GPa")
    print(f"差异: {abs(result_standard['elastic_constants']['C11'] - result_precise['elastic_constants']['C11']):.3f} GPa")

势函数选择说明
==============

EAM势函数（推荐）
---------------------

Al 和 Cu 零温弹性常数计算推荐使用EAM势函数，因为其对金属材料描述精确。EAM势基于嵌入原子方法，考虑了金属中电子密度的多体效应：

.. code-block:: python

    from thermoelasticsim.potentials.eam import EAMAl1Potential, EAMCu1Potential

    # 铝：Mendelev et al. (2008)参数
    al_potential = EAMAl1Potential()

    # 铜：Mendelev et al. (2008)参数
    cu_potential = EAMCu1Potential()

    # 直接用于应力计算
    stress = cell.calculate_stress_tensor(al_potential)

Lennard-Jones说明
-----------------

LJ示例须显式使用NeighborList，本阶段不提供LJ代码实现。LJ势更适合简单原子（稀有气体）的教学演示，实际弹性常数计算建议使用EAM势。

高阶效应考虑
============

非谐效应
--------

大应变下必须考虑能量展开的高阶项：

.. math::
   E = E_0 + \frac{1}{2}C_{ijkl}\varepsilon_{ij}\varepsilon_{kl} + \frac{1}{6}C_{ijklmn}\varepsilon_{ij}\varepsilon_{kl}\varepsilon_{mn} + ...

三阶弹性常数 :math:`C_{ijklmn}` 描述了非线性弹性效应，在以下情况变得重要：

1. **大应变形变**: :math:`\varepsilon > 0.01`
2. **声子-声子相互作用**: 决定热导率和声子寿命
3. **压力导数**: :math:`\partial C_{ij}/\partial P`

Born稳定性与相变
----------------

弹性常数与晶格动力学稳定性密切相关。长波声学声子的色散关系为：

.. math::
   \omega^2(\vec{q}) = \frac{1}{\rho} C_{ijkl} q_i q_j n_k n_l

其中 :math:`\vec{n}` 是极化方向。Born稳定性条件确保所有声学模式频率的平方为正，即 :math:`\omega^2 > 0`。违反Born稳定性意味着存在 :math:`\omega^2 < 0` 的软模，系统将自发发生结构相变以降低能量。

小结
====

本章展示了零温弹性常数计算的两种路径：

- **一键方案**：:py:func:`~thermoelasticsim.elastic.benchmark.run_zero_temp_benchmark` 完整封装
- **手动流程**：从 :py:meth:`~thermoelasticsim.core.crystalline_structures.CrystallineStructureBuilder.create_fcc` 开始，采用"单轴 + 剪切"两类形变的应力-应变拟合
- **核心计算**：通过 :py:meth:`~thermoelasticsim.core.structure.Cell.calculate_stress_tensor` 统一应力接口
- **单位转换**：使用 :data:`~thermoelasticsim.utils.utils.EV_TO_GPA` 实现 eV/Å³ → GPa

所有API引用均指向真实实现，示例代码可直接运行。下一章将介绍分子动力学系综的理论与实现。
