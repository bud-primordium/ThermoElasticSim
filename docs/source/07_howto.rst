==============
How-To 指南
==============

本章提供常见任务的实用代码模板和解决方案。

计算特定材料的弹性常数
======================

铝的完整计算
------------

.. code-block:: python

    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential

    # 标准计算
    results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=(3, 3, 3),
        output_dir='results/Al',
        save_json=True,
        precision=False
    )

    # 打印结果
    print(f"铝的弹性常数 (0K):")
    print(f"  C11 = {results['elastic_constants']['C11']:.1f} GPa")
    print(f"  C12 = {results['elastic_constants']['C12']:.1f} GPa")
    print(f"  C44 = {results['elastic_constants']['C44']:.1f} GPa")
    print(f"  体模量 B = {results['elastic_moduli']['bulk_modulus']:.1f} GPa")

铜的完整计算
------------

.. code-block:: python

    from thermoelasticsim.elastic.materials import COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMCu1Potential

    # 高精度计算
    results = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=(3, 3, 3),
        output_dir='results/Cu',
        save_json=True,
        precision=True  # 使用高精度模式
    )

自定义材料参数
--------------

.. code-block:: python

    from thermoelasticsim.elastic.materials import MaterialParameters
    from thermoelasticsim.potentials.eam import EAMAl1Potential

    # 创建自定义材料（以 Al 为例，便于直接使用 EAMAl1 势）
    my_material = MaterialParameters(
        name="CustomAl",
        symbol="Al",
        mass_amu=26.9815,       # amu
        lattice_constant=4.06,  # Å（略作调整）
        structure="fcc",
        literature_elastic_constants={"C11": 110.0, "C12": 61.0, "C44": 33.0}
    )

    # 使用自定义材料 + 对应势函数
    results = run_zero_temp_benchmark(
        material_params=my_material,
        potential=EAMAl1Potential(),
        supercell_size=(3, 3, 3)
    )

LJ体系最小示例
==============

完整的LJ计算流程
----------------

.. code-block:: python

    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.potentials.lennard_jones import LennardJonesPotential
    from thermoelasticsim.utils.utils import NeighborList, EV_TO_GPA
    from thermoelasticsim.elastic.deformation import Deformer
    from thermoelasticsim.elastic.deformation_method.zero_temp import StructureRelaxer

    # 1. 创建氩晶体（使用较小的 3×3×3 超胞，示例足够）
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc('Ar', lattice_constant=5.26, supercell=(3, 3, 3))

    # 2. 设置LJ势（以邻居列表驱动）
    lj = LennardJonesPotential(
        epsilon=0.0104,  # eV
        sigma=3.40,      # Å
        cutoff=12.5      # Å (约3.7σ)
    )
    neighbor_list = NeighborList(cutoff=12.5, skin=0.5)

    # 3. 包装势函数：自动维护邻居列表
    class LJWithNeighborList:
        def __init__(self, lj_potential, neighbor_list):
            self.lj = lj_potential
            self.nl = neighbor_list
        def calculate_forces(self, cell):
            self.nl.build(cell)
            self.lj.calculate_forces(cell, self.nl)
        def calculate_energy(self, cell):
            self.nl.build(cell)
            return self.lj.calculate_energy(cell, self.nl)
    potential_wrapped = LJWithNeighborList(lj, neighbor_list)

    # 4. 形变与弛豫器（API 参数名对齐源码）
    deformer = Deformer(delta=0.01, num_steps=5)
    relaxer = StructureRelaxer(
        optimizer_type='L-BFGS',
        optimizer_params={'gtol': 1e-5, 'maxiter': 5000}
    )

    # 5. 计算 C11 与 C12（单轴 σ_xx/ε、σ_yy/ε 的斜率）
    strains = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    stresses_xx = []  # σ_xx (eV/Å³)
    stresses_yy = []  # σ_yy (eV/Å³)

    for strain in strains:
        # 施加形变 F = diag(1+ε, 1, 1)
        F = np.eye(3)
        F[0, 0] = 1.0 + strain
        deformed = cell.copy()
        deformer.apply_deformation(deformed, F)

        # 形变后内部弛豫（固定晶格，优化内部坐标）
        relaxer.internal_relax(deformed, potential_wrapped)

        # 应力（维里项；单位 eV/Å³），取 σ_xx
        stress_tensor = deformed.calculate_stress_tensor(potential_wrapped)
        stresses_xx.append(stress_tensor[0, 0])
        stresses_yy.append(stress_tensor[1, 1])
        print(f"应变 {strain:+.3f}: σ_xx = {stress_tensor[0,0]:.6f} eV/Å³, σ_yy = {stress_tensor[1,1]:.6f} eV/Å³")

    # 6. 线性拟合与换算到 GPa
    coeffs_xx = np.polyfit(strains, stresses_xx, 1)
    coeffs_yy = np.polyfit(strains, stresses_yy, 1)
    C11 = coeffs_xx[0] * EV_TO_GPA
    C12 = coeffs_yy[0] * EV_TO_GPA

    # 计算 R²（决定系数）
    ypred_xx = np.polyval(coeffs_xx, strains)
    ypred_yy = np.polyval(coeffs_yy, strains)
    ss_res_xx = np.sum((np.array(stresses_xx) - ypred_xx) ** 2)
    ss_tot_xx = np.sum((np.array(stresses_xx) - np.mean(stresses_xx)) ** 2)
    ss_res_yy = np.sum((np.array(stresses_yy) - ypred_yy) ** 2)
    ss_tot_yy = np.sum((np.array(stresses_yy) - np.mean(stresses_yy)) ** 2)
    r2_xx = 1.0 - ss_res_xx / ss_tot_xx if ss_tot_xx != 0 else 1.0
    r2_yy = 1.0 - ss_res_yy / ss_tot_yy if ss_tot_yy != 0 else 1.0

    print(f"\nLJ氩体系:")
    print(f"C11 ≈ {C11:.2f} GPa (R²={r2_xx:.6f})")
    print(f"C12 ≈ {C12:.2f} GPa (R²={r2_yy:.6f})")

处理周期性边界
--------------

.. code-block:: python

    # 确保LJ势正确处理PBC
    def check_pbc_consistency(cell, potential):
        """检查PBC实现的一致性"""
        # 保存原始位置
        orig_pos = cell.atoms[0].position.copy()

        # 计算原始能量
        E1 = potential.calculate_energy(cell)

        # 移动原子跨越边界
        cell.atoms[0].position += cell.lattice_vectors[0]
        E2 = potential.calculate_energy(cell)

        # 恢复
        cell.atoms[0].position = orig_pos

        # 能量应该相同
        assert abs(E1 - E2) < 1e-10, f"PBC错误: {E1} != {E2}"
        print("PBC一致性检查通过")

尺寸收敛测试
============

系统尺寸扫描
------------

.. code-block:: python

    import numpy as np

    def size_convergence_study(material='Al', sizes=None):
        """研究超胞尺寸对弹性常数的影响"""

        if sizes is None:
            # 示例范围不超过 4×4×4；555 对含截断的势并不优于 4×4×4
            sizes = [(2,2,2), (3,3,3), (4,4,4)]

        from thermoelasticsim.elastic.materials import ALUMINUM_FCC
        from thermoelasticsim.potentials.eam import EAMAl1Potential

        results = []

        for size in sizes:
            print(f"\n计算 {size} 超胞...")

            res = run_zero_temp_benchmark(
                material_params=ALUMINUM_FCC,
                potential=EAMAl1Potential(),
                supercell_size=size,
                save_json=False
            )

            n_atoms = np.prod(size) * 4  # FCC
            results.append({
                'size': size,
                'n_atoms': n_atoms,
                'C11': res['elastic_constants']['C11'],
                'C12': res['elastic_constants']['C12'],
                'C44': res['elastic_constants']['C44'],
                'time': res.get('computation_time', 0)
            })

        # 简要输出结果（避免外部依赖如 pandas/matplotlib）
        print("\n尺寸收敛结果：")
        for row in results:
            size = row['size']
            print(f"  {size}: C11={row['C11']:.2f} GPa, C12={row['C12']:.2f} GPa, C44={row['C44']:.2f} GPa, N={row['n_atoms']} | "
                  f"文献: C11={ALUMINUM_FCC.literature_elastic_constants['C11']:.1f}, "
                  f"C12={ALUMINUM_FCC.literature_elastic_constants['C12']:.1f}, "
                  f"C44={ALUMINUM_FCC.literature_elastic_constants['C44']:.1f} GPa")
        return results

    # 运行收敛测试
    _ = size_convergence_study('Al')

误差条估计
----------

.. code-block:: python

    from dataclasses import replace
    from thermoelasticsim.potentials.eam import EAMAl1Potential

    def calculate_with_error_bars(material, size=(4,4,4), n_runs=5):
        """多次运行估计统计误差"""

        all_results = []

        for run in range(n_runs):
            print(f"运行 {run+1}/{n_runs}...")

            # 添加随机扰动（对 MaterialParameters 使用 dataclasses.replace）
            perturbed_material = replace(
                material,
                lattice_constant=material.lattice_constant * (1 + np.random.normal(0, 1e-5))
            )

            res = run_zero_temp_benchmark(
                material_params=perturbed_material,
                potential=EAMAl1Potential(),
                supercell_size=size,
                save_json=False
            )

            all_results.append(res['elastic_constants'])

        # 统计分析
        C11_values = [r['C11'] for r in all_results]
        C12_values = [r['C12'] for r in all_results]
        C44_values = [r['C44'] for r in all_results]

        print(f"\n统计结果 ({n_runs}次运行):")
        print(f"C11 = {np.mean(C11_values):.1f} ± {np.std(C11_values):.1f} GPa")
        print(f"C12 = {np.mean(C12_values):.1f} ± {np.std(C12_values):.1f} GPa")
        print(f"C44 = {np.mean(C44_values):.1f} ± {np.std(C44_values):.1f} GPa")

回归测试
========

验证计算正确性
--------------

.. code-block:: python

    def regression_test():
        """回归测试确保结果一致性"""

        # 预期值（来自之前的验证运行）
        expected = {
            'Al_C11': 114.3,  # GPa
            'Al_C12': 61.9,
            'Al_C44': 31.6,
            'Cu_C11': 176.0,
            'Cu_C12': 124.0,
            'Cu_C44': 82.0
        }

        tolerance = 1.0  # GPa

        # 测试铝
        al_results = run_zero_temp_benchmark(
            ALUMINUM_FCC,
            EAMAl1Potential(),
            (3, 3, 3),
            save_json=False
        )

        assert abs(al_results['elastic_constants']['C11'] - expected['Al_C11']) < tolerance
        assert abs(al_results['elastic_constants']['C12'] - expected['Al_C12']) < tolerance
        assert abs(al_results['elastic_constants']['C44'] - expected['Al_C44']) < tolerance

        print("✓ 铝回归测试通过")

        # 测试铜
        cu_results = run_zero_temp_benchmark(
            COPPER_FCC,
            EAMCu1Potential(),
            (3, 3, 3),
            save_json=False
        )

        assert abs(cu_results['elastic_constants']['C11'] - expected['Cu_C11']) < tolerance * 2
        assert abs(cu_results['elastic_constants']['C12'] - expected['Cu_C12']) < tolerance * 2
        assert abs(cu_results['elastic_constants']['C44'] - expected['Cu_C44']) < tolerance * 2

        print("✓ 铜回归测试通过")

        print("\n所有回归测试通过！")

    # 运行测试
    regression_test()

单元测试模板
------------

.. code-block:: python

    import unittest
    import numpy as np
    from thermoelasticsim.core.crystalline_structures import CrystallineStructureBuilder
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC, COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential, EAMCu1Potential
    from thermoelasticsim.elastic.deformation_method.zero_temp import StructureRelaxer
    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark

    class TestElasticConstants(unittest.TestCase):
        """弹性常数计算的单元测试"""

        def setUp(self):
            """设置测试环境"""
            self.builder = CrystallineStructureBuilder()
            self.potential = EAMAl1Potential()

        def test_cubic_symmetry(self):
            """测试立方对称性"""
            cell = self.builder.create_fcc('Al', 4.05, (2, 2, 2))

            # 计算应力张量
            stress = cell.calculate_stress_tensor(self.potential)

            # 检查对称性
            np.testing.assert_allclose(stress[0,1], stress[1,0], rtol=1e-10)
            np.testing.assert_allclose(stress[0,2], stress[2,0], rtol=1e-10)
            np.testing.assert_allclose(stress[1,2], stress[2,1], rtol=1e-10)

        def test_zero_strain_zero_stress(self):
            """测试零应变零应力"""
            cell = self.builder.create_fcc('Al', 4.05, (3, 3, 3))

            # 完全弛豫到近零应力基态
            relaxer = StructureRelaxer(
                optimizer_type='L-BFGS',
                optimizer_params={'gtol': 1e-6},
                supercell_dims=(3, 3, 3)
            )
            ok = relaxer.full_relax(cell, self.potential)
            self.assertTrue(ok)

            # 应力应该接近零
            stress = cell.calculate_stress_tensor(self.potential)
            np.testing.assert_allclose(stress.diagonal(), 0, atol=1e-4)

        def test_born_stability(self):
            """测试Born稳定性条件"""
            results = run_zero_temp_benchmark(
                ALUMINUM_FCC,
                self.potential,
                (3, 3, 3)
            )

            C11 = results['elastic_constants']['C11']
            C12 = results['elastic_constants']['C12']
            C44 = results['elastic_constants']['C44']

            # Born条件
            self.assertGreater(C11 - C12, 0)
            self.assertGreater(C11 + 2*C12, 0)
            self.assertGreater(C44, 0)

    if __name__ == '__main__':
        unittest.main()

批处理脚本
==========

参数扫描
--------

.. code-block:: python

    import itertools
    import json
    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC, COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential, EAMCu1Potential

    def parameter_scan():
        """扫描不同参数组合（无第三方依赖）"""

        materials = ['Al', 'Cu']
        sizes = [(3,3,3), (4,4,4)]
        precisions = [False, True]

        all_results = []

        for mat, size, prec in itertools.product(materials, sizes, precisions):
            print(f"\n处理: {mat}, {size}, precision={prec}")
            material = ALUMINUM_FCC if mat == 'Al' else COPPER_FCC
            potential = EAMAl1Potential() if mat == 'Al' else EAMCu1Potential()

            try:
                res = run_zero_temp_benchmark(
                    material_params=material,
                    potential=potential,
                    supercell_size=size,
                    precision=prec,
                    save_json=False
                )
                all_results.append({
                    'material': mat,
                    'size': size,
                    'precision': prec,
                    'C11': res['elastic_constants']['C11'],
                    'C12': res['elastic_constants']['C12'],
                    'C44': res['elastic_constants']['C44'],
                    'time': res.get('computation_time', 0)
                })
            except Exception as e:
                print(f"  错误: {e}")
                continue

        with open('parameter_scan_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n参数扫描结果:")
        for r in all_results:
            print(f"  {r['material']} {r['size']} precision={r['precision']}: C11={r['C11']:.1f}, C12={r['C12']:.1f}, C44={r['C44']:.1f} GPa")

        return all_results

并行计算
--------

说明：并行化示例依赖额外的序列化工具函数，后续将以完整工具提供。当前版本推荐使用单进程示例以保证可运行性。

数据后处理
==========

结果可视化
----------

说明：可视化示例依赖 matplotlib/seaborn，安装后可按需绘制。为确保示例可运行性，本节省略具体代码。

导出为其他格式
--------------

说明：CSV/LaTeX/图形导出属于配套工具，后续将以 utils 子模块提供。当前版本建议直接打印或保存 JSON。

小结
====

本章提供了实用的代码模板：

- **材料计算**：Al、Cu及自定义材料
- **LJ体系**：完整的计算流程
- **收敛测试**：尺寸和误差分析
- **回归测试**：确保计算正确性
- **批处理**：参数扫描和并行计算
- **数据处理**：可视化和导出

这些模板可直接用于实际计算任务。
