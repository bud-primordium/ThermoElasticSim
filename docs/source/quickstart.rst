================
快速开始
================

欢迎使用ThermoElasticSim！本页面将帮助您在1分钟内运行第一个弹性常数计算。

一键运行示例
============

计算铝的弹性常数只需几行代码：

.. code-block:: python

    from thermoelasticsim.elastic.benchmark import run_zero_temp_benchmark
    from thermoelasticsim.elastic.materials import ALUMINUM_FCC
    from thermoelasticsim.potentials.eam import EAMAl1Potential

    # 运行基准测试
    results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=(3, 3, 3)
    )

    # 输出结果
    print(f"C11 = {results['elastic_constants']['C11']:.1f} GPa")
    print(f"C12 = {results['elastic_constants']['C12']:.1f} GPa")
    print(f"C44 = {results['elastic_constants']['C44']:.1f} GPa")

这段代码将：

1. 创建一个3×3×3的铝FCC超胞（108个原子）
2. 使用EAM势函数描述原子间相互作用
3. 应用6种独立形变模式
4. 优化原子位置并计算应力
5. 通过线性拟合得到弹性常数

预期输出
========

运行上述代码，您将看到类似以下的输出::

    INFO:thermoelasticsim.elastic.benchmark:开始Al弹性常数基准：超胞=(3, 3, 3)（原子数=108）
    WARNING:thermoelasticsim.elastic.benchmark:超胞过小: 最小半盒长 6.067 Å <= 截断半径 6.500 Å。建议增大尺寸以减少体积相关误差。
    INFO:thermoelasticsim.elastic.benchmark:制备无应力基态（优先等比例晶格弛豫）...
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:开始等比例晶格弛豫：只优化晶格常数
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:等比例晶格弛豫成功
    INFO:thermoelasticsim.elastic.benchmark:基态应力(GPa): [-0.000003, 0.002750, 0.002750; 0.002750, -0.000003, 0.002750; 0.002750, 0.002750, -0.000003]
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=-3.000000e-03 → σxx=-0.300064 GPa, σyy=-0.181577 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=-2.000000e-03 → σxx=-0.198274 GPa, σyy=-0.120700 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=-1.000000e-03 → σxx=-0.098262 GPa, σyy=-0.060181 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=-5.000000e-04 → σxx=-0.048915 GPa, σyy=-0.030050 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=0.000000e+00 → σxx=-0.000003 GPa, σyy=-0.000003 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=5.000000e-04 → σxx=0.048478 GPa, σyy=0.029963 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=1.000000e-03 → σxx=0.096533 GPa, σyy=0.059849 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=2.000000e-03 → σxx=0.191378 GPa, σyy=0.119391 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.benchmark:单轴应变 εxx=3.000000e-03 → σxx=0.284569 GPa, σyy=0.178639 GPa（收敛=True）
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:开始C44剪切响应计算
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:制备无应力基态（优先等比例晶格弛豫）...
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:开始等比例晶格弛豫：只优化晶格常数
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:等比例晶格弛豫成功
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:计算yz剪切(C44)响应...
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  yz剪切(C44): 30.53 GPa (R²=0.999267, 收敛率=100.0%)
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:计算xz剪切(C55)响应...
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  xz剪切(C55): 30.73 GPa (R²=0.999092, 收敛率=100.0%)
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:计算xy剪切(C66)响应...
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  xy剪切(C66): 30.33 GPa (R²=0.998119, 收敛率=100.0%)
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:C44剪切响应计算完成
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  平均C44: 30.53 ± 0.16 GPa
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  平均拟合优度: 0.998826
    INFO:thermoelasticsim.elastic.deformation_method.zero_temp:  总收敛率: 100.0% (39/39)
    C11 = 97.4 GPa
    C12 = 60.0 GPa
    C44 = 30.5 GPa

自定义材料
==========

您也可以计算铜或其他FCC材料：

.. code-block:: python

    from thermoelasticsim.elastic.materials import COPPER_FCC
    from thermoelasticsim.potentials.eam import EAMCu1Potential

    # 计算铜的弹性常数
    results = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=(4, 4, 4)  # 使用更大的超胞
    )

下一步
======

- 阅读 :doc:`conventions` 了解术语和约定
- 学习 :doc:`01_elastic_fundamentals` 理解理论基础
- 探索 :doc:`02_zero_temperature_elastic` 掌握详细工作流程
- 查看 :doc:`08_howto` 获取更多实用示例

常见问题
========

**Q: 超胞大小如何选择？**

A: 一般建议：

- 快速测试：(3, 3, 3) - 108 原子
- 标准/高精度：(4, 4, 4) - 256 原子（含截断势下通常已足够；更大尺寸收益有限）

**Q: 计算需要多长时间？**

A: 在标准桌面CPU上：
零温Al弹性常数计算（:code:`examples/legacy_py/zero_temp_al_benchmark.py`）

- (2, 2, 2) 超胞：约 1.3 秒
- (3, 3, 3) 超胞：约 7.8 秒
- (4, 4, 4) 超胞：约 15.3 秒

**Q: 如何提高精度？**

A: 使用precision模式：

.. code-block:: python

    results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=(4, 4, 4),
        precision=True  # 启用高精度模式
    )

这将使用更小的应变幅度和更严格的优化收敛标准。
