=====================
YAML 场景快速上手
=====================

本项目提供基于 YAML 的一键运行场景，便于教学与复现。所有示例位于
``examples/modern_yaml/`` 目录，包含完整工作流与教学拆分单元。

运行方法
========

.. code-block:: bash

   # 零温完整（多尺寸扫描）
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/zero_temp_elastic.yaml

   # 有限温完整（预热→NPT→NHC生产）
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/finite_temp_elastic.yaml

   # 教学单元（弛豫/NVE/NVT/NPT）
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/relax.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nve.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_langevin.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_nhc.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_andersen.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_berendsen.yaml
   python -m thermoelasticsim.cli.run -c examples/modern_yaml/npt.yaml

参数说明
========

每个 YAML 文件包含详细中文注释，覆盖：

- 材料与结构：``material: {symbol: Al|Cu|C, structure: fcc|diamond}``
- 势函数：``potential: EAM_Al1|EAM_Cu1|Tersoff_C1988``
- 步长/步数/采样：``dt``、``steps``、``sample_every``
- 恒温/恒压参数：``friction``、``tdamp``、``pdamp``、``tchain/pchain/tloop`` 等
- 零温应变点：显式 ``uniaxial_strains/shear_strains`` 或 ``strain_amplitude/num_points/include_zero``

输出产物
========

所有产物保存至 ``examples/logs/{name}_{timestamp}`` 目录，包括：

- 零温：图/CSV/JSON 与尺寸扫描汇总
- 有限温：``npt_pressure_evolution.png``、``C11/C12/C44`` 拟合图与 JSON
- 教学单元：``thermo.csv``、温度/能量/压力曲线等
