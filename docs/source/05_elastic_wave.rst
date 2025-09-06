弹性波传播模拟
===============

本章介绍ThermoElasticSim中弹性波传播的理论基础和模拟方法。

理论背景
--------

弹性波基础
~~~~~~~~~~

弹性波是固体中应力扰动的传播现象 :cite:`musgrave1970crystal`，主要分为两类：

1. **纵波（L波）**：粒子振动方向与传播方向平行，压缩波
2. **横波（T波）**：粒子振动方向与传播方向垂直，剪切波

在立方晶系中，波速由弹性常数和密度决定：

.. math::

    v = \sqrt{\frac{M}{\rho}}

其中M是相应的弹性模量，ρ是材料密度。

Christoffel方程
~~~~~~~~~~~~~~~

任意方向的波速通过求解Christoffel方程获得 :cite:`christoffel1877ueber`：

.. math::

    (\Gamma_{ik} - \rho v^2 \delta_{ik}) u_k = 0

其中Christoffel矩阵元素为：

.. math::

    \Gamma_{ik} = C_{ijkl} n_j n_l

这里n是传播方向的单位矢量，C是弹性常数张量。

特殊方向波速
~~~~~~~~~~~~

对于立方晶系的特殊方向，波速有解析解 :cite:`every1980general`：

**[100]方向**：

- 纵波：:math:`v_L = \sqrt{C_{11}/\rho}`
- 横波：:math:`v_T = \sqrt{C_{44}/\rho}` （二重简并）

**[110]方向**：

- 纵波：:math:`v_L = \sqrt{(C_{11}+C_{12}+2C_{44})/(2\rho)}`
- 横波1：:math:`v_{T1} = \sqrt{C_{44}/\rho}`
- 横波2：:math:`v_{T2} = \sqrt{(C_{11}-C_{12})/(2\rho)}`

**[111]方向**：

- 纵波：:math:`v_L = \sqrt{(C_{11}+2C_{12}+4C_{44})/(3\rho)}`
- 横波：:math:`v_T = \sqrt{(C_{11}-C_{12}+C_{44})/(3\rho)}` （二重简并）

模拟方法
--------

Phase A：解析计算
~~~~~~~~~~~~~~~~~

解析计算模块实现了基于弹性常数的波速计算：

.. code-block:: python

    from thermoelasticsim.elastic.wave import ElasticWaveAnalyzer

    # 创建分析器（Al的弹性常数，单位GPa）
    analyzer = ElasticWaveAnalyzer(
        C11=110.0, C12=61.0, C44=33.0,
        density=2.70  # g/cm³
    )

    # 计算[100]方向的波速
    result = analyzer.calculate_wave_velocities([1, 0, 0])
    print(f"纵波速度: {result['longitudinal']:.2f} km/s")
    print(f"横波速度: {result['transverse1']:.2f} km/s")

Phase B：动力学模拟
~~~~~~~~~~~~~~~~~~~

分子动力学模拟直接观察波在晶格中的传播：

.. code-block:: python

    from thermoelasticsim.elastic.wave.dynamics import (
        DynamicsConfig, simulate_plane_wave_mvp
    )

    # 配置动力学参数（2分钟内可完成的教学配置）
    config = DynamicsConfig(
        supercell=(24, 6, 6),   # 适中超胞，快速运行
        steps=1200,             # 约600 fs模拟时间
        polarization="L",       # 纵波
        source_type="gaussian", # 高斯脉冲激发
        source_amplitude_velocity=3e-4,
        source_t0_fs=250.0,
        source_sigma_fs=70.0,
        detector_frac_a=0.25,   # 探测点A位置
        detector_frac_b=0.80,   # 探测点B位置
        measure_method="arrival"  # 多点到达时间拟合
    )

    # 运行模拟
    result = simulate_plane_wave_mvp(
        material_symbol="Al",
        dynamics=config,
        out_xt_path="wave_xt.png"
    )

波源激发方式
~~~~~~~~~~~~

系统支持多种波源激发方式：

1. **高斯脉冲**：宽频激发，适合观察色散

   .. math::

      v(t) = A \exp\left(-\frac{(t-t_0)^2}{2\sigma^2}\right)

2. **Tone Burst**：窄带激发，适合精确测速

   .. math::

      v(t) = A \sin(2\pi f t) \cdot w(t)

   其中w(t)是汉宁窗包络。

实现要点
~~~~~~~~

- **源注入策略**：在x≈0的薄片区域（默认占Lx的6%）对速度施加时间域脉冲。每步注入后移除质心平动，避免整体漂移。

- **吸收边界（海绵层）**：左右两端各占Lx约10-12%的区域，速度按exp(-(dt/τ)·w(x))指数衰减，其中w(x)为位置相关的权重函数（cos²或线性）。可有效减少边界反射和PBC绕回。

- **测速策略**：
  - 高斯源优先使用多探测点到达时间拟合（arrival方法）
  - 设置早期时间窗口（t_gate到t_early_end），避免晚期干涉导致的伪峰
  - 横波测速时使用纵波约束（L guard），剔除纵波污染
  - 互相关法限制正滞后并设置物理速度上限v_max约束

速度测量算法
~~~~~~~~~~~~

实现了多种波速估计方法：

1. **到达时间拟合**：追踪波前到达多个探测点的时间，线性拟合t(x)获得速度
2. **互相关法**：计算两探测点信号的时间延迟，仅考虑正滞后并受v_max约束
3. **k-ω谱分析**：傅里叶域的相速度提取
4. **横波L约束**：利用纵波先到达的物理约束，配合幅值门控
5. **早期时间窗**：到达拟合与互相关默认仅在早期窗内进行，避免晚期干涉导致的伪峰或抬阈值

可视化输出
~~~~~~~~~~

生成的x-t二联图（双面板图）：

- **左图**：位移场u·e的时空演化
  - 色标以0为中心（红蓝对称）
  - 黑色虚线标记探测点位置
  - 若到达拟合成功，叠加绿色虚线显示波前t=a+b·x

- **右图**：包络|u·e|的RMS时空图
  - 热色图显示波包传播亮带
  - 白色虚线标记探测点位置
  - 同样叠加拟合波前线

**物理图像说明**：向+x传播的波在x-t图中呈现"左下→右上"的斜纹或亮带，斜率的倒数即为波速v=dx/dt。

符号说明
~~~~~~~~

- **e**：极化单位向量（L: ex；Ty: ey；Tz: ez）
- **u·e**：位移在极化方向的投影（x-t图的颜色量）
- **v_max**：互相关中用于设定"最小物理滞后"的速度上限（L约7.0 km/s，T约5.0 km/s）

YAML配置示例
------------

完整的YAML配置文件示例：

.. code-block:: yaml

    scenario: elastic_wave

    material: { symbol: Al, structure: fcc }

    wave:
      density: 2.70  # g/cm³

      # 解析计算配置
      visualization:
        enabled: true
        planes: ["001", "110", "111"]

      # 动力学模拟配置
      dynamics:
        enabled: true
        supercell: [24, 6, 6]   # 教学默认，2分钟内可完成
        steps: 1200             # 约600 fs
        polarization: L         # L, Ty, 或 Tz

        # 源注入参数
        source:
          enabled: true
          type: gaussian
          amplitude_velocity: 3.0e-4  # 线性响应区
          t0_fs: 250.0
          sigma_fs: 70.0

        # 探测点位置（占Lx的比例）
        detectors: [0.25, 0.80]  # 默认值

        # 测速方法
        measure:
          method: arrival       # arrival优先（高斯源）
          v_max_km_s: 7.0      # 纵波物理上限

        # 吸收边界（建议开启）
        absorber:
          enabled: true
          slab_fraction: 0.12
          tau_fs: 250.0
          profile: cos2

运行命令
--------

使用CLI运行弹性波模拟：

.. code-block:: bash

    # 运行解析计算和可视化
    python -m thermoelasticsim.cli.run -c examples/modern_yaml/elastic_wave.yaml

    # 运行纵波传播模拟
    python -m thermoelasticsim.cli.run -c examples/modern_yaml/elastic_wave_dynamics_L.yaml

    # 运行横波传播模拟
    python -m thermoelasticsim.cli.run -c examples/modern_yaml/elastic_wave_dynamics_T.yaml

输出文件
--------

模拟生成的文件包括：

- ``wave_velocities.json``：各方向的解析波速
- ``wave_xt.png``：x-t二联图（位移场和包络）
- ``wave_trajectory.h5``：完整的原子轨迹（H5MD格式，可选）
- ``wave_trajectory.gif``：传播动画（可选）
- ``analytic_anisotropy_001.png``：(001)平面的极坐标图
- ``analytic_anisotropy_110.png``：(110)平面的极坐标图（如配置）
- ``analytic_anisotropy_111.png``：(111)平面的极坐标图（如配置）

物理参数建议
------------

获得良好模拟结果的参数建议：

**超胞尺寸**：

- 传播方向（x）：至少24个晶格常数（教学）或40个（精确）
- 垂直方向（y,z）：6-10个晶格常数

**时间步长**：

- 纵波：0.5-1.0 fs
- 横波：1.0-2.0 fs

**源参数**：

- 振幅：2.5e-4 到 4e-4 Å/fs（线性响应区）
- 高斯宽度：60-90 fs
- 源区域：占总长度的5-10%

**吸收边界**：

- 建议始终开启，特别是纵波模拟
- 海绵层厚度：左右各10-12%
- 衰减时间常数：200-300 fs

**温度控制**：

- 默认使用零温NVE模拟
- 可选极低温（1-10 K）减少热噪声

常见问题
--------

**Q: 为什么模拟波速与理论值有偏差？**

A: 主要原因包括：

1. 有限尺寸效应（10-15%误差正常）
2. 数值色散（高频成分）
3. 边界反射干扰

**Q: 如何区分纵波和横波？**

A: 通过以下特征：

1. 速度差异（纵波通常快1.5-2倍）
2. 偏振方向（检查位移矢量）
3. 到达时间（纵波先到）

**Q: 吸收边界何时需要？**

A: 当观察到：

1. 后期信号混乱（边界反射）
2. x-t图出现交叉条纹
3. 速度估计不稳定

参考文献
--------

.. bibliography:: references.bib
   :cited:
   :style: unsrt
