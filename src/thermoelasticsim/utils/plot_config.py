#!/usr/bin/env python3
"""
matplotlib字体配置模块
一劳永逸解决中文字体显示问题

Usage:
    import matplotlib.pyplot as plt
    from thermoelasticsim.utils.plot_config import setup_matplotlib
    setup_matplotlib()  # 之后就可以正常使用中文
    
    # 或者直接导入已配置的pyplot
    from thermoelasticsim.utils.plot_config import plt

Created: 2025-08-17
"""

import platform

import matplotlib

# 使用Agg后端，避免GUI问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_system_fonts():
    """获取系统可用的中文字体"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        fonts = [
            'Arial Unicode MS',  # macOS内置中文字体
            'PingFang SC',       # macOS系统字体
            'STHeiti',           # 华文黑体
            'Hiragino Sans GB',  # 冬青黑体
            'STSong',            # 华文宋体
            'SimHei',            # 黑体
            'SimSun',            # 宋体
            'DejaVu Sans',       # 后备字体
            'Arial',             # 最终后备
        ]
    elif system == 'Windows':
        fonts = [
            'Microsoft YaHei',   # 微软雅黑
            'SimHei',            # 黑体
            'SimSun',            # 宋体
            'KaiTi',             # 楷体
            'FangSong',          # 仿宋
            'DejaVu Sans',       # 后备字体
            'Arial',             # 最终后备
        ]
    else:  # Linux
        fonts = [
            'Noto Sans CJK SC',      # Google字体
            'WenQuanYi Micro Hei',   # 文泉驿微米黑
            'WenQuanYi Zen Hei',     # 文泉驿正黑
            'Droid Sans Fallback',   # Android字体
            'AR PL UMing CN',        # 文鼎字体
            'DejaVu Sans',           # 后备字体
            'Liberation Sans',       # 最终后备
        ]

    return fonts


def test_font_availability(fonts):
    """测试字体可用性"""
    from matplotlib.font_manager import FontProperties

    available_fonts = []
    for font in fonts:
        try:
            fp = FontProperties(family=font)
            # 尝试创建一个简单的图来测试字体
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontproperties=fp)
            plt.close(fig)
            available_fonts.append(font)
            break  # 找到第一个可用字体就停止
        except Exception:
            continue

    return available_fonts


def setup_matplotlib(force_english=False):
    """
    设置matplotlib的字体配置
    
    Parameters
    ----------
    force_english : bool
        是否强制使用英文字体（当中文字体有问题时）
    """
    if force_english:
        # 强制使用英文字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("INFO: 使用英文字体配置")
        return

    try:
        # 获取系统字体列表
        system_fonts = get_system_fonts()

        # 设置字体优先级
        plt.rcParams['font.sans-serif'] = system_fonts
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

        # 其他常用配置
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1

        # 测试中文字体
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '中文测试', fontsize=12)
        plt.close(fig)

        print(f"INFO: 字体配置成功，优先使用: {system_fonts[0]}")

    except Exception as e:
        # 字体配置失败，回退到英文
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"WARNING: 中文字体配置失败 ({e})，回退到英文字体")


def get_chinese_font():
    """获取第一个可用的中文字体名称"""
    system_fonts = get_system_fonts()
    available = test_font_availability(system_fonts)
    return available[0] if available else 'DejaVu Sans'


# 自动设置字体
setup_matplotlib()

# 导出配置好的pyplot，可以直接导入使用
__all__ = ['setup_matplotlib', 'get_chinese_font', 'plt']
