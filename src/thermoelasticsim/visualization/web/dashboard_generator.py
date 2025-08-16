#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web仪表板生成器

使用Jinja2模板引擎生成交互式HTML仪表板，提供：
- 现代化的UI设计
- 真正的JavaScript交互功能  
- 模块化的模板系统
- 响应式布局

Author: Gilbert Young
Created: 2025-08-15
"""

from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
import base64
import mimetypes

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    仪表板生成器
    
    使用Jinja2模板引擎生成高质量的HTML仪表板。
    
    Parameters
    ----------
    template_dir : str, optional
        模板目录路径，默认使用内置模板
    
    Examples
    --------
    >>> generator = DashboardGenerator()
    >>> dashboard_data = {...}
    >>> html_content = generator.generate_elastic_dashboard(dashboard_data)
    >>> generator.save_dashboard(html_content, 'output.html')
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化生成器
        
        Parameters
        ----------
        template_dir : str, optional
            自定义模板目录
        """
        if template_dir is None:
            # 使用内置模板目录
            template_dir = Path(__file__).parent / 'templates'
        
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,  # 自动转义HTML，提高安全性
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 添加自定义过滤器
        self._add_custom_filters()
        
        logger.info(f"仪表板生成器初始化完成，模板目录: {self.template_dir}")
    
    def _add_custom_filters(self):
        """添加自定义Jinja2过滤器"""
        
        def format_number(value, decimals=1):
            """格式化数字"""
            try:
                return f"{float(value):.{decimals}f}"
            except (ValueError, TypeError):
                return str(value)
        
        def format_percent(value, decimals=1, show_sign=True):
            """格式化百分比"""
            try:
                sign = '+' if show_sign and float(value) >= 0 else ''
                return f"{sign}{float(value):.{decimals}f}%"
            except (ValueError, TypeError):
                return str(value)
        
        def abs_filter(value):
            """绝对值过滤器"""
            try:
                return abs(float(value))
            except (ValueError, TypeError):
                return value
        
        def basename_filter(path):
            """获取文件名"""
            return Path(str(path)).name
        
        def embed_image(image_path):
            """将图片嵌入为base64"""
            try:
                image_file = Path(image_path)
                if image_file.exists():
                    mime_type, _ = mimetypes.guess_type(str(image_file))
                    if mime_type and mime_type.startswith('image/'):
                        with open(image_file, 'rb') as f:
                            encoded = base64.b64encode(f.read()).decode()
                        return f"data:{mime_type};base64,{encoded}"
            except Exception as e:
                logger.warning(f"无法嵌入图片 {image_path}: {e}")
            
            return str(image_path)  # 回退到原路径
        
        def json_safe(value):
            """安全的JSON序列化"""
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return '""'
        
        # 正确注册过滤器
        self.env.filters['format_number'] = format_number
        self.env.filters['format_percent'] = format_percent
        self.env.filters['abs'] = abs_filter
        self.env.filters['basename'] = basename_filter
        self.env.filters['embed_image'] = embed_image
        self.env.filters['json_safe'] = json_safe
    
    def generate_elastic_dashboard(
        self,
        dashboard_data: Dict[str, Any],
        template_name: str = 'elastic_dashboard.html',
        embed_images: bool = True
    ) -> str:
        """
        生成弹性常数仪表板
        
        Parameters
        ----------
        dashboard_data : Dict[str, Any]
            仪表板数据
        template_name : str
            模板文件名
        embed_images : bool
            是否将图片嵌入HTML（便于分享）
            
        Returns
        -------
        str
            生成的HTML内容
        """
        logger.info(f"生成弹性常数仪表板，模板: {template_name}")
        
        try:
            template = self.env.get_template(template_name)
        except Exception as e:
            logger.error(f"无法加载模板 {template_name}: {e}")
            # 使用内置简单模板作为后备
            return self._generate_fallback_dashboard(dashboard_data)
        
        # 预处理数据
        processed_data = self._preprocess_dashboard_data(dashboard_data, embed_images)
        
        try:
            html_content = template.render(**processed_data)
            logger.info("仪表板生成成功")
            return html_content
        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            return self._generate_fallback_dashboard(dashboard_data)
    
    def _preprocess_dashboard_data(
        self, 
        data: Dict[str, Any], 
        embed_images: bool = True
    ) -> Dict[str, Any]:
        """预处理仪表板数据"""
        processed = data.copy()
        
        # 确保必需的字段存在
        processed.setdefault('generation_time', datetime.now().isoformat())
        processed.setdefault('metadata', {})
        processed.setdefault('summary_statistics', {})
        processed.setdefault('analysis_results', {})
        processed.setdefault('plot_files', {})
        
        # 处理图片路径
        if embed_images and 'plot_files' in processed:
            embedded_plots = {}
            for plot_type, plot_path in processed['plot_files'].items():
                # 尝试嵌入图片
                embedded_path = self.env.filters['embed_image'](plot_path)
                embedded_plots[plot_type] = embedded_path
            processed['plot_files'] = embedded_plots
        
        # 添加辅助数据
        processed['current_year'] = datetime.now().year
        processed['total_plots'] = len(processed.get('plot_files', {}))
        
        # 计算额外的统计信息
        if 'analysis_results' in processed:
            results = processed['analysis_results']
            if results:
                # 计算平均误差
                errors = [abs(r.get('relative_error', 0)) for r in results.values()]
                processed['average_error'] = sum(errors) / len(errors) if errors else 0
                
                # 计算最佳和最差结果
                sorted_results = sorted(
                    results.items(), 
                    key=lambda x: abs(x[1].get('relative_error', float('inf')))
                )
                if sorted_results:
                    processed['best_result'] = sorted_results[0]
                    processed['worst_result'] = sorted_results[-1]
        
        # 添加标签页结构（解决模板继承中set变量不可见的问题）
        processed['tabs'] = self._create_tabs_structure(processed)
        
        return processed
    
    def _create_tabs_structure(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建标签页数据结构"""
        tabs = [
            {
                "id": "overview",
                "title": "概览", 
                "content": self._generate_overview_content(data)
            },
            {
                "id": "plots", 
                "title": "可视化图表",
                "content": self._generate_plots_content(data)
            },
            {
                "id": "details",
                "title": "详细数据", 
                "content": self._generate_details_content(data)
            },
            {
                "id": "trajectory", 
                "title": "轨迹动画",
                "content": self._generate_trajectory_content(data)
            }
        ]
        return tabs
    
    def _generate_overview_content(self, data: Dict[str, Any]) -> str:
        """生成概览内容"""
        # 使用简化的内联模板生成内容
        overview_template = """
<div class="section">
    <div class="section-title">汇总统计</div>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-title">弹性常数总数</div>
            <div class="stat-value">{{ summary_statistics.overall.total_constants | default(0) }}</div>
            <div class="stat-detail">
                数据点总数: {{ summary_statistics.overall.total_data_points | default(0) }}<br>
                平均收敛率: {{ "%.1f" | format(summary_statistics.overall.average_convergence_rate * 100) | default("N/A") }}%
            </div>
        </div>
        
        {% for const_type, const_stats in summary_statistics.elastic_constants.items() %}
        <div class="stat-card elastic-constant-card" data-constant="{{ const_type }}">
            <div class="stat-title">{{ const_type }}</div>
            <div class="stat-value">{{ "%.1f" | format(const_stats.value_GPa) }}</div>
            <div class="stat-detail">
                文献值: {{ "%.1f" | format(const_stats.literature_GPa) }} GPa<br>
                <span class="{% if const_stats.error_percent|abs <= 10 %}success{% elif const_stats.error_percent|abs <= 25 %}warning{% else %}error{% endif %}">
                    误差: {{ "%+.1f" | format(const_stats.error_percent) }}%
                </span>
            </div>
        </div>
        {% endfor %}
    </div>
</div>

<div class="section">
    <h3>计算结果概览</h3>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>弹性常数</th>
                    <th>计算值 (GPa)</th>
                    <th>文献值 (GPa)</th>
                    <th>相对误差</th>
                    <th>R²</th>
                </tr>
            </thead>
            <tbody>
                {% for const_type, result in analysis_results.items() %}
                <tr>
                    <td><strong>{{ const_type }}</strong></td>
                    <td>{{ "%.1f" | format(result.elastic_constant) }}</td>
                    <td>{{ "%.1f" | format(result.literature_value) }}</td>
                    <td class="{% if result.relative_error|abs <= 10 %}success{% elif result.relative_error|abs <= 25 %}warning{% else %}error{% endif %}">
                        {{ "%+.1f" | format(result.relative_error) }}%
                    </td>
                    <td>{{ "%.4f" | format(result.r_squared) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
        """
        template = Template(overview_template)
        return template.render(**data)
    
    def _generate_plots_content(self, data: Dict[str, Any]) -> str:
        """生成可视化图表内容"""
        plots_template = """
<div class="section">
    <div class="section-title">可视化图表</div>
    
    {% if plot_files %}
        {% for plot_type, plot_path in plot_files.items() %}
        <div class="plot-container">
            <h3>{{ plot_type.replace('_', ' ').title() }}</h3>
            <img src="{{ plot_path }}" 
                 alt="{{ plot_type.replace('_', ' ').title() }}" 
                 class="plot-image"
                 onclick="togglePlotFullscreen(this)"
                 style="cursor: pointer;"
                 title="点击查看全屏">
        </div>
        {% endfor %}
    {% else %}
        <div class="alert alert-info">
            <h4>暂无可视化图表</h4>
            <p>图表正在生成中，或者数据不足以生成可视化内容。</p>
        </div>
    {% endif %}
</div>
        """
        template = Template(plots_template)
        return template.render(**data)
    
    def _generate_details_content(self, data: Dict[str, Any]) -> str:
        """生成详细数据内容"""
        details_template = """
<div class="section">
    <div class="section-title">详细分析</div>
    
    {% for const_type, result in analysis_results.items() %}
    <div class="section">
        <h3>{{ const_type }} 详细分析</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">弹性常数值</div>
                <div class="stat-value">{{ "%.2f" | format(result.elastic_constant) }}</div>
                <div class="stat-detail">GPa</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">拟合质量</div>
                <div class="stat-value">{{ "%.4f" | format(result.r_squared) }}</div>
                <div class="stat-detail">R² 决定系数</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">相对误差</div>
                <div class="stat-value {% if result.relative_error|abs <= 10 %}success{% elif result.relative_error|abs <= 25 %}warning{% else %}error{% endif %}">
                    {{ "%+.1f" | format(result.relative_error) }}%
                </div>
                <div class="stat-detail">
                    文献值: {{ "%.1f" | format(result.literature_value) }} GPa
                </div>
            </div>
        </div>
    </div>
    {% endfor %}
</div>
        """
        template = Template(details_template)
        return template.render(**data)
    
    def _generate_trajectory_content(self, data: Dict[str, Any]) -> str:
        """生成轨迹动画内容"""
        trajectory_template = """
<div class="section">
    <div class="section-title">轨迹动画</div>
    
    {% if metadata.has_trajectory %}
    <div class="trajectory-preview">
        <h4>轨迹数据概览</h4>
        <p>该计算包含完整的形变过程轨迹数据。</p>
        <div class="alert alert-info">
            <h5>功能开发中</h5>
            <p>轨迹动画功能正在开发中，当前版本支持基础可视化。</p>
        </div>
    </div>
    {% else %}
    <div class="alert alert-warning">
        <h4>无轨迹数据</h4>
        <p>当前数据集不包含轨迹信息。</p>
    </div>
    {% endif %}
</div>
        """
        template = Template(trajectory_template)
        return template.render(**data)
    
    def _generate_fallback_dashboard(self, data: Dict[str, Any]) -> str:
        """生成后备仪表板（当模板加载失败时）"""
        logger.warning("使用后备仪表板模板")
        
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>弹性常数分析仪表板</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; padding: 20px; background: #007bff; color: white; margin: -20px -20px 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .stat-title { font-weight: bold; color: #666; margin-bottom: 10px; }
        .stat-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .error { color: #dc3545; } .warning { color: #ffc107; } .success { color: #28a745; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        .plot { text-align: center; margin: 20px 0; }
        .plot img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>弹性常数分析仪表板</h1>
            <p>生成时间: {{ generation_time }}</p>
        </div>
        
        <div class="stats">
            {% for const_type, result in analysis_results.items() %}
            <div class="stat-card">
                <div class="stat-title">{{ const_type }}</div>
                <div class="stat-value">{{ "%.1f"|format(result.elastic_constant) }} GPa</div>
                <div class="{% if result.relative_error|abs <= 10 %}success{% elif result.relative_error|abs <= 25 %}warning{% else %}error{% endif %}">
                    误差: {{ "%+.1f"|format(result.relative_error) }}%
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>详细结果</h2>
        <table>
            <thead>
                <tr>
                    <th>弹性常数</th>
                    <th>计算值 (GPa)</th>
                    <th>文献值 (GPa)</th>
                    <th>相对误差</th>
                    <th>R²</th>
                </tr>
            </thead>
            <tbody>
                {% for const_type, result in analysis_results.items() %}
                <tr>
                    <td>{{ const_type }}</td>
                    <td>{{ "%.1f"|format(result.elastic_constant) }}</td>
                    <td>{{ "%.1f"|format(result.literature_value) }}</td>
                    <td class="{% if result.relative_error|abs <= 10 %}success{% elif result.relative_error|abs <= 25 %}warning{% else %}error{% endif %}">
                        {{ "%+.1f"|format(result.relative_error) }}%
                    </td>
                    <td>{{ "%.4f"|format(result.r_squared) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if plot_files %}
        <h2>可视化图表</h2>
        {% for plot_type, plot_path in plot_files.items() %}
        <div class="plot">
            <h3>{{ plot_type.replace('_', ' ').title() }}</h3>
            <img src="{{ plot_path }}" alt="{{ plot_type }}">
        </div>
        {% endfor %}
        {% endif %}
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(**data)
    
    def save_dashboard(
        self, 
        html_content: str, 
        output_path: str
    ) -> str:
        """
        保存仪表板到文件
        
        Parameters
        ----------
        html_content : str
            HTML内容
        output_path : str
            输出文件路径
            
        Returns
        -------
        str
            实际保存的文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"仪表板已保存: {output_file}")
        return str(output_file)
    
    def generate_and_save(
        self,
        dashboard_data: Dict[str, Any],
        output_path: str,
        template_name: str = 'elastic_dashboard.html',
        embed_images: bool = True
    ) -> str:
        """
        生成并保存仪表板（一步完成）
        
        Parameters
        ----------
        dashboard_data : Dict[str, Any]
            仪表板数据
        output_path : str
            输出文件路径
        template_name : str
            模板文件名
        embed_images : bool
            是否嵌入图片
            
        Returns
        -------
        str
            保存的文件路径
        """
        html_content = self.generate_elastic_dashboard(
            dashboard_data, template_name, embed_images
        )
        return self.save_dashboard(html_content, output_path)
    
    def create_multi_page_dashboard(
        self,
        datasets: List[Dict[str, Any]],
        output_dir: str,
        index_title: str = "弹性常数分析汇总"
    ) -> Dict[str, str]:
        """
        创建多页面仪表板（适用于多个数据集的比较）
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            数据集列表
        output_dir : str
            输出目录
        index_title : str
            索引页标题
            
        Returns
        -------
        Dict[str, str]
            生成的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # 为每个数据集生成单独页面
        for i, dataset in enumerate(datasets):
            dataset_name = dataset.get('metadata', {}).get('source_file', f'dataset_{i+1}')
            filename = f"dashboard_{i+1}.html"
            file_path = self.generate_and_save(
                dataset, 
                str(output_path / filename),
                embed_images=True
            )
            files[dataset_name] = file_path
        
        # 生成索引页
        index_data = {
            'generation_time': datetime.now().isoformat(),
            'title': index_title,
            'datasets': [
                {
                    'name': name,
                    'file': Path(path).name,
                    'summary': self._extract_dataset_summary(datasets[i])
                }
                for i, (name, path) in enumerate(files.items())
            ]
        }
        
        index_html = self._generate_index_page(index_data)
        index_path = self.save_dashboard(index_html, str(output_path / "index.html"))
        files['index'] = index_path
        
        logger.info(f"多页面仪表板生成完成: {len(files)}个文件")
        return files
    
    def _extract_dataset_summary(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """提取数据集摘要信息"""
        summary = {}
        
        if 'analysis_results' in dataset:
            results = dataset['analysis_results']
            summary['constants'] = list(results.keys())
            summary['total_constants'] = len(results)
            
            if results:
                errors = [abs(r.get('relative_error', 0)) for r in results.values()]
                summary['average_error'] = sum(errors) / len(errors)
                summary['best_error'] = min(errors)
                summary['worst_error'] = max(errors)
        
        if 'metadata' in dataset:
            metadata = dataset['metadata']
            summary['data_points'] = metadata.get('data_points', 0)
            summary['load_time'] = metadata.get('load_time', '')
        
        return summary
    
    def _generate_index_page(self, index_data: Dict[str, Any]) -> str:
        """生成索引页HTML"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; background: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header h1 { font-size: 2.5rem; color: #343a40; margin-bottom: 10px; }
        .datasets { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        .dataset-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .dataset-card:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
        .dataset-title { font-size: 1.2rem; font-weight: 600; color: #007bff; margin-bottom: 15px; }
        .dataset-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #495057; }
        .stat-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; }
        .view-button { display: block; width: 100%; padding: 10px; background: #007bff; color: white; text-decoration: none; text-align: center; border-radius: 6px; font-weight: 600; }
        .view-button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>生成时间: {{ generation_time }}</p>
            <p>共 {{ datasets|length }} 个数据集</p>
        </div>
        
        <div class="datasets">
            {% for dataset in datasets %}
            <div class="dataset-card">
                <div class="dataset-title">{{ dataset.name }}</div>
                <div class="dataset-stats">
                    <div class="stat">
                        <div class="stat-value">{{ dataset.summary.get('total_constants', 0) }}</div>
                        <div class="stat-label">弹性常数</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{{ "%.1f"|format(dataset.summary.get('average_error', 0)) }}%</div>
                        <div class="stat-label">平均误差</div>
                    </div>
                </div>
                <a href="{{ dataset.file }}" class="view-button">查看详细分析</a>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(**index_data)


# 使用示例和工厂函数
def create_dashboard_generator(template_dir: Optional[str] = None) -> DashboardGenerator:
    """
    创建仪表板生成器的工厂函数
    
    Parameters
    ----------
    template_dir : str, optional
        自定义模板目录
        
    Returns
    -------
    DashboardGenerator
        配置好的生成器实例
    """
    return DashboardGenerator(template_dir)


def quick_dashboard(
    dashboard_data: Dict[str, Any],
    output_path: str,
    embed_images: bool = True
) -> str:
    """
    快速生成仪表板的便捷函数
    
    Parameters
    ----------
    dashboard_data : Dict[str, Any]
        仪表板数据
    output_path : str
        输出文件路径
    embed_images : bool
        是否嵌入图片
        
    Returns
    -------
    str
        生成的文件路径
    """
    generator = DashboardGenerator()
    return generator.generate_and_save(dashboard_data, output_path, embed_images=embed_images)