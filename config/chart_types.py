"""
图表类型枚举和配置
定义系统支持的所有图表类型及其属性
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class ChartType(Enum):
    """图表类型枚举"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    PARALLEL_COORDINATES = "parallel_coordinates"
    HEATMAP = "heatmap"
    SANKEY_DIAGRAM = "sankey_diagram"
    UNKNOWN = "unknown"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """从字符串转换为枚举类型"""
        chart_type_lower = chart_type.lower().replace(' ', '_').replace('-', '_')
        for ct in cls:
            if ct.value == chart_type_lower:
                return ct
        return cls.UNKNOWN


@dataclass
class ChartTypeConfig:
    """图表类型配置"""
    name: str
    display_name: str
    description: str
    typical_marks: List[str]  # Vega-Lite 中典型的 mark 类型
    typical_encodings: List[str]  # 典型的编码通道
    supported_interactions: List[str]  # 支持的交互类型
    prompt_file: str  # 对应的提示词文件名
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'typical_marks': self.typical_marks,
            'typical_encodings': self.typical_encodings,
            'supported_interactions': self.supported_interactions,
            'prompt_file': self.prompt_file
        }


# 图表类型详细配置
CHART_TYPE_CONFIGS = {
    ChartType.BAR_CHART: ChartTypeConfig(
        name="bar_chart",
        display_name="条形图/柱状图",
        description="用于比较不同类别的数值",
        typical_marks=["bar"],
        typical_encodings=["x", "y", "color"],
        supported_interactions=["sort", "filter", "highlight", "compare"],
        prompt_file="bar_chart.txt"
    ),
    
    ChartType.LINE_CHART: ChartTypeConfig(
        name="line_chart",
        display_name="折线图",
        description="用于展示趋势和时间序列数据",
        typical_marks=["line", "point"],
        typical_encodings=["x", "y", "color"],
        supported_interactions=["zoom", "time_range", "trend_detection", "compare"],
        prompt_file="line_chart.txt"
    ),
    
    ChartType.SCATTER_PLOT: ChartTypeConfig(
        name="scatter_plot",
        display_name="散点图",
        description="用于展示两个变量之间的关系和分布",
        typical_marks=["point", "circle"],
        typical_encodings=["x", "y", "color", "size"],
        supported_interactions=["select", "cluster", "correlation", "zoom"],
        prompt_file="scatter_plot.txt"
    ),
    
    ChartType.PARALLEL_COORDINATES: ChartTypeConfig(
        name="parallel_coordinates",
        display_name="平行坐标图",
        description="用于多维数据的可视化和模式识别",
        typical_marks=["line"],
        typical_encodings=["x", "y", "color"],
        supported_interactions=["reorder", "brush", "highlight"],
        prompt_file="parallel_coordinates.txt"
    ),
    
    ChartType.HEATMAP: ChartTypeConfig(
        name="heatmap",
        display_name="热力图",
        description="用于展示矩阵数据和相关性",
        typical_marks=["rect"],
        typical_encodings=["x", "y", "color"],
        supported_interactions=["color_scale", "select_submatrix", "cluster"],
        prompt_file="heatmap.txt"
    ),
    
    ChartType.SANKEY_DIAGRAM: ChartTypeConfig(
        name="sankey_diagram",
        display_name="桑基图",
        description="用于展示流量和转换关系",
        typical_marks=["rect", "path"],
        typical_encodings=["x", "y", "color", "opacity"],
        supported_interactions=["filter_flow", "highlight_path", "trace_node"],
        prompt_file="sankey_diagram.txt"
    )
}


def get_chart_config(chart_type: ChartType) -> ChartTypeConfig:
    """获取图表类型配置"""
    return CHART_TYPE_CONFIGS.get(chart_type, None)


def get_all_chart_types() -> List[ChartType]:
    """获取所有支持的图表类型"""
    return [ct for ct in ChartType if ct != ChartType.UNKNOWN]


def get_chart_type_by_mark(mark: str) -> ChartType:
    """通过 Vega-Lite mark 类型推断图表类型"""
    mark = mark.lower()
    for chart_type, config in CHART_TYPE_CONFIGS.items():
        if mark in config.typical_marks:
            return chart_type
    return ChartType.UNKNOWN


def get_supported_interactions(chart_type: ChartType) -> List[str]:
    """获取图表支持的交互类型"""
    config = get_chart_config(chart_type)
    return config.supported_interactions if config else []


def get_candidate_chart_types(vega_spec: Dict[str, Any]) -> List[ChartType]:
    """
    从Vega-Lite规范中推断可能的图表类型
    
    Args:
        vega_spec: Vega-Lite JSON规范
    
    Returns:
        可能的图表类型列表，按可能性排序
    """
    candidates = []
    
    # 获取 mark 类型
    mark = vega_spec.get('mark')
    if isinstance(mark, dict):
        mark_type = mark.get('type', '')
    else:
        mark_type = mark or ''
    
    mark_type = str(mark_type).lower()
    
    # 获取 encoding 信息
    encoding = vega_spec.get('encoding', {})
    x_encoding = encoding.get('x', {})
    y_encoding = encoding.get('y', {})
    x_type = x_encoding.get('type', '')
    y_type = y_encoding.get('type', '')
    
    # 特殊处理：point mark 可能是折线图或散点图
    if mark_type == 'point':
        # 如果 X 轴是时间类型，更可能是折线图
        if x_type == 'temporal':
            candidates.append(ChartType.LINE_CHART)
        # 如果 X 和 Y 都是数值类型，更可能是散点图
        elif x_type == 'quantitative' and y_type == 'quantitative':
            candidates.append(ChartType.SCATTER_PLOT)
        # 如果有 shape 或 size encoding，更可能是散点图
        elif 'shape' in encoding or 'size' in encoding:
            candidates.append(ChartType.SCATTER_PLOT)
        else:
            # 无法确定，添加两者但散点图优先
            candidates.append(ChartType.SCATTER_PLOT)
            candidates.append(ChartType.LINE_CHART)
    
    # 处理其他 mark 类型
    elif mark_type:
        chart_type = get_chart_type_by_mark(mark_type)
        if chart_type != ChartType.UNKNOWN:
            candidates.append(chart_type)
    
    # 如果还没有找到候选，尝试根据编码通道推断
    if not candidates:
        # 检查是否有特定的编码模式
        if 'x' in encoding and 'y' in encoding:
            # 检查是否有时间字段
            if x_type == 'temporal' or y_type == 'temporal':
                candidates.append(ChartType.LINE_CHART)
            
            # 检查是否有分类和数值字段
            if x_type == 'nominal' and y_type == 'quantitative':
                candidates.append(ChartType.BAR_CHART)
            elif x_type == 'quantitative' and y_type == 'quantitative':
                candidates.append(ChartType.SCATTER_PLOT)
        
        # 检查是否有颜色编码且为数值类型（可能是热力图）
        if 'color' in encoding:
            color_type = encoding.get('color', {}).get('type', '')
            if color_type == 'quantitative':
                candidates.append(ChartType.HEATMAP)
    
    # 如果仍然没有候选，返回UNKNOWN
    if not candidates:
        candidates.append(ChartType.UNKNOWN)
    
    return candidates
