"""
工具注册器（简化版 - 使用 vega_spec 而非 view_id）
"""

from typing import Dict, List, Callable, Any
from config.chart_types import ChartType

from . import common
from . import bar_chart_tools
from . import line_chart_tools
from . import scatter_plot_tools
from . import parallel_coordinates_tools
from . import heatmap_tools
from . import sankey_tools


class ToolRegistry:
    """工具注册表类"""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._chart_tools: Dict[ChartType, List[str]] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """注册所有工具"""
        
        # 通用工具（保留感知类和基础操作）
        common_tools = {
            'get_view_spec': {
                'function': common.get_view_spec,
                'category': 'perception',
                'description': '返回当前视图的结构化状态（encoding, domain, transforms, selections等）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'get_data': {
                'function': common.get_data,
                'category': 'perception',
                'description': '返回原始数据',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'scope': {'type': 'str', 'required': False, 'default': 'all', 'description': 'all | filter | visible | selected'}
                }
            },
            'get_data_summary': {
                'function': common.get_data_summary,
                'category': 'perception',
                'description': '获取数据统计摘要',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'scope': {'type': 'str', 'required': False, 'default': 'all'}
                }
            },
            'get_tooltip_data': {
                'function': common.get_tooltip_data,
                'category': 'perception',
                'description': '获取工具提示数据（鼠标悬浮）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'position': {'type': 'list', 'required': True, 'description': '数据坐标 [x, y]'}
                }
            },
            'reset_view': {
                'function': common.reset_view,
                'category': 'action',
                'description': '重置视图到原始状态（从 vega_spec._original_spec 元数据读取）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'undo_view': {
                'function': common.undo_view,
                'category': 'action',
                'description': '撤销上一步视图，返回上一版本（从 vega_spec._spec_history 元数据读取）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'render_chart': {
                'function': common.render_chart,
                'category': 'action',
                'description': '渲染图表',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            }
        }
        
        # 柱状图工具
        bar_chart_tools_dict = {
            'sort_bars': {
                'function': bar_chart_tools.sort_bars,
                'category': 'action',
                'description': '按值排序条形。堆叠图：传 by_subcategory 则按该子类值排 x 轴；不传则按子类总值排堆叠层顺序。分组/简单图：按 y 总值排 x 轴。',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'order': {'type': 'str', 'required': False, 'default': 'descending', 'description': 'ascending 或 descending'},
                    'by_subcategory': {'type': 'str', 'required': False, 'description': '堆叠图专用：指定子类名（如 Diesel），则按该子类的值排序 x 轴；不传则排序堆叠层'}
                }
            },
            'filter_categories': {
                'function': bar_chart_tools.filter_categories,
                'category': 'action',
                'description': '筛选特定类别',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'categories': {'type': 'list', 'required': True}
                }
            },
            'filter_subcategories': {
                'function': bar_chart_tools.filter_subcategories,
                'category': 'action',
                'description': '过滤子类别（color/xOffset 编码的类别，如堆叠层或分组）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'subcategories_to_remove': {'type': 'list', 'required': True, 'description': '要移除的子类别列表'},
                    'sub_field': {'type': 'str', 'required': False, 'description': '子分类字段名（可选，自动探测）'}
                }
            },
            'highlight_top_n': {
                'function': bar_chart_tools.highlight_top_n,
                'category': 'action',
                'description': '高亮前N个条形',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'n': {'type': 'int', 'required': False, 'default': 2},
                    'order': {'type': 'str', 'required': False, 'default': 'descending'}
                }
            },
            'expand_stack': {
                'function': bar_chart_tools.expand_stack,
                'category': 'action',
                'description': '展开堆叠条形图中某个类别的堆叠部分为平排条形图（交互必要性：解决堆叠图中间层基线不同难以比较的问题）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'category': {'type': 'str', 'required': True, 'description': '要展开的 x 轴类别名（如 "华东"）'}
                }
            },
            'toggle_stack_mode': {
                'function': bar_chart_tools.toggle_stack_mode,
                'category': 'action',
                'description': '全局切换堆叠/分组显示模式',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'mode': {'type': 'str', 'required': True, 'description': '"grouped"（分组并排）或 "stacked"（堆叠）'}
                }
            },
            'add_bars': {
                'function': bar_chart_tools.add_bars,
                'category': 'action',
                'description': '增加整根条形：按 x 轴类别增量显示（堆叠/分组均适用；可从 full_data_path 补全数据）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'values': {'type': 'list', 'required': True, 'description': '要新增显示的 x 类别值列表，如 ["华东"]'},
                    'x_field': {'type': 'str', 'required': False, 'description': 'x 类别字段名（可选，自动探测）'}
                }
            },
            'remove_bars': {
                'function': bar_chart_tools.remove_bars,
                'category': 'action',
                'description': '减少整根条形：按 x 轴类别增量隐藏（堆叠/分组均适用）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'values': {'type': 'list', 'required': True, 'description': '要隐藏的 x 类别值列表，如 ["华东"]'},
                    'x_field': {'type': 'str', 'required': False, 'description': 'x 类别字段名（可选，自动探测）'}
                }
            },
            'add_bar_items': {
                'function': bar_chart_tools.add_bar_items,
                'category': 'action',
                'description': '增加单根子柱：按 (x, 子分组) 增量显示（堆叠=颜色层，分组=xOffset）并可从 full_data_path 补全数据',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'items': {'type': 'list', 'required': True, 'description': '要新增的子柱列表，如 [{"x":"华东","sub":"电子产品"}]'},
                    'x_field': {'type': 'str', 'required': False, 'description': 'x 字段名（可选，自动探测）'},
                    'sub_field': {'type': 'str', 'required': False, 'description': '子分组字段名（可选，优先 xOffset.field，其次 color.field）'}
                }
            },
            'remove_bar_items': {
                'function': bar_chart_tools.remove_bar_items,
                'category': 'action',
                'description': '减少单根子柱：按 (x, 子分组) 增量隐藏（堆叠=颜色层，分组=xOffset）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'items': {'type': 'list', 'required': True, 'description': '要隐藏的子柱列表，如 [{"x":"华东","sub":"电子产品"}]'},
                    'x_field': {'type': 'str', 'required': False, 'description': 'x 字段名（可选，自动探测）'},
                    'sub_field': {'type': 'str', 'required': False, 'description': '子分组字段名（可选，优先 xOffset.field，其次 color.field）'}
                }
            },
            'change_encoding': {
                'function': bar_chart_tools.change_encoding,
                'category': 'action',
                'description': '修改指定编码通道的字段映射 (color, size, shape, opacity, x, y 等)',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'channel': {'type': 'str', 'required': True, 'description': '编码通道 (color, size, shape, opacity, x, y 等)'},
                    'field': {'type': 'str', 'required': True, 'description': '新的字段名'}
                }
            }
        }
        
        # 折线图工具
        line_chart_tools_dict = {
            'zoom_time_range': {
                'function': line_chart_tools.zoom_time_range,
                'category': 'action',
                'description': '缩放时间范围',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'start': {'type': 'str', 'required': True},
                    'end': {'type': 'str', 'required': True}
                }
            },
            'highlight_trend': {
                'function': line_chart_tools.highlight_trend,
                'category': 'action',
                'description': '高亮趋势',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'trend_type': {'type': 'str', 'required': False, 'default': 'increasing'}
                }
            },
            'detect_anomalies': {
                'function': line_chart_tools.detect_anomalies,
                'category': 'analysis',
                'description': '检测异常点',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'threshold': {'type': 'float', 'required': False, 'default': 2.0}
                }
            },
            'bold_lines': {
                'function': line_chart_tools.bold_lines,
                'category': 'action',
                'description': '加粗指定的折线',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'line_names': {'type': 'list', 'required': True, 'description': '要加粗的折线名称列表'},
                    'line_field': {'type': 'str', 'required': False, 'description': '折线分组字段名（可选，自动探测）'}
                }
            },
            'filter_lines': {
                'function': line_chart_tools.filter_lines,
                'category': 'action',
                'description': '过滤掉指定的折线',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'lines_to_remove': {'type': 'list', 'required': True, 'description': '要移除的折线名称列表'},
                    'line_field': {'type': 'str', 'required': False, 'description': '折线分组字段名（可选，自动探测）'}
                }
            },
            'show_moving_average': {
                'function': line_chart_tools.show_moving_average,
                'category': 'analysis',
                'description': '叠加移动平均线',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'window_size': {'type': 'int', 'required': False, 'default': 3, 'description': '移动平均窗口大小'}
                }
            },
            'focus_lines': {
                'function': line_chart_tools.focus_lines,
                'category': 'action',
                'description': '聚焦指定折线，其余变暗或隐藏（认知降噪）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'lines': {'type': 'list', 'required': True, 'description': '需要聚焦的折线名称列表'},
                    'line_field': {'type': 'str', 'required': False, 'description': '折线分组字段名（可选，自动探测）'},
                    'mode': {'type': 'str', 'required': False, 'default': 'dim', 'description': 'dim | hide'},
                    'dim_opacity': {'type': 'float', 'required': False, 'default': 0.08, 'description': '非聚焦折线透明度（mode=dim）'}
                }
            },
            'drilldown_line_time': {
                'function': line_chart_tools.drilldown_line_time,
                'category': 'action',
                'description': '时间下钻（年→月→日）：交互必要性工具，逐层深入发现更细粒度的模式',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'level': {'type': 'str', 'required': True, 'description': '"year" | "month" | "date"'},
                    'value': {'type': 'int', 'required': True, 'description': '年份(2020-2030) | 月份(1-12) | 日期(1-31)，必须是整数'},
                    'parent': {'type': 'dict', 'required': False, 'description': '父级信息，如 {"year": 2023} 或 {"year": 2023, "month": 3}'}
                }
            },
            'reset_line_drilldown': {
                'function': line_chart_tools.reset_line_drilldown,
                'category': 'action',
                'description': '重置折线图时间下钻，恢复到初始年度视图',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'resample_time': {
                'function': line_chart_tools.resample_time,
                'category': 'action',
                'description': '时间粒度切换（重采样）：将时间序列从细粒度聚合到粗粒度（日→周→月→季度→年）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'granularity': {'type': 'str', 'required': True, 'description': '"day" | "week" | "month" | "quarter" | "year"'},
                    'agg': {'type': 'str', 'required': False, 'default': 'mean', 'description': '"mean" | "sum" | "max" | "min" | "median"'}
                }
            },
            'reset_resample': {
                'function': line_chart_tools.reset_resample,
                'category': 'action',
                'description': '重置时间重采样，恢复到原始粒度',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'change_encoding': {
                'function': line_chart_tools.change_encoding,
                'category': 'action',
                'description': '修改指定编码通道的字段映射 (color, size, shape, opacity, x, y 等)',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'channel': {'type': 'str', 'required': True, 'description': '编码通道 (color, size, shape, opacity, x, y 等)'},
                    'field': {'type': 'str', 'required': True, 'description': '新的字段名'}
                }
            }
        }
        
        # 散点图工具
        scatter_tools = {
            'identify_clusters': {
                'function': scatter_plot_tools.identify_clusters,
                'category': 'analysis',
                'description': '识别聚类',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'n_clusters': {'type': 'int', 'required': False, 'default': 3},
                    'method': {'type': 'str', 'required': False, 'default': 'kmeans'}
                }
            },
            'calculate_correlation': {
                'function': scatter_plot_tools.calculate_correlation,
                'category': 'analysis',
                'description': '计算相关性',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'method': {'type': 'str', 'required': False, 'default': 'pearson'}
                }
            },
            'zoom_dense_area': {
                'function': scatter_plot_tools.zoom_dense_area,
                'category': 'action',
                'description': '放大密集区域',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_range': {'type': 'tuple', 'required': True},
                    'y_range': {'type': 'tuple', 'required': True}
                }
            },
            'select_region': {
                'function': scatter_plot_tools.select_region,
                'category': 'action',
                'description': '选择区域',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_range': {'type': 'tuple', 'required': True},
                    'y_range': {'type': 'tuple', 'required': True}
                }
            },
            'filter_categorical': {
                'function': scatter_plot_tools.filter_categorical,
                'category': 'action',
                'description': '过滤掉指定类别的数据点',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'categories_to_remove': {'type': 'list', 'required': True, 'description': '要移除的类别列表'},
                    'field': {'type': 'str', 'required': False, 'description': '分类字段名（可选，自动探测）'}
                }
            },
            'brush_region': {
                'function': scatter_plot_tools.brush_region,
                'category': 'action',
                'description': '刷选特定区域，区域外数据点变淡',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_range': {'type': 'tuple', 'required': True, 'description': 'X轴范围 (min, max)'},
                    'y_range': {'type': 'tuple', 'required': True, 'description': 'Y轴范围 (min, max)'}
                }
            },
            'show_regression': {
                'function': scatter_plot_tools.show_regression,
                'category': 'analysis',
                'description': '叠加回归线',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'method': {'type': 'str', 'required': False, 'default': 'linear', 'description': '回归方法 (linear, log, exp, poly, quad)'}
                }
            },
            'change_encoding': {
                'function': scatter_plot_tools.change_encoding,
                'category': 'action',
                'description': '修改指定编码通道的字段映射 (color, size, shape, opacity, x, y 等)',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'channel': {'type': 'str', 'required': True, 'description': '编码通道 (color, size, shape, opacity, x, y 等)'},
                    'field': {'type': 'str', 'required': True, 'description': '新的字段名'}
                }
            }
        }
        
        # 热力图工具
        heatmap_tools_dict = {
            'adjust_color_scale': {
                'function': heatmap_tools.adjust_color_scale,
                'category': 'action',
                'description': '调整颜色比例和范围',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'scheme': {'type': 'str', 'required': False, 'default': 'viridis', 'description': '颜色方案'},
                    'domain': {'type': 'list', 'required': False, 'description': '数值范围 [min, max]'}
                }
            },
            'filter_cells': {
                'function': heatmap_tools.filter_cells,
                'category': 'action',
                'description': '按 color 字段数值筛选单元格。min_value 或 max_value 至少提供一个（单边区间即可）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'min_value': {'type': 'float', 'required': False, 'description': '下界（含），只传此项则保留 >= min_value'},
                    'max_value': {'type': 'float', 'required': False, 'description': '上界（含），只传此项则保留 <= max_value'}
                }
            },
            'filter_cells_by_region': {
                'function': heatmap_tools.filter_cells_by_region,
                'category': 'action',
                'description': '按格子坐标 (x,y) 过滤掉聚合后的热力图格子（通过 transform.filter 排除该坐标对应的格子）。x 或 y 至少提供一个。',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_value': {'type': 'any', 'required': False, 'description': '要过滤的格子 x 轴值（单值）；x_values 为多值'},
                    'y_value': {'type': 'any', 'required': False, 'description': '要过滤的格子 y 轴值（单值）；y_values 为多值'},
                    'x_values': {'type': 'list', 'required': False, 'description': '要过滤的 x 轴值列表'},
                    'y_values': {'type': 'list', 'required': False, 'description': '要过滤的 y 轴值列表'}
                }
            },
            'highlight_region': {
                'function': heatmap_tools.highlight_region,
                'category': 'action',
                'description': '高亮区域。x_values 或 y_values 至少提供一个（仅 x 高亮整列，仅 y 高亮整行，两者都提供则高亮交叉区域）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_values': {'type': 'list', 'required': False, 'description': 'X 轴要高亮的值列表'},
                    'y_values': {'type': 'list', 'required': False, 'description': 'Y 轴要高亮的值列表'}
                }
            },
            'highlight_region_by_value': {
                'function': heatmap_tools.highlight_region_by_value,
                'category': 'action',
                'description': '按格子显示值高亮：范围内保持不变，范围外变淡（不删除数据；支持单侧阈值）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'min_value': {'type': 'float', 'required': False, 'description': '下阈值（可选）'},
                    'max_value': {'type': 'float', 'required': False, 'description': '上阈值（可选）'},
                    'outside_opacity': {'type': 'float', 'required': False, 'default': 0.12, 'description': '范围外透明度'}
                }
            },
            'cluster_rows_cols': {
                'function': heatmap_tools.cluster_rows_cols,
                'category': 'action',
                'description': '对行列进行聚类排序',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'cluster_rows': {'type': 'bool', 'required': False, 'default': True},
                    'cluster_cols': {'type': 'bool', 'required': False, 'default': True},
                    'method': {'type': 'str', 'required': False, 'default': 'sum'}
                }
            },
            'select_submatrix': {
                'function': heatmap_tools.select_submatrix,
                'category': 'action',
                'description': '选择子矩阵',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'x_values': {'type': 'list', 'required': False},
                    'y_values': {'type': 'list', 'required': False}
                }
            },
            'find_extremes': {
                'function': heatmap_tools.find_extremes,
                'category': 'analysis',
                'description': '标记极值点位置',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'top_n': {'type': 'int', 'required': False, 'default': 5, 'description': '标记前N个极值'},
                    'mode': {'type': 'str', 'required': False, 'default': 'both', 'description': 'max | min | both'}
                }
            },
            'threshold_mask': {
                'function': heatmap_tools.threshold_mask,
                'category': 'action',
                'description': '阈值遮罩：范围外单元格变淡（不删除数据）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'min_value': {'type': 'float', 'required': True, 'description': '下阈值（包含）'},
                    'max_value': {'type': 'float', 'required': True, 'description': '上阈值（包含）'},
                    'outside_opacity': {'type': 'float', 'required': False, 'default': 0.1, 'description': '范围外透明度'}
                }
            },
            'drilldown_time': {
                'function': heatmap_tools.drilldown_time,
                'category': 'action',
                'description': '时间热力图下钻：年→月→日',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'level': {'type': 'str', 'required': True, 'description': 'year | month | date'},
                    'value': {'type': 'any', 'required': True, 'description': '对应 level 的值（year=int, month=1-12, date=1-31）'},
                    'parent': {'type': 'dict', 'required': False, 'description': '父级信息，如 {\"year\":2012} 或 {\"year\":2012,\"month\":3}'}
                }
            },
            'reset_drilldown': {
                'function': heatmap_tools.reset_drilldown,
                'category': 'action',
                'description': '重置时间热力图下钻，恢复到初始粒度',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'add_marginal_bars': {
                'function': heatmap_tools.add_marginal_bars,
                'category': 'action',
                'description': '为热力图添加边际条形图（行/列聚合，默认均值 mean），用于快速判断哪一行/列总体更高',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'op': {'type': 'str', 'required': False, 'default': 'mean', 'description': '聚合口径：mean（默认）| sum | median | max | min | count'},
                    'show_top': {'type': 'bool', 'required': False, 'default': True, 'description': '是否显示顶部（按列/x 聚合）边际条'},
                    'show_right': {'type': 'bool', 'required': False, 'default': True, 'description': '是否显示右侧（按行/y 聚合）边际条'},
                    'bar_size': {'type': 'int', 'required': False, 'default': 70, 'description': '边际条形图厚度（像素）'},
                    'bar_color': {'type': 'str', 'required': False, 'default': '#666666', 'description': '边际条颜色'}
                }
            },
            'transpose': {
                'function': heatmap_tools.transpose,
                'category': 'action',
                'description': '热力图行列转置：交换 x 轴和 y 轴（交互必要性：快速切换视角）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            },
            'change_encoding': {
                'function': heatmap_tools.change_encoding,
                'category': 'action',
                'description': '修改指定编码通道的字段映射 (color, size, shape, opacity, x, y 等)',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'channel': {'type': 'str', 'required': True, 'description': '编码通道 (color, size, shape, opacity, x, y 等)'},
                    'field': {'type': 'str', 'required': True, 'description': '新的字段名'}
                }
            }
        }
        
        # 平行坐标图工具
        parallel_coords_tools_dict = {
            'reorder_dimensions': {
                'function': parallel_coordinates_tools.reorder_dimensions,
                'category': 'action',
                'description': '重新排序维度',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'dimension_order': {'type': 'list', 'required': True}
                }
            },
            'filter_by_category': {
                'function': parallel_coordinates_tools.filter_by_category,
                'category': 'action',
                'description': '按分类字段筛选数据',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'field': {'type': 'str', 'required': True, 'description': '分类字段名'},
                    'values': {'type': 'str|list', 'required': True, 'description': '要保留的值列表'}
                }
            },
            'highlight_category': {
                'function': parallel_coordinates_tools.highlight_category,
                'category': 'action',
                'description': '高亮指定类别，其他变暗',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'field': {'type': 'str', 'required': True, 'description': '分类字段名'},
                    'values': {'type': 'str|list', 'required': True, 'description': '要高亮的值列表'}
                }
            },
            'hide_dimensions': {
                'function': parallel_coordinates_tools.hide_dimensions,
                'category': 'action',
                'description': '隐藏/显示平行坐标图中的维度轴（交互必要性：维度太多时临时隐藏不关心的维度）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'dimensions': {'type': 'list', 'required': True, 'description': '要隐藏或显示的维度名称列表'},
                    'mode': {'type': 'str', 'required': False, 'default': 'hide', 'description': '"hide" 或 "show"'}
                }
            },
            'reset_hidden_dimensions': {
                'function': parallel_coordinates_tools.reset_hidden_dimensions,
                'category': 'action',
                'description': '重置所有隐藏的维度，恢复到全部可见状态',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True}
                }
            }
        }
        
        # 桑基图工具
        sankey_tools_dict = {
            'filter_flow': {
                'function': sankey_tools.filter_flow,
                'category': 'action',
                'description': '筛选流量：只显示流量大于等于阈值的连接',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'min_value': {'type': 'float', 'required': True, 'description': '最小流量阈值'}
                }
            },
            'highlight_path': {
                'function': sankey_tools.highlight_path,
                'category': 'action',
                'description': '高亮多步路径：高亮路径上所有相邻节点之间的连接',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'path': {'type': 'list', 'required': True, 'description': '节点路径列表，如 ["A", "B", "C", "D"]'}
                }
            },
            'calculate_conversion_rate': {
                'function': sankey_tools.calculate_conversion_rate,
                'category': 'analysis',
                'description': '计算转化率：分析每个节点的入流、出流和转化率',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'node_name': {'type': 'str', 'required': False, 'description': '指定节点名称（可选），不指定则返回所有节点'}
                }
            },
            'trace_node': {
                'function': sankey_tools.trace_node,
                'category': 'action',
                'description': '追踪节点：高亮与该节点相连的所有连接',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'node_name': {'type': 'str', 'required': True, 'description': '要追踪的节点名称'}
                }
            },
            'collapse_nodes': {
                'function': sankey_tools.collapse_nodes,
                'category': 'action',
                'description': '折叠节点：将多个节点合并为一个聚合节点',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'nodes_to_collapse': {'type': 'list', 'required': True, 'description': '要折叠的节点名称列表'},
                    'aggregate_name': {'type': 'str', 'required': False, 'default': 'Other', 'description': '聚合节点名称'}
                }
            },
            'expand_node': {
                'function': sankey_tools.expand_node,
                'category': 'action',
                'description': '展开聚合节点：恢复被折叠的原始节点',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'aggregate_name': {'type': 'str', 'required': True, 'description': '要展开的聚合节点名称'}
                }
            },
            'auto_collapse_by_rank': {
                'function': sankey_tools.auto_collapse_by_rank,
                'category': 'action',
                'description': '按排名自动折叠：每层只保留 top N 个节点，其余折叠到 Others',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'top_n': {'type': 'int', 'required': False, 'default': 5, 'description': '每层保留的节点数量'}
                }
            },
            'color_flows': {
                'function': sankey_tools.color_flows,
                'category': 'action',
                'description': '给与指定节点相连的流着色',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'nodes': {'type': 'list', 'required': True, 'description': '节点列表，与这些节点相连的流将被着色'},
                    'color': {'type': 'str', 'required': False, 'default': '#e74c3c', 'description': '着色颜色'}
                }
            },
            'find_bottleneck': {
                'function': sankey_tools.find_bottleneck,
                'category': 'analysis',
                'description': '识别流失最严重的节点（瓶颈）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'top_n': {'type': 'int', 'required': False, 'default': 3, 'description': '返回流失最严重的前N个节点'}
                }
            },
            'reorder_nodes_in_layer': {
                'function': sankey_tools.reorder_nodes_in_layer,
                'category': 'action',
                'description': '重排桑基图某一层内节点的上下顺序（交互必要性：减少边交叉，提高可读性）。⚠️ 必须提供 depth（层号，0=第一层）和 order/sort_by 中的一个（互斥）',
                'params': {
                    'vega_spec': {'type': 'dict', 'required': True},
                    'depth': {'type': 'int', 'required': True, 'description': '要重排的层号（必需）：0=第一层/最左侧，1=第二层，2=第三层，以此类推'},
                    'order': {'type': 'list', 'required': False, 'description': '节点名称列表，按从上到下顺序。与 sort_by 互斥，必须提供其中一个'},
                    'sort_by': {'type': 'str', 'required': False, 'description': '排序方式（与 order 互斥，必须提供其中一个）："value_desc"（按流量降序，流量大的在上）| "value_asc"（按流量升序）| "name"（按名称字母顺序）'}
                }
            }
        }
        
        # 注册所有工具
        for name, info in common_tools.items():
            self._tools[name] = info
        
        for name, info in bar_chart_tools_dict.items():
            self._tools[name] = info
        
        for name, info in line_chart_tools_dict.items():
            self._tools[name] = info
        
        for name, info in scatter_tools.items():
            self._tools[name] = info
        
        for name, info in heatmap_tools_dict.items():
            self._tools[name] = info
        
        for name, info in parallel_coords_tools_dict.items():
            self._tools[name] = info
        
        for name, info in sankey_tools_dict.items():
            self._tools[name] = info
        
        # 映射图表类型到工具
        self._chart_tools[ChartType.BAR_CHART] = list(bar_chart_tools_dict.keys()) + list(common_tools.keys())
        self._chart_tools[ChartType.LINE_CHART] = list(line_chart_tools_dict.keys()) + list(common_tools.keys())
        self._chart_tools[ChartType.SCATTER_PLOT] = list(scatter_tools.keys()) + list(common_tools.keys())
        self._chart_tools[ChartType.HEATMAP] = list(heatmap_tools_dict.keys()) + list(common_tools.keys())
        self._chart_tools[ChartType.PARALLEL_COORDINATES] = list(parallel_coords_tools_dict.keys()) + list(common_tools.keys())
        self._chart_tools[ChartType.SANKEY_DIAGRAM] = list(sankey_tools_dict.keys()) + list(common_tools.keys())
    
    def get_tool(self, tool_name: str) -> Dict[str, Any]:
        """获取工具信息"""
        return self._tools.get(tool_name)
    
    def list_tools_for_chart(self, chart_type: ChartType) -> List[str]:
        """列出指定图表类型可用的工具"""
        return self._chart_tools.get(chart_type, list(self._tools.keys()))
    
    def list_all_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())


tool_registry = ToolRegistry()
