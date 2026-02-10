"""
折线图专用工具（简化版 - 使用 vega_spec）
"""

from typing import List, Dict, Any, Tuple, Optional
import copy
import json


def _datum_ref(field: str) -> str:
    """Vega expr: datum access for field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"


def _get_time_field(vega_spec: Dict) -> Optional[str]:
    """获取时间字段（支持 layer 结构）"""
    # 检查是否有 layer 结构
    if 'layer' in vega_spec and len(vega_spec['layer']) > 0:
        encoding = vega_spec['layer'][0].get('encoding', {})
    else:
        encoding = vega_spec.get('encoding', {})
    
    # 时间字段通常在 x 轴，但也可能在 y 轴
    x_encoding = encoding.get('x', {})
    y_encoding = encoding.get('y', {})
    
    if x_encoding.get('type') == 'temporal':
        return x_encoding.get('field')
    elif y_encoding.get('type') == 'temporal':
        return y_encoding.get('field')
    
    # 如果类型未指定，尝试从字段推断（通常时间在 x 轴）
    return x_encoding.get('field') or y_encoding.get('field')


def zoom_time_range(vega_spec: Dict, start: str, end: str) -> Dict[str, Any]:
    """缩放时间范围 - 放大视图到特定时间段（不删除数据）"""
    new_spec = copy.deepcopy(vega_spec)
    
    # 获取时间字段名（支持 layer 结构）
    time_field = _get_time_field(new_spec)
    
    if not time_field:
        return {
            'success': False,
            'error': 'Cannot find time field'
        }
    
    # 确定时间字段在哪个轴上
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    x_field = encoding.get('x', {}).get('field')
    time_axis = 'x' if x_field == time_field else 'y'
    
    # 直接使用VLM提供的日期格式，不做任何转换
    # VLM应该根据提示词观察原始数据格式并返回一致的格式
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        layer_encoding = new_spec['layer'][0].get('encoding', {})
        if time_axis not in layer_encoding:
            layer_encoding[time_axis] = {}
        if 'scale' not in layer_encoding[time_axis]:
            layer_encoding[time_axis]['scale'] = {}
        layer_encoding[time_axis]['scale']['domain'] = [start, end]
    else:
        if 'encoding' not in new_spec:
            new_spec['encoding'] = {}
        if time_axis not in new_spec['encoding']:
            new_spec['encoding'][time_axis] = {}
        if 'scale' not in new_spec['encoding'][time_axis]:
            new_spec['encoding'][time_axis]['scale'] = {}
        new_spec['encoding'][time_axis]['scale']['domain'] = [start, end]
    
    # 确保 mark 有 clip: true，裁剪超出范围的点
    if 'layer' in new_spec:
        for layer in new_spec['layer']:
            if 'mark' in layer:
                if isinstance(layer['mark'], dict):
                    layer['mark']['clip'] = True
                else:
                    layer['mark'] = {'type': layer['mark'], 'clip': True}
    else:
        if 'mark' in new_spec:
            if isinstance(new_spec['mark'], dict):
                new_spec['mark']['clip'] = True
            else:
                new_spec['mark'] = {'type': new_spec['mark'], 'clip': True}
    
    return {
        'success': True,
        'operation': 'zoom_time_range',
        'vega_spec': new_spec,
        'message': f'Zoomed to time range: {start} to {end}',
        'details': [f'View zoomed to show time range between {start} and {end}']
    }


def highlight_trend(vega_spec: Dict, trend_type: str = "increasing") -> Dict[str, Any]:
    """高亮趋势 - 添加回归趋势线"""
    new_spec = copy.deepcopy(vega_spec)
    
    # 获取 x 和 y 字段
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    y_field = encoding.get('y', {}).get('field')
    x_field = encoding.get('x', {}).get('field')
    
    if not y_field or not x_field:
        return {
            'success': False,
            'error': 'Cannot find x or y field for trend line'
        }
    
    # 如果原规范没有 layer，转换为 layer 结构
    if 'layer' not in new_spec:
        original_layer = copy.deepcopy(new_spec)
        # 移除顶层的 mark 和 encoding，因为它们现在在 layer 中
        for key in ['mark', 'encoding']:
            if key in original_layer:
                del original_layer[key]
        
        new_spec = original_layer
        new_spec['layer'] = [{
            'mark': vega_spec.get('mark', 'line'),
            'encoding': vega_spec.get('encoding', {})
        }]
    
    # 添加趋势线图层
    new_spec['layer'].append({
        'mark': {
            'type': 'line',
            'color': 'red',
            'strokeDash': [5, 5],
            'strokeWidth': 2
        },
        'transform': [{
            'regression': y_field,
            'on': x_field
        }],
        'encoding': {
            'x': {'field': x_field, 'type': encoding['x'].get('type', 'temporal')},
            'y': {'field': y_field, 'type': encoding['y'].get('type', 'quantitative')}
        }
    })
    
    return {
        'success': True,
        'operation': 'highlight_trend',
        'vega_spec': new_spec,
        'message': f'Added {trend_type} regression trend line',
        'details': [f'Trend line shows overall {trend_type} pattern']
    }



def detect_anomalies(vega_spec: Dict, threshold: float = 2.0) -> Dict[str, Any]:
    """检测异常点 - 检测并在视图中高亮标记异常数据点"""
    import numpy as np
    
    data = vega_spec.get('data', {}).get('values', [])
    
    # 获取字段（支持 layer 结构）
    if 'layer' in vega_spec and len(vega_spec['layer']) > 0:
        encoding = vega_spec['layer'][0].get('encoding', {})
    else:
        encoding = vega_spec.get('encoding', {})
    
    y_field = encoding.get('y', {}).get('field')
    x_field = encoding.get('x', {}).get('field')
    
    if not data or not y_field:
        return {'success': False, 'error': 'Missing data or y field'}
    
    values = [row.get(y_field) for row in data if row.get(y_field) is not None]
    
    if len(values) < 3:
        return {'success': False, 'error': 'Not enough data for anomaly detection'}
    
    # 计算统计特征
    mean = np.mean(values)
    std = np.std(values)
    
    # 识别异常点
    anomaly_data = []
    for row in data:
        val = row.get(y_field)
        if val is not None and abs(val - mean) > threshold * std:
            anomaly_data.append(row)
    
    new_spec = copy.deepcopy(vega_spec)
    
    # 如果检测到异常点，在视图中标记
    if anomaly_data:
        # 转换为 layer 结构
        if 'layer' not in new_spec:
            original_layer = copy.deepcopy(new_spec)
            for key in ['mark', 'encoding']:
                if key in original_layer:
                    del original_layer[key]
            
            new_spec = original_layer
            new_spec['layer'] = [{
                'mark': vega_spec.get('mark', 'line'),
                'encoding': vega_spec.get('encoding', {})
            }]
        
        # 添加异常点标记图层
        new_spec['layer'].append({
            'data': {'values': anomaly_data},
            'mark': {
                'type': 'point',
                'color': 'red',
                'size': 100,
                'filled': True
            },
            'encoding': {
                'x': {'field': x_field, 'type': encoding['x'].get('type', 'temporal')} if x_field else {},
                'y': {'field': y_field, 'type': encoding['y'].get('type', 'quantitative')},
                'tooltip': [
                    {'field': x_field, 'type': encoding['x'].get('type', 'temporal'), 'title': 'Time'} if x_field else {},
                    {'field': y_field, 'type': 'quantitative', 'title': 'Value (Anomaly)'}
                ]
            }
        })
    
    return {
        'success': True,
        'operation': 'detect_anomalies',
        'vega_spec': new_spec,
        'anomaly_count': len(anomaly_data),
        'anomalies': anomaly_data[:10],
        'message': f'Detected and highlighted {len(anomaly_data)} anomalies (threshold={threshold} std)',
        'details': [
            f'Mean: {mean:.2f}, Std: {std:.2f}',
            f'Anomalies are values beyond {threshold} standard deviations from mean',
            f'Anomalies marked with red points on the chart'
        ]
    }


def bold_lines(vega_spec: Dict, line_names: List[str], line_field: str = None) -> Dict[str, Any]:
    """
    加粗指定的折线
    
    Args:
        vega_spec: Vega-Lite规范
        line_names: 要加粗的折线名称列表
        line_field: 折线分组字段名（可选，自动探测 color/detail 字段）
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # 自动探测分组字段（优先 color，其次 detail）
    if line_field is None:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            encoding = new_spec['layer'][0].get('encoding', {})
        else:
            encoding = new_spec.get('encoding', {})
        
        color_enc = encoding.get('color', {})
        line_field = color_enc.get('field')
        
        if not line_field:
            # 尝试从 detail 字段获取
            detail_enc = encoding.get('detail', {})
            line_field = detail_enc.get('field')
    
    if not line_field:
        return {
            'success': False,
            'error': 'Cannot find line grouping field. Please specify line_field parameter.'
        }
    
    # 构建 strokeWidth 条件编码（支持含空格的字段名）
    lines_json = json.dumps(line_names)
    stroke_width_encoding = {
        'condition': {
            'test': f'indexof({lines_json}, {_datum_ref(line_field)}) >= 0',
            'value': 4
        },
        'value': 1
    }
    
    # 应用到 spec
    if 'layer' in new_spec:
        for layer in new_spec['layer']:
            mark = layer.get('mark', {})
            if (isinstance(mark, dict) and mark.get('type') == 'line') or mark == 'line':
                if 'encoding' not in layer:
                    layer['encoding'] = {}
                layer['encoding']['strokeWidth'] = stroke_width_encoding
    else:
        if 'encoding' not in new_spec:
            new_spec['encoding'] = {}
        new_spec['encoding']['strokeWidth'] = stroke_width_encoding
    
    return {
        'success': True,
        'operation': 'bold_lines',
        'vega_spec': new_spec,
        'message': f'Bolded lines: {line_names}'
    }


def filter_lines(vega_spec: Dict, lines_to_remove: List[str], line_field: str = None) -> Dict[str, Any]:
    """
    过滤掉指定的折线
    
    Args:
        vega_spec: Vega-Lite规范
        lines_to_remove: 要移除的折线名称列表
        line_field: 折线分组字段名（可选，自动探测 color/detail 字段）
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # 自动探测分组字段（优先 color，其次 detail）
    if line_field is None:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            encoding = new_spec['layer'][0].get('encoding', {})
        else:
            encoding = new_spec.get('encoding', {})
        
        color_enc = encoding.get('color', {})
        line_field = color_enc.get('field')
        
        if not line_field:
            detail_enc = encoding.get('detail', {})
            line_field = detail_enc.get('field')
    
    if not line_field:
        return {
            'success': False,
            'error': 'Cannot find line grouping field. Please specify line_field parameter.'
        }
    
    # 添加 filter transform 排除指定系列
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    lines_json = json.dumps(lines_to_remove)
    new_spec['transform'].append({
        'filter': f'indexof({lines_json}, {_datum_ref(line_field)}) < 0'
    })
    
    return {
        'success': True,
        'operation': 'filter_lines',
        'vega_spec': new_spec,
        'message': f'Filtered out lines: {lines_to_remove}'
    }


def show_moving_average(vega_spec: Dict, window_size: int = 3) -> Dict[str, Any]:
    """
    叠加移动平均线
    
    Args:
        vega_spec: Vega-Lite规范
        window_size: 移动平均窗口大小
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 获取字段
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    y_field = encoding.get('y', {}).get('field')
    x_field = encoding.get('x', {}).get('field')
    
    if not y_field or not x_field:
        return {
            'success': False,
            'error': 'Cannot find x or y field for moving average'
        }
    
    # 如果原规范没有 layer，转换为 layer 结构
    if 'layer' not in new_spec:
        original_layer = copy.deepcopy(new_spec)
        for key in ['mark', 'encoding']:
            if key in original_layer:
                del original_layer[key]
        
        new_spec = original_layer
        new_spec['layer'] = [{
            'mark': vega_spec.get('mark', 'line'),
            'encoding': vega_spec.get('encoding', {})
        }]
    
    # 检测是否有分组字段（多条线的情况）
    color_field = encoding.get('color', {}).get('field')
    detail_field = encoding.get('detail', {}).get('field')
    group_field = color_field or detail_field
    
    # 移动平均字段名
    ma_field = f'{y_field}_ma'
    
    # 构建 window transform 计算移动平均
    window_transform = {
        'window': [{
            'op': 'mean',
            'field': y_field,
            'as': ma_field
        }],
        'frame': [-(window_size - 1), 0],
        'sort': [{'field': x_field, 'order': 'ascending'}]  # 按 x 轴排序，确保时间顺序正确
    }
    
    # 如果有分组字段（多条线），需要按组分别计算移动平均
    if group_field:
        window_transform['groupby'] = [group_field]
    
    # 添加移动平均线图层
    ma_encoding = {
        'x': {'field': x_field, 'type': encoding['x'].get('type', 'temporal')},
        'y': {'field': ma_field, 'type': 'quantitative'}
    }
    
    # 保持与原折线一致的分组/颜色编码，避免不同 lines 被连成一条线
    if color_field and isinstance(encoding.get('color'), dict):
        ma_encoding['color'] = copy.deepcopy(encoding.get('color'))
    elif detail_field and isinstance(encoding.get('detail'), dict):
        ma_encoding['detail'] = copy.deepcopy(encoding.get('detail'))
    
    new_spec['layer'].append({
        'mark': {
            'type': 'line',
            'color': 'orange',
            'strokeWidth': 3,
            'opacity': 0.8
        },
        'transform': [window_transform],
        'encoding': ma_encoding
    })
    
    return {
        'success': True,
        'operation': 'show_moving_average',
        'vega_spec': new_spec,
        'message': f'Added {window_size}-period moving average line'
    }


def focus_lines(
    vega_spec: Dict,
    lines: List[str],
    line_field: Optional[str] = None,
    mode: str = "dim",
    dim_opacity: float = 0.08,
) -> Dict[str, Any]:
    """
    认知交互必要性：聚焦少数系列，其余变暗或隐藏。
    
    Args:
        vega_spec: Vega-Lite规范
        lines: 需要聚焦的折线名称列表
        line_field: 折线分组字段名（可选，自动探测 color.field 或 detail.field）
        mode: 'dim'（其余变暗）
        dim_opacity: mode='dim' 时非聚焦系列的透明度
    """
    import json

    new_spec = copy.deepcopy(vega_spec)

    if not isinstance(lines, list) or not lines:
        return {'success': False, 'error': 'lines must be a non-empty list'}

    # 自动探测分组字段（优先 color，其次 detail）
    if line_field is None:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            encoding = new_spec['layer'][0].get('encoding', {})
        else:
            encoding = new_spec.get('encoding', {})

        line_field = (encoding.get('color', {}) or {}).get('field')
        if not line_field:
            line_field = (encoding.get('detail', {}) or {}).get('field')

    if not line_field:
        return {'success': False, 'error': 'Cannot find line grouping field. Please specify line_field.'}

    lines_json = json.dumps(lines)

    ref = _datum_ref(line_field)
    if mode == "hide":
        if 'transform' not in new_spec:
            new_spec['transform'] = []
        new_spec['transform'].append({
            'filter': f'indexof({lines_json}, {ref}) >= 0',
            '_avs_tag': 'focus_lines'
        })
    else:
        opacity_encoding = {
            'condition': {
                'test': f'indexof({lines_json}, {ref}) >= 0',
                'value': 1.0
            },
            'value': float(dim_opacity)
        }

        if 'layer' in new_spec:
            for layer in new_spec.get('layer', []):
                mark = layer.get('mark', {})
                if (isinstance(mark, dict) and mark.get('type') == 'line') or mark == 'line':
                    if 'encoding' not in layer:
                        layer['encoding'] = {}
                    layer['encoding']['opacity'] = opacity_encoding
        else:
            if 'encoding' not in new_spec:
                new_spec['encoding'] = {}
            new_spec['encoding']['opacity'] = opacity_encoding

    return {
        'success': True,
        'operation': 'focus_lines',
        'vega_spec': new_spec,
        'message': f'Focused on lines: {lines} (mode={mode})'
    }


def drilldown_line_time(
    vega_spec: Dict,
    level: str,
    value: int,
    parent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    时间折线图下钻：年 → 月 → 日
    
    交互必要性：多年日度数据量巨大，初始只能展示年度聚合趋势。
    通过下钻可以逐层深入，发现更细粒度的模式。
    
    初始视图（年度）只显示每年一个聚合点，下钻后：
    - 下钻到年份：显示该年12个月的月度聚合数据
    - 下钻到月份：显示该月的日度数据
    
    Args:
        vega_spec: Vega-Lite 规范
        level: 'year' | 'month' | 'date'
        value: 对应 level 的数值
               - year: 4位年份如 2023
               - month: 1-12 (1=一月，12=十二月)
               - date: 1-31
        parent: 父级信息，如 {'year': 2023} 或 {'year': 2023, 'month': 3}
    
    Returns:
        下钻后的视图规格
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 初始化或获取下钻状态
    state = new_spec.get('_line_drilldown_state')
    if not isinstance(state, dict):
        state = {}
    
    # 首次下钻时保存原始 transform 和 encoding，以及检测字段名
    if 'original_transform' not in state:
        state['original_transform'] = copy.deepcopy(new_spec.get('transform', []))
        state['original_encoding'] = copy.deepcopy(new_spec.get('encoding', {}))
        state['original_title'] = new_spec.get('title', '')
        
        # 动态检测原始时间字段
        # 优先从 transform 中的 timeUnit 获取原始字段
        raw_time_field = None
        for t in state['original_transform']:
            if isinstance(t, dict) and 'timeUnit' in t and 'field' in t:
                raw_time_field = t['field']
                break
        
        # 如果没找到，从 x encoding 推断
        if not raw_time_field:
            x_enc = state['original_encoding'].get('x', {})
            raw_time_field = x_enc.get('field', 'date')
            # 如果 x 字段是聚合后的（如 year_date），尝试推断原始字段
            if raw_time_field and '_' in raw_time_field:
                # year_date, month_date 等 -> 可能原始字段是 date
                possible_raw = raw_time_field.split('_')[-1]
                # 检查数据中是否有这个字段
                data = new_spec.get('data', {}).get('values', [])
                if data and possible_raw in data[0]:
                    raw_time_field = possible_raw
        
        state['raw_time_field'] = raw_time_field or 'date'
        
        # 动态检测数值字段
        # 优先从 transform 中的 aggregate 获取
        raw_value_field = None
        for t in state['original_transform']:
            if isinstance(t, dict) and 'aggregate' in t:
                aggs = t['aggregate']
                if isinstance(aggs, list) and len(aggs) > 0:
                    raw_value_field = aggs[0].get('field')
                    break
        
        # 如果没找到，从 y encoding 推断
        if not raw_value_field:
            y_enc = state['original_encoding'].get('y', {})
            y_field = y_enc.get('field', '')
            # 如果是聚合后的字段（如 total_sales），尝试从数据中找原始字段
            if y_field.startswith('total_') or y_field.startswith('sum_'):
                possible_raw = y_field.replace('total_', '').replace('sum_', '')
                data = new_spec.get('data', {}).get('values', [])
                if data and possible_raw in data[0]:
                    raw_value_field = possible_raw
            else:
                raw_value_field = y_field
        
        state['raw_value_field'] = raw_value_field or 'value'
        
        # 检测分组字段（color encoding）
        color_enc = state['original_encoding'].get('color', {})
        state['group_field'] = color_enc.get('field')
    
    # 使用存储的字段名
    raw_date_field = state.get('raw_time_field', 'date')
    raw_value_field = state.get('raw_value_field', 'value')
    group_field = state.get('group_field')
    
    # 合并父级信息（显式 parent > 存储的状态）
    p = {}
    if isinstance(state.get('parent'), dict):
        p.update(state.get('parent'))
    if isinstance(parent, dict):
        p.update(parent)
    
    # 规范化 level
    level = str(level).lower().strip()
    
    try:
        value = int(value)
    except (TypeError, ValueError):
        return {'success': False, 'error': f'无效的值: {value}，应为整数'}
    
    # 重建 transform：先过滤，再聚合
    new_transforms = []
    title_suffix = ""
    
    # 构建 groupby 列表
    groupby_fields = ['_time_field_']  # 占位，后面替换
    if group_field:
        groupby_fields.append(group_field)
    
    if level == 'year':
        # 下钻到指定年份，显示月度聚合数据
        if value < 1900 or value > 2100:
            return {'success': False, 'error': f'无效的年份: {value}'}
        
        # 1. 过滤到指定年份
        new_transforms.append({
            'filter': f'year(datum.{raw_date_field}) == {value}',
            '_avs_tag': 'line_drilldown_time'
        })
        # 2. 按月聚合
        new_transforms.append({
            'timeUnit': 'yearmonth',
            'field': raw_date_field,
            'as': 'month_date',
            '_avs_tag': 'line_drilldown_time'
        })
        
        # 构建 groupby
        month_groupby = ['month_date']
        if group_field:
            month_groupby.append(group_field)
        
        new_transforms.append({
            'aggregate': [{'op': 'sum', 'field': raw_value_field, 'as': 'total_value'}],
            'groupby': month_groupby,
            '_avs_tag': 'line_drilldown_time'
        })
        
        state['parent'] = {'year': value}
        title_suffix = f'{value}年月度趋势'
        
        # 更新 encoding
        new_spec['encoding'] = {
            'x': {
                'field': 'month_date',
                'type': 'temporal',
                'title': '月份',
                'axis': {'format': '%Y-%m'}
            },
            'y': {
                'field': 'total_value',
                'type': 'quantitative',
                'title': f'月度总{raw_value_field}'
            },
            'color': state['original_encoding'].get('color', {})
        }
        
    elif level == 'month':
        # 下钻到指定月份，显示日度数据
        year_val = p.get('year')
        if not year_val:
            return {'success': False, 'error': '下钻到月份需要提供 parent.year'}
        
        if value < 1 or value > 12:
            return {'success': False, 'error': f'无效的月份: {value}，应为 1-12'}
        
        # Vega 的 month() 返回 0-11
        vega_month = value - 1
        
        # 1. 过滤到指定年月
        new_transforms.append({
            'filter': f'year(datum.{raw_date_field}) == {year_val} && month(datum.{raw_date_field}) == {vega_month}',
            '_avs_tag': 'line_drilldown_time'
        })
        # 2. 按日聚合（如果同一天有多条记录）
        new_transforms.append({
            'timeUnit': 'yearmonthdate',
            'field': raw_date_field,
            'as': 'day_date',
            '_avs_tag': 'line_drilldown_time'
        })
        
        # 构建 groupby
        day_groupby = ['day_date']
        if group_field:
            day_groupby.append(group_field)
        
        new_transforms.append({
            'aggregate': [{'op': 'sum', 'field': raw_value_field, 'as': 'total_value'}],
            'groupby': day_groupby,
            '_avs_tag': 'line_drilldown_time'
        })
        
        state['parent'] = {'year': year_val, 'month': value}
        title_suffix = f'{year_val}年{value}月日度趋势'
        
        # 更新 encoding
        new_spec['encoding'] = {
            'x': {
                'field': 'day_date',
                'type': 'temporal',
                'title': '日期',
                'axis': {'format': '%m-%d'}
            },
            'y': {
                'field': 'total_value',
                'type': 'quantitative',
                'title': f'日{raw_value_field}'
            },
            'color': state['original_encoding'].get('color', {})
        }
        
    elif level == 'date':
        # 通常不需要进一步下钻到具体日期
        return {'success': False, 'error': '日度数据已是最细粒度，无法继续下钻'}
        
    else:
        return {'success': False, 'error': f'无效的 level: {level}，应为 year/month/date'}
    
    # 应用新的 transform（替换原有的聚合 transform）
    new_spec['transform'] = new_transforms
    
    # 更新标题
    new_spec['title'] = title_suffix
    
    # 保留 tooltip
    if 'tooltip' in state['original_encoding']:
        # 简化 tooltip
        tooltip_list = [
            {'field': new_spec['encoding']['x']['field'], 'type': 'temporal', 'title': '时间'},
        ]
        if group_field:
            tooltip_list.append({'field': group_field, 'type': 'nominal', 'title': group_field})
        tooltip_list.append({'field': 'total_value', 'type': 'quantitative', 'title': raw_value_field, 'format': ',.0f'})
        new_spec['encoding']['tooltip'] = tooltip_list
    
    # 保存状态
    new_spec['_line_drilldown_state'] = state
    
    return {
        'success': True,
        'operation': 'drilldown_line_time',
        'vega_spec': new_spec,
        'message': f'下钻到 {title_suffix}',
        'current_level': level,
        'parent': state.get('parent', {})
    }


def reset_line_drilldown(vega_spec: Dict) -> Dict[str, Any]:
    """
    重置折线图时间下钻，恢复到初始年度视图。
    
    Args:
        vega_spec: Vega-Lite 规范
    
    Returns:
        恢复后的视图规格
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 获取下钻状态
    state = new_spec.get('_line_drilldown_state')
    if not isinstance(state, dict):
        return {
            'success': True,
            'operation': 'reset_line_drilldown',
            'vega_spec': new_spec,
            'message': '未进行过下钻，无需重置'
        }
    
    # 恢复原始 transform
    original_transform = state.get('original_transform')
    if original_transform is not None:
        new_spec['transform'] = copy.deepcopy(original_transform)
    else:
        # 如果没有保存原始 transform，移除下钻相关的 transform
        if 'transform' in new_spec:
            new_spec['transform'] = [
                t for t in new_spec['transform']
                if not (isinstance(t, dict) and t.get('_avs_tag') == 'line_drilldown_time')
            ]
    
    # 恢复原始 encoding
    original_encoding = state.get('original_encoding')
    if original_encoding:
        new_spec['encoding'] = copy.deepcopy(original_encoding)
    
    # 恢复原始标题
    original_title = state.get('original_title')
    if original_title:
        new_spec['title'] = original_title
    
    # 清除状态
    if '_line_drilldown_state' in new_spec:
        del new_spec['_line_drilldown_state']
    
    return {
        'success': True,
        'operation': 'reset_line_drilldown',
        'vega_spec': new_spec,
        'message': '已重置到初始年度视图'
    }


def resample_time(
    vega_spec: Dict,
    granularity: str,
    agg: str = "mean",
) -> Dict[str, Any]:
    """
    时间粒度切换（重采样）：将时间序列从细粒度聚合到粗粒度。
    
    交互必要性：
    - 日数据太密时需要聚合到周/月才能看清趋势
    - 反之，年度数据需要下钻到月/日查看细节
    
    Args:
        vega_spec: Vega-Lite 规范
        granularity: 目标时间粒度 ("day" | "week" | "month" | "quarter" | "year")
        agg: 聚合方式 ("mean" | "sum" | "max" | "min" | "median")，默认 mean
    
    Returns:
        重采样后的规格
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 支持的粒度映射到 Vega-Lite timeUnit
    GRANULARITY_MAP = {
        "day": "yearmonthdate",
        "week": "yearweek",
        "month": "yearmonth",
        "quarter": "yearquarter",
        "year": "year",
    }
    
    granularity_lower = str(granularity).lower().strip()
    if granularity_lower not in GRANULARITY_MAP:
        return {
            'success': False,
            'error': f'Unsupported granularity: {granularity}. Use one of {list(GRANULARITY_MAP.keys())}'
        }
    
    target_timeunit = GRANULARITY_MAP[granularity_lower]
    
    # 支持的聚合方式
    ALLOWED_AGG = {"mean", "sum", "max", "min", "median", "count"}
    agg_lower = str(agg).lower().strip()
    if agg_lower not in ALLOWED_AGG:
        return {
            'success': False,
            'error': f'Unsupported agg: {agg}. Use one of {sorted(list(ALLOWED_AGG))}'
        }
    
    # 获取时间字段
    time_field = _get_time_field(new_spec)
    if not time_field:
        return {'success': False, 'error': 'Cannot find temporal field in encoding'}
    
    # 保存原始状态以便恢复
    state = new_spec.get('_resample_state')
    if not isinstance(state, dict):
        state = {}
    if 'original_encoding' not in state:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            state['original_encoding'] = copy.deepcopy(new_spec['layer'][0].get('encoding', {}))
        else:
            state['original_encoding'] = copy.deepcopy(new_spec.get('encoding', {}))
    
    # 确定时间轴和值轴
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    x_enc = encoding.get('x', {})
    y_enc = encoding.get('y', {})
    
    # 判断时间在哪个轴
    if x_enc.get('field') == time_field or x_enc.get('type') == 'temporal':
        time_axis = 'x'
        value_axis = 'y'
    else:
        time_axis = 'y'
        value_axis = 'x'
    
    # 修改时间轴的 timeUnit
    def _update_encoding(enc: Dict) -> None:
        if time_axis in enc:
            enc[time_axis]['timeUnit'] = target_timeunit
            enc[time_axis]['type'] = 'temporal'
        
        # 在值轴添加聚合
        if value_axis in enc:
            value_field = enc[value_axis].get('field')
            if value_field:
                enc[value_axis]['aggregate'] = agg_lower
    
    if 'layer' in new_spec:
        for layer in new_spec['layer']:
            if 'encoding' in layer:
                _update_encoding(layer['encoding'])
    else:
        if 'encoding' not in new_spec:
            new_spec['encoding'] = {}
        _update_encoding(new_spec['encoding'])
    
    # 保存状态
    state['current_granularity'] = granularity_lower
    state['current_agg'] = agg_lower
    new_spec['_resample_state'] = state
    
    return {
        'success': True,
        'operation': 'resample_time',
        'vega_spec': new_spec,
        'message': f'Resampled time to {granularity} with {agg} aggregation'
    }


def reset_resample(vega_spec: Dict) -> Dict[str, Any]:
    """
    重置时间重采样，恢复到原始粒度。
    """
    new_spec = copy.deepcopy(vega_spec)
    
    state = new_spec.get('_resample_state')
    if not isinstance(state, dict):
        return {
            'success': True,
            'operation': 'reset_resample',
            'vega_spec': new_spec,
            'message': 'No resample state to reset'
        }
    
    original_encoding = state.get('original_encoding')
    if original_encoding:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            new_spec['layer'][0]['encoding'] = copy.deepcopy(original_encoding)
        else:
            new_spec['encoding'] = copy.deepcopy(original_encoding)
    
    if '_resample_state' in new_spec:
        del new_spec['_resample_state']
    
    return {
        'success': True,
        'operation': 'reset_resample',
        'vega_spec': new_spec,
        'message': 'Reset to original time granularity'
    }


def change_encoding(vega_spec: Dict, channel: str, field: str) -> Dict[str, Any]:
    """
    Modify the field mapping of the specified encoding channel
    
    Args:
        vega_spec: Vega spec
        channel: encoding channel ("x", "y", "color", "size", "shape")
        field: new field name
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 检查字段是否存在
    data = new_spec.get('data', {}).get('values', [])
    if data and field not in data[0]:
        available_fields = list(data[0].keys()) if data else []
        return {
            'success': False,
            'error': f'Field "{field}" not found in data. Available fields: {available_fields}'
        }
    
    # 推断字段类型
    field_type = 'nominal'
    if data:
        sample_value = data[0].get(field)
        if isinstance(sample_value, (int, float)):
            field_type = 'quantitative'
        elif isinstance(sample_value, str):
            if any(sep in sample_value for sep in ['-', '/', ':']):
                field_type = 'temporal'
    
    # 更新指定通道的 encoding
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    
    new_spec['encoding'][channel] = {
        'field': field,
        'type': field_type
    }
    
    # 为特定通道添加额外配置
    if channel == 'color':
        new_spec['encoding'][channel]['legend'] = {'title': field}
        if field_type == 'quantitative':
            new_spec['encoding'][channel]['scale'] = {'scheme': 'viridis'}
    elif channel == 'size':
        if field_type == 'quantitative':
            new_spec['encoding'][channel]['scale'] = {'range': [50, 500]}
    
    return {
        'success': True,
        'operation': 'change_encoding',
        'vega_spec': new_spec,
        'message': f'Changed {channel} encoding to field "{field}" (type: {field_type})'
    }


__all__ = [
    'zoom_time_range',
    'highlight_trend',
    'detect_anomalies',
    'bold_lines',
    'filter_lines',
    'show_moving_average',
    'focus_lines',
    'drilldown_line_time',
    'reset_line_drilldown',
    'resample_time',
    'reset_resample',
    'change_encoding',
]
