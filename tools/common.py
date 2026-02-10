"""
通用工具库（简化版 - 直接使用 vega_spec，无需 view_id）
包含感知类API和行动类API，适用于所有图表类型
"""

import json
import copy
import hashlib
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime


def _datum_ref(field: str) -> str:
    """Vega expr: datum access that supports field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"


# ==================== 感知类 API (Perception APIs) ====================

def _get_primary_encoding(vega_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer layer[0].encoding when present (common for line/scatter)."""
    if isinstance(vega_spec.get('layer'), list) and len(vega_spec['layer']) > 0:
        enc = vega_spec['layer'][0].get('encoding', {})
        return enc if isinstance(enc, dict) else {}
    enc = vega_spec.get('encoding', {})
    return enc if isinstance(enc, dict) else {}


def _coerce_comparable(v: Any) -> Any:
    """Coerce values for range comparisons (numbers, ISO dates)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        # number-like
        try:
            return float(s)
        except Exception:
            pass
        # ISO-ish datetime
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return s
    return v


def _apply_selected_region(data: List[Dict[str, Any]], vega_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply _selected_region (from scatter tools) if present."""
    region = vega_spec.get('_selected_region')
    if not isinstance(region, dict):
        return data
    x_field = region.get('x_field')
    y_field = region.get('y_field')
    x_range = region.get('x_range')
    y_range = region.get('y_range')
    if not x_field or not y_field or not isinstance(x_range, list) or not isinstance(y_range, list):
        return data
    if len(x_range) != 2 or len(y_range) != 2:
        return data

    xl, xu = _coerce_comparable(x_range[0]), _coerce_comparable(x_range[1])
    yl, yu = _coerce_comparable(y_range[0]), _coerce_comparable(y_range[1])

    out: List[Dict[str, Any]] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        xv = _coerce_comparable(r.get(x_field))
        yv = _coerce_comparable(r.get(y_field))
        if xv is None or yv is None:
            continue
        try:
            if xl <= xv <= xu and yl <= yv <= yu:
                out.append(r)
        except Exception:
            # if comparison fails, do not filter it out aggressively
            out.append(r)
    return out


def get_view_spec(vega_spec: Dict) -> Dict[str, Any]:
    """
    返回当前视图的结构化状态信息
    
    Args:
        vega_spec: Vega-Lite/Vega 规范
        
    Returns:
        视图状态的结构化描述
    """
    # 计算 spec hash
    spec_str = json.dumps(vega_spec, sort_keys=True, default=str)
    spec_hash = hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    # 检测图表类型
    chart_type = _detect_chart_type(vega_spec)
    
    # 提取数据信息
    data = _get_spec_data(vega_spec)
    data_count = len(data) if data else 0
    
    # 提取 encoding（支持 layer）
    encoding = _get_primary_encoding(vega_spec)
    
    # 提取 mark
    mark = vega_spec.get('mark', {})
    if isinstance(mark, str):
        mark = {'type': mark}
    
    # 提取可见域（从 scale 或 encoding 中，支持 layer）
    visible_domain = _extract_visible_domain(vega_spec, data)
    
    # 提取 transforms
    transforms = vega_spec.get('transform', [])
    # 过滤掉内部标记的 transform
    transforms = [t for t in transforms if not t.get('_avs_tag')]
    
    # 提取 selections（从我们的内部状态中）
    selections = vega_spec.get('_avs_selections', [])
    
    return {
        'success': True,
        'spec_hash': f'sha256:{spec_hash}',
        'payload': {
            'chart_type': chart_type,
            'title': vega_spec.get('title', ''),
            'data_count': data_count,
            'mark': mark,
            'encoding': _simplify_encoding(encoding),
            'visible_domain': visible_domain,
            'transforms': transforms,
            'selections': selections
        }
    }


def get_data(vega_spec: Dict, scope: str = 'all') -> Dict[str, Any]:
    """
    返回原始数据
    
    Args:
        vega_spec: Vega-Lite/Vega 规范
        scope: 数据范围
            - 'all': 全部原始数据
            - 'filter': 经过 transform filter 后的数据
            - 'visible': 当前视图可见的数据（考虑 domain）
            - 'selected': 被选中的数据
        
    Returns:
        数据列表
    """
    data = _get_spec_data(vega_spec)
    
    if not data:
        return {
            'success': False,
            'error': 'No data available in spec'
        }
    
    total_count = len(data)
    fields = list(data[0].keys()) if data else []
    
    transforms = vega_spec.get('transform', [])

    if scope == 'all':
        result_data = data

    elif scope == 'filter':
        result_data = _apply_filters(data, transforms)

    elif scope == 'visible':
        # visible = filter transforms + scale.domain + selection region (if any)
        result_data = _apply_filters(data, transforms)
        result_data = _filter_by_domain(result_data, vega_spec)
        result_data = _apply_selected_region(result_data, vega_spec)
        selections = vega_spec.get('_avs_selections', [])
        if selections:
            result_data = _apply_selections(result_data, selections)

    elif scope == 'selected':
        # Prefer _selected_region; fallback to _avs_selections.
        result_data = _apply_filters(data, transforms)
        result_data = _apply_selected_region(result_data, vega_spec)
        if result_data == data:
            selections = vega_spec.get('_avs_selections', [])
            result_data = _apply_selections(result_data, selections) if selections else []
    else:
        return {
            'success': False,
            'error': f'Unknown scope: {scope}. Valid values: all, filter, visible, selected'
        }
    
    return {
        'success': True,
        'scope': scope,
        'total_count': total_count,
        'returned_count': len(result_data),
        'fields': fields,
        'data': result_data
    }


def get_data_summary(vega_spec: Dict, scope: str = 'all') -> Dict[str, Any]:
    """
    返回数据的统计摘要
    
    Args:
        vega_spec: Vega-Lite规范
        scope: 'visible' 或 'all' - 返回可见数据或全部数据的统计
        
    Returns:
        统计摘要字典
    """
    # 与 get_data 使用同一套 scope 语义
    data_result = get_data(vega_spec, scope=scope if scope else 'all')
    if not data_result.get('success'):
        return {'success': False, 'error': data_result.get('error', 'No data available')}
    data = data_result.get('data', [])
    
    if not data:
        return {'success': False, 'error': 'No data available'}
    
    # 计算统计信息
    summary = {
        'count': len(data),
        'numeric_fields': {},
        'categorical_fields': {}
    }
    
    if data:
        sample = data[0]
        for field_name in sample.keys():
            values = [row.get(field_name) for row in data if row.get(field_name) is not None]
            
            if not values:
                continue
            
            if isinstance(values[0], (int, float)):
                summary['numeric_fields'][field_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            else:
                unique_values = list(set(values))
                value_counts = {v: values.count(v) for v in unique_values}
                
                summary['categorical_fields'][field_name] = {
                    'unique_count': len(unique_values),
                    'categories': unique_values[:20],
                    'distribution': value_counts
                }
    
    return {
        'success': True,
        'scope': scope,
        'summary': summary
    }


def get_tooltip_data(vega_spec: Dict, position: Tuple[float, float]) -> Dict[str, Any]:
    """获取指定位置的工具提示数据"""
    data = vega_spec.get('data', {}).get('values', [])
    
    x_pos, y_pos = position
    x_field = _get_encoding_field(vega_spec, 'x')
    y_field = _get_encoding_field(vega_spec, 'y')
    
    if not x_field or not y_field:
        return {'success': False, 'message': 'Cannot find x/y fields'}
    
    closest_point = None
    min_distance = float('inf')
    
    for row in data:
        x_val = row.get(x_field)
        y_val = row.get(y_field)
        
        if x_val is not None and y_val is not None:
            distance = np.sqrt((x_val - x_pos)**2 + (y_val - y_pos)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = row
    
    if closest_point:
        return {'success': True, 'data': closest_point, 'distance': min_distance}
    
    return {'success': False, 'message': 'No data point found'}

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

# ==================== 行动类 API (Action APIs) ====================

def reset_view(vega_spec: Dict, original_spec: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Reset view to original state.
    
    Reads original_spec from vega_spec._original_spec metadata field.
    If original_spec parameter is provided (for backward compatibility), it takes precedence.
    
    Args:
        vega_spec: Current view's vega_spec (contains metadata)
        original_spec: Original view's vega_spec (optional, for backward compatibility)
        
    Returns:
        Reset vega_spec
    """
    # Try parameter first (backward compatibility), then metadata
    if original_spec is None:
        original_spec = vega_spec.get('_original_spec')
    
    if original_spec is None:
        return {
            'success': False,
            'error': 'original_spec not found in vega_spec metadata'
        }

    return {
        'success': True,
        'operation': 'reset_view',
        'vega_spec': copy.deepcopy(original_spec),
        'message': '视图已重置到原始状态'
    }


def undo_view(vega_spec: Dict, spec_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Undo previous view, return previous version of vega_spec.
    
    Reads spec_history from vega_spec._spec_history metadata field.
    If spec_history parameter is provided (for backward compatibility), it takes precedence.
    
    Args:
        vega_spec: Current view's vega_spec (contains metadata)
        spec_history: View history list (optional, for backward compatibility; will be modified using pop)
    """
    # Try parameter first (backward compatibility), then metadata
    if spec_history is None:
        spec_history = vega_spec.get('_spec_history')
    
    if spec_history is None:
        return {'success': False, 'error': 'spec_history not found in vega_spec metadata'}

    if not isinstance(spec_history, list):
        return {'success': False, 'error': 'spec_history must be a list'}

    if not spec_history:
        return {'success': False, 'error': 'no previous view to undo'}

    prev_spec = spec_history.pop()  # LIFO
    return {
        'success': True,
        'operation': 'undo_view',
        'vega_spec': copy.deepcopy(prev_spec),
        'message': '已回到上一步视图'
    }


def render_chart(vega_spec: Dict) -> Dict[str, Any]:
    """渲染图表"""
    from core.vega_service import get_vega_service
    from core.utils import app_logger
    
    try:
        vega_service = get_vega_service()
        render_result = vega_service.render(vega_spec)
        
        if render_result.get("success"):
            return {
                'success': True,
                'operation': 'render',
                'image_base64': render_result["image_base64"],
                'renderer': render_result.get("renderer"),
                'message': f'Rendered using {render_result.get("renderer")}'
            }
        else:
            return {
                'success': False,
                'operation': 'render',
                'error': render_result.get("error")
            }
    except Exception as e:
        return {'success': False, 'operation': 'render', 'error': str(e)}


# ==================== 辅助函数 ====================

def _get_encoding_field(vega_spec: Dict, channel: str) -> Optional[str]:
    """获取编码字段"""
    return vega_spec.get('encoding', {}).get(channel, {}).get('field')


def _get_primary_category_field(vega_spec: Dict) -> str:
    """获取主分类字段"""
    encoding = vega_spec.get('encoding', {})
    for channel in ['color', 'x', 'y']:
        if channel in encoding:
            field = encoding[channel].get('field')
            field_type = encoding[channel].get('type')
            if field and field_type in ['nominal', 'ordinal']:
                return field
    return 'category'


def _infer_field_type(vega_spec: Dict, field_name: str) -> str:
    """推断字段类型"""
    data = vega_spec.get('data', {}).get('values', [])
    if not data:
        return 'nominal'
    
    for row in data:
        value = row.get(field_name)
        if value is not None:
            if isinstance(value, (int, float)):
                return 'quantitative'
            elif isinstance(value, str):
                if any(sep in value for sep in ['-', '/', ':']):
                    return 'temporal'
                return 'nominal'
    return 'nominal'


def _get_spec_data(vega_spec: Dict) -> List[Dict]:
    """获取 spec 中的数据（兼容 Vega-Lite 和 Vega）"""
    # Vega-Lite 格式: data.values
    data_obj = vega_spec.get('data', {})
    if isinstance(data_obj, dict) and 'values' in data_obj:
        return data_obj['values']
    
    # Vega 格式: data 是数组
    if isinstance(data_obj, list):
        # 尝试找到主数据源
        for d in data_obj:
            if isinstance(d, dict) and 'values' in d:
                return d['values']
    
    return []


def _detect_chart_type(vega_spec: Dict) -> str:
    """检测图表类型"""
    # 检查是否是 Vega（非 Vega-Lite）
    schema = vega_spec.get('$schema', '')
    if 'vega.github.io/schema/vega/' in schema and 'vega-lite' not in schema:
        # 尝试从 marks 推断
        marks = vega_spec.get('marks', [])
        for mark in marks:
            mark_type = mark.get('type', '')
            if mark_type == 'rect' and 'group' in str(marks):
                return 'sankey_diagram'
        return 'vega_custom'
    
    # Vega-Lite: 从 mark 推断
    mark = vega_spec.get('mark', {})
    if isinstance(mark, str):
        mark_type = mark
    else:
        mark_type = mark.get('type', '')
    
    encoding = vega_spec.get('encoding', {})
    
    if mark_type == 'bar':
        return 'bar_chart'
    elif mark_type in ['line', 'trail']:
        return 'line_chart'
    elif mark_type in ['point', 'circle']:
        return 'scatter_plot'
    elif mark_type == 'rect':
        # 判断是热力图还是其他
        if 'color' in encoding:
            return 'heatmap'
        return 'rect_chart'
    elif mark_type == 'rule':
        return 'parallel_coordinates'
    
    return 'unknown'


def _simplify_encoding(encoding: Dict) -> Dict:
    """简化 encoding 结构，只保留关键信息"""
    simplified = {}
    for channel, config in encoding.items():
        if isinstance(config, dict):
            simplified[channel] = {
                'field': config.get('field'),
                'type': config.get('type')
            }
            # 保留聚合信息
            if 'aggregate' in config:
                simplified[channel]['aggregate'] = config['aggregate']
            if 'timeUnit' in config:
                simplified[channel]['timeUnit'] = config['timeUnit']
        else:
            simplified[channel] = config
    return simplified


def _extract_visible_domain(vega_spec: Dict, data: List[Dict]) -> Dict:
    """提取可见数据域"""
    domain = {}
    encoding = _get_primary_encoding(vega_spec)
    
    for channel in ['x', 'y']:
        if channel in encoding:
            field = encoding[channel].get('field')
            field_type = encoding[channel].get('type')
            
            # 尝试从 scale 中获取 domain
            scale = encoding[channel].get('scale', {})
            if 'domain' in scale:
                domain[channel] = scale['domain']
            elif field and data and field_type == 'quantitative':
                # 从数据中计算
                values = [row.get(field) for row in data if row.get(field) is not None]
                if values and all(isinstance(v, (int, float)) for v in values):
                    domain[channel] = [min(values), max(values)]
    
    return domain


def _apply_filters(data: List[Dict], transforms: List[Dict]) -> List[Dict]:
    """应用 transform 中的 filter"""
    result = data
    
    for t in transforms:
        if 'filter' in t:
            filter_expr = t['filter']
            if isinstance(filter_expr, str):
                # 简单解析常见的 filter 表达式
                result = _eval_filter_expr(result, filter_expr)
            elif isinstance(filter_expr, dict):
                # Vega-Lite filter object
                field = filter_expr.get('field')
                if 'equal' in filter_expr:
                    result = [r for r in result if r.get(field) == filter_expr['equal']]
                elif 'oneOf' in filter_expr:
                    result = [r for r in result if r.get(field) in filter_expr['oneOf']]
                elif 'range' in filter_expr:
                    rng = filter_expr['range']
                    result = [r for r in result if rng[0] <= r.get(field, 0) <= rng[1]]
    
    return result


def _eval_filter_expr(data: List[Dict], expr: str) -> List[Dict]:
    """简单解析 filter 表达式"""
    # 简化处理：支持常见模式
    # datum.field == value, datum.field > value, etc.
    result = []
    
    for row in data:
        try:
            # 创建 datum 对象
            datum = type('datum', (), row)()
            # 安全评估（仅支持简单比较）
            if eval(expr, {'datum': datum, '__builtins__': {}}):
                result.append(row)
        except:
            # 表达式太复杂，跳过
            result.append(row)
    
    return result


def _filter_by_domain(data: List[Dict], vega_spec: Dict) -> List[Dict]:
    """根据可见域过滤数据"""
    encoding = _get_primary_encoding(vega_spec)
    result = data
    
    for channel in ['x', 'y']:
        if channel in encoding:
            field = encoding[channel].get('field')
            scale = encoding[channel].get('scale', {})
            domain = scale.get('domain')
            
            if field and domain and isinstance(domain, list) and len(domain) == 2:
                lb = _coerce_comparable(domain[0])
                ub = _coerce_comparable(domain[1])
                filtered = []
                for r in result:
                    if not isinstance(r, dict):
                        continue
                    rv = _coerce_comparable(r.get(field))
                    if rv is None:
                        continue
                    try:
                        if lb <= rv <= ub:
                            filtered.append(r)
                    except Exception:
                        # If comparison fails, keep row to avoid over-filtering.
                        filtered.append(r)
                result = filtered
    
    return result


def _apply_selections(data: List[Dict], selections: List[Dict]) -> List[Dict]:
    """应用选择条件"""
    if not selections:
        return []
    
    result = data
    
    for sel in selections:
        field = sel.get('field')
        op = sel.get('op', '==')
        values = sel.get('values', [])
        
        if not field:
            continue
        
        if op == '==' or op == 'eq':
            if isinstance(values, list):
                result = [r for r in result if r.get(field) in values]
            else:
                result = [r for r in result if r.get(field) == values]
        elif op == '!=' or op == 'neq':
            if isinstance(values, list):
                result = [r for r in result if r.get(field) not in values]
            else:
                result = [r for r in result if r.get(field) != values]
        elif op == '>' or op == 'gt':
            val = values[0] if isinstance(values, list) else values
            result = [r for r in result if r.get(field, 0) > val]
        elif op == '>=' or op == 'gte':
            val = values[0] if isinstance(values, list) else values
            result = [r for r in result if r.get(field, 0) >= val]
        elif op == '<' or op == 'lt':
            val = values[0] if isinstance(values, list) else values
            result = [r for r in result if r.get(field, 0) < val]
        elif op == '<=' or op == 'lte':
            val = values[0] if isinstance(values, list) else values
            result = [r for r in result if r.get(field, 0) <= val]
        elif op == 'in':
            result = [r for r in result if r.get(field) in values]
        elif op == 'not_in':
            result = [r for r in result if r.get(field) not in values]
    
    return result


__all__ = [
    'get_view_spec',
    'get_data',
    'get_data_summary',
    'get_tooltip_data',
    'reset_view',
    'undo_view',
    'render_chart',
]
