"""
热力图专用工具（简化版 - 使用 vega_spec）
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import copy
import json
from datetime import datetime


def _datum_ref(field: str) -> str:
    """Vega expr: datum access for field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"


def adjust_color_scale(vega_spec: Dict, scheme: str = "viridis", domain: List = None) -> Dict[str, Any]:
    """
    调整颜色比例
    
    Args:
        vega_spec: Vega-Lite规范
        scheme: 颜色方案 (如 "viridis", "blues", "reds", "greens", "oranges", "purples")
        domain: 数值范围 [min, max]，用于控制颜色映射的数值范围
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    if 'color' not in new_spec['encoding']:
        new_spec['encoding']['color'] = {}
    if 'scale' not in new_spec['encoding']['color']:
        new_spec['encoding']['color']['scale'] = {}
    
    # 设置颜色方案
    new_spec['encoding']['color']['scale']['scheme'] = scheme
    
    # 如果指定了 domain，设置数值范围
    if domain is not None and len(domain) == 2:
        new_spec['encoding']['color']['scale']['domain'] = domain
    
    message = f'Changed color scheme to {scheme}'
    if domain:
        message += f' with domain [{domain[0]}, {domain[1]}]'
    
    return {
        'success': True,
        'operation': 'adjust_color_scale',
        'vega_spec': new_spec,
        'message': message
    }


def filter_cells(
    vega_spec: Dict,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    按数值范围筛选热力图单元格。
    支持单边区间：只传 min_value 保留 >= min 的格子，只传 max_value 保留 <= max 的格子，两者都传则保留 [min, max] 区间。
    """
    if min_value is None and max_value is None:
        return {'success': False, 'error': 'Must provide at least one of min_value or max_value'}

    new_spec = copy.deepcopy(vega_spec)

    color_enc = (new_spec.get('encoding', {}) or {}).get('color', {}) or {}
    color_field = color_enc.get('field')
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field'}

    agg = color_enc.get('aggregate')
    agg_as = color_enc.get('as')
    value_field = color_field
    if agg:
        if isinstance(agg_as, str) and agg_as.strip():
            value_field = agg_as.strip()
        else:
            value_field = f'{str(agg).lower()}_{color_field}'

    if 'transform' not in new_spec:
        new_spec['transform'] = []

    ref = _datum_ref(value_field)
    tests: List[str] = []
    if min_value is not None:
        tests.append(f'{ref} >= {float(min_value)}')
    if max_value is not None:
        tests.append(f'{ref} <= {float(max_value)}')
    filter_expr = ' && '.join(tests)

    new_spec['transform'].append({'filter': filter_expr})

    msg = 'Filtered cells'
    if min_value is not None and max_value is not None:
        msg = f'Filtered cells to [{min_value}, {max_value}]'
    elif min_value is not None:
        msg = f'Filtered cells to >= {min_value}'
    else:
        msg = f'Filtered cells to <= {max_value}'

    return {
        'success': True,
        'operation': 'filter_cells',
        'vega_spec': new_spec,
        'message': msg
    }


def highlight_region(
    vega_spec: Dict,
    x_values: Optional[List] = None,
    y_values: Optional[List] = None,
) -> Dict[str, Any]:
    """
    高亮区域。x_values 或 y_values 至少提供一个：
    - 仅 x_values：高亮整列（该 x 轴下的所有 y）
    - 仅 y_values：高亮整行（该 y 轴下的所有 x）
    - 两者都提供：高亮交叉区域
    """
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    x_timeunit = new_spec.get('encoding', {}).get('x', {}).get('timeUnit')
    y_timeunit = new_spec.get('encoding', {}).get('y', {}).get('timeUnit')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x/y fields'}

    x_vals = x_values if x_values is not None else []
    y_vals = y_values if y_values is not None else []
    if not x_vals and not y_vals:
        return {'success': False, 'error': 'Must provide at least one of x_values or y_values'}

    # Month name map (Vega month(): 0=Jan ... 11=Dec)
    MONTH_MAP = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11,
        "January": 0, "February": 1, "March": 2, "April": 3, "June": 5,
        "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
    }

    def _normalize_values_for_timeunit(field_name: str, values: List, timeunit: Optional[str]) -> Tuple[str, str]:
        """
        Returns (value_list_str, expr) where expr is the datum-side expression to test.
        value_list_str is already formatted for use inside [ ... ].
        """
        if not timeunit:
            # Treat as raw field value (string compare)
            value_list_str = ','.join([f'"{v}"' for v in values])
            return value_list_str, f'datum["{field_name}"]'

        tu = str(timeunit).lower().strip()
        if tu == 'date':
            nums = []
            for v in values:
                try:
                    nums.append(str(int(v)))
                except Exception:
                    pass
            value_list_str = ','.join(nums) if nums else ','.join([f'"{v}"' for v in values])
            return value_list_str, f'date(datum["{field_name}"])'
        if tu == 'month':
            months = []
            for v in values:
                if isinstance(v, str) and v in MONTH_MAP:
                    months.append(str(MONTH_MAP[v]))
                else:
                    try:
                        m = int(v)
                        # user-facing month is 1-12; vega month() is 0-11
                        months.append(str(m - 1))
                    except Exception:
                        pass
            value_list_str = ','.join(months) if months else ','.join([f'"{v}"' for v in values])
            return value_list_str, f'month(datum["{field_name}"])'
        if tu == 'year':
            years = []
            for v in values:
                try:
                    years.append(str(int(v)))
                except Exception:
                    pass
            value_list_str = ','.join(years) if years else ','.join([f'"{v}"' for v in values])
            return value_list_str, f'year(datum["{field_name}"])'

        # other timeUnit fall back
        value_list_str = ','.join([f'"{v}"' for v in values])
        return value_list_str, f'{tu}(datum["{field_name}"])'

    # Build expressions for x/y (support temporal+timeUnit); only for axes with values
    parts: List[str] = []
    if x_vals:
        x_list, x_expr = _normalize_values_for_timeunit(x_field, x_vals, x_timeunit)
        parts.append(f'indexof([{x_list}], {x_expr}) >= 0')
    if y_vals:
        y_list, y_expr = _normalize_values_for_timeunit(y_field, y_vals, y_timeunit)
        parts.append(f'indexof([{y_list}], {y_expr}) >= 0')
    test_expr = ' && '.join(parts)
    
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': test_expr,
            'value': 1.0
        },
        # 未选中格子更淡，提升区分度
        'value': 0.15
    }
    
    return {
        'success': True,
        'operation': 'highlight_region',
        'vega_spec': new_spec,
        'message': 'Highlighted specified region'
    }


def highlight_region_by_value(
    vega_spec: Dict,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    outside_opacity: float = 0.12,
) -> Dict[str, Any]:
    """
    按“格子显示值”（通常是 color 编码后的值/聚合值）高亮：范围内不变，范围外变淡；不删除数据。

    说明：
    - 这是视觉高亮工具，不会追加 transform.filter，不会改变底层数据。
    - 支持单侧阈值：只传 min_value 或只传 max_value。
    """
    if min_value is None and max_value is None:
        return {'success': False, 'error': 'Must provide at least one of min_value or max_value'}

    new_spec = copy.deepcopy(vega_spec)

    color_enc = (new_spec.get('encoding', {}) or {}).get('color', {}) or {}
    color_field = color_enc.get('field')
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field'}

    # If color uses aggregation, Vega-Lite datum field is typically "<op>_<field>" unless "as" is set.
    agg = color_enc.get('aggregate')
    agg_as = color_enc.get('as')
    value_field = color_field
    if agg:
        if isinstance(agg_as, str) and agg_as.strip():
            value_field = agg_as.strip()
        else:
            value_field = f'{str(agg).lower()}_{color_field}'

    ref = _datum_ref(value_field)
    tests: List[str] = []
    if min_value is not None:
        tests.append(f'{ref} >= {float(min_value)}')
    if max_value is not None:
        tests.append(f'{ref} <= {float(max_value)}')
    test_expr = ' && '.join(tests) if tests else 'true'

    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': test_expr,
            'value': 1.0
        },
        'value': float(outside_opacity)
    }

    return {
        'success': True,
        'operation': 'highlight_region_by_value',
        'vega_spec': new_spec,
        'message': f'Highlighted cells by value (min={min_value}, max={max_value}); outside_opacity={outside_opacity}'
    }


def filter_cells_by_region(
    vega_spec: Dict,
    x_value: Any = None,
    y_value: Any = None,
    x_values: Optional[List[Any]] = None,
    y_values: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    按热力图格子坐标（x/y）过滤掉对应格子（通过 transform.filter 排除匹配的坐标）。
    x 或 y 至少提供一个：
    - 仅 x：过滤掉该 x 轴下的整列
    - 仅 y：过滤掉该 y 轴下的整行
    - 两者都提供：过滤掉交叉区域

    用法：
    - 单格子：传 x_value + y_value
    - 多格子（笛卡尔积）：传 x_values + y_values
    """
    new_spec = copy.deepcopy(vega_spec)

    x_field = (new_spec.get('encoding', {}) or {}).get('x', {}).get('field')
    y_field = (new_spec.get('encoding', {}) or {}).get('y', {}).get('field')
    x_timeunit = (new_spec.get('encoding', {}) or {}).get('x', {}).get('timeUnit')
    y_timeunit = (new_spec.get('encoding', {}) or {}).get('y', {}).get('timeUnit')
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x/y fields'}

    # normalize to lists
    if x_values is None and x_value is not None:
        x_values = [x_value]
    if y_values is None and y_value is not None:
        y_values = [y_value]
    if (not x_values or len(x_values) == 0) and (not y_values or len(y_values) == 0):
        return {'success': False, 'error': 'Must provide at least one of x_value/x_values or y_value/y_values'}

    # Reuse the same normalization logic as highlight_region for timeUnit (month/year/date)
    MONTH_MAP = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11,
        "January": 0, "February": 1, "March": 2, "April": 3, "June": 5,
        "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11
    }

    def _normalize_values_for_timeunit(field_name: str, values: List[Any], timeunit: Optional[str]) -> Tuple[str, str]:
        if not timeunit:
            value_list_str = ','.join([json.dumps(v, ensure_ascii=False) for v in values])
            return value_list_str, _datum_ref(field_name)

        tu = str(timeunit).lower().strip()
        if tu == 'date':
            nums = []
            for v in values:
                try:
                    nums.append(str(int(v)))
                except Exception:
                    pass
            value_list_str = ','.join(nums) if nums else ','.join([json.dumps(v, ensure_ascii=False) for v in values])
            return value_list_str, f'date({_datum_ref(field_name)})'
        if tu == 'month':
            months = []
            for v in values:
                if isinstance(v, str) and v in MONTH_MAP:
                    months.append(str(MONTH_MAP[v]))
                else:
                    try:
                        m = int(v)
                        months.append(str(m - 1))
                    except Exception:
                        pass
            value_list_str = ','.join(months) if months else ','.join([json.dumps(v, ensure_ascii=False) for v in values])
            return value_list_str, f'month({_datum_ref(field_name)})'
        if tu == 'year':
            years = []
            for v in values:
                try:
                    years.append(str(int(v)))
                except Exception:
                    pass
            value_list_str = ','.join(years) if years else ','.join([json.dumps(v, ensure_ascii=False) for v in values])
            return value_list_str, f'year({_datum_ref(field_name)})'

        value_list_str = ','.join([json.dumps(v, ensure_ascii=False) for v in values])
        return value_list_str, f'{tu}({_datum_ref(field_name)})'

    exclude_parts: List[str] = []
    if x_values and len(x_values) > 0:
        x_list, x_expr = _normalize_values_for_timeunit(x_field, list(x_values), x_timeunit)
        exclude_parts.append(f'indexof([{x_list}], {x_expr}) >= 0')
    if y_values and len(y_values) > 0:
        y_list, y_expr = _normalize_values_for_timeunit(y_field, list(y_values), y_timeunit)
        exclude_parts.append(f'indexof([{y_list}], {y_expr}) >= 0')
    exclude_expr = ' && '.join(exclude_parts)

    if 'transform' not in new_spec:
        new_spec['transform'] = []
    new_spec['transform'].append({'filter': f'!({exclude_expr})', '_avs_tag': 'filter_cells_by_region'})

    return {
        'success': True,
        'operation': 'filter_cells_by_region',
        'vega_spec': new_spec,
        'message': f'Filtered out selected region cells (x={x_values}, y={y_values})'
    }


def cluster_rows_cols(vega_spec: Dict, cluster_rows: bool = True, 
                     cluster_cols: bool = True, method: str = "sum") -> Dict[str, Any]:
    """
    对热力图的行/列按数值聚合结果重新排序（实现逻辑与说明）
    
    实现逻辑：
    - 使用 encoding.color.field 作为度量字段（即热力图着色的数值）。
    - 对行（Y 轴）：按 color 字段在每行上的聚合（sum/mean/max）降序排序；
      对列（X 轴）：按 color 字段在每列上的聚合降序排序。
    - 通过设置 encoding.y.sort / encoding.x.sort 为 { op, field, order } 实现，
      Vega-Lite 自动按指定 op 对 field 聚合后排序，使高值行/列靠前，便于发现热点。
    
    说明：实现的是“按行/列聚合排序”，而非严格聚类算法；效果类似行列重排，使高值区域更集中。
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if 'encoding' not in new_spec:
        return {'success': False, 'error': 'No encoding found'}
    
    encoding = new_spec['encoding']
    color_field = encoding.get('color', {}).get('field')
    
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field for sorting'}
    
    if method == "sum":
        sort_op = "sum"
    elif method == "mean":
        sort_op = "mean"
    elif method == "max":
        sort_op = "max"
    else:
        sort_op = "sum"
    
    if cluster_rows and 'y' in encoding:
        encoding['y']['sort'] = {
            'op': sort_op,
            'field': color_field,
            'order': 'descending'
        }
    
    if cluster_cols and 'x' in encoding:
        encoding['x']['sort'] = {
            'op': sort_op,
            'field': color_field,
            'order': 'descending'
        }
    
    return {
        'success': True,
        'operation': 'cluster_rows_cols',
        'vega_spec': new_spec,
        'message': f'Sorted rows={cluster_rows}, cols={cluster_cols} by {method}'
    }


def select_submatrix(vega_spec: Dict, x_values: List = None, 
                    y_values: List = None) -> Dict[str, Any]:
    """选择子矩阵"""
    if not x_values and not y_values:
        return {'success': False, 'error': 'Must specify x_values or y_values'}
    
    new_spec = copy.deepcopy(vega_spec)
    
    # 月份名称到数字的映射 (Vega month 从 0 开始: 0=Jan, 11=Dec)
    MONTH_MAP = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
        "May": 4, "Jun": 5, "Jul": 6, "Aug": 7,
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11,
        "January": 0, "February": 1, "March": 2, "April": 3,
        "May": 4, "June": 5, "July": 6, "August": 7,
        "September": 8, "October": 9, "November": 10, "December": 11
    }
    
    encoding = new_spec.get('encoding', {})
    x_encoding = encoding.get('x', {})
    y_encoding = encoding.get('y', {})
    
    x_field = x_encoding.get('field')
    y_field = y_encoding.get('field')
    x_timeunit = x_encoding.get('timeUnit')
    y_timeunit = y_encoding.get('timeUnit')
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    filters = []
    
    # 处理 X 轴过滤
    if x_values and x_field:
        if x_timeunit:
            # 有 timeUnit，使用 Vega 表达式函数
            if x_timeunit == 'date':
                # 提取日期（1-31）
                x_nums = ','.join([str(int(v)) for v in x_values])
                filters.append(f'indexof([{x_nums}], date(datum.{x_field})) >= 0')
            elif x_timeunit == 'month':
                # 提取月份，尝试转换月份名称为数字
                x_months = []
                for v in x_values:
                    if v in MONTH_MAP:
                        x_months.append(str(MONTH_MAP[v]))
                    else:
                        try:
                            x_months.append(str(int(v)))
                        except:
                            x_months.append(f'"{v}"')
                x_str = ','.join(x_months)
                filters.append(f'indexof([{x_str}], month(datum.{x_field})) >= 0')
            elif x_timeunit == 'year':
                x_nums = ','.join([str(int(v)) for v in x_values])
                filters.append(f'indexof([{x_nums}], year(datum.{x_field})) >= 0')
            else:
                # 其他 timeUnit，直接使用函数名
                x_str = ','.join([f'"{v}"' for v in x_values])
                filters.append(f'indexof([{x_str}], {x_timeunit}(datum.{x_field})) >= 0')
        else:
            # 没有 timeUnit，直接匹配字段值
            x_str = ','.join([f'"{v}"' for v in x_values])
            filters.append(f'indexof([{x_str}], datum.{x_field}) >= 0')
    
    # 处理 Y 轴过滤
    if y_values and y_field:
        if y_timeunit:
            # 有 timeUnit，使用 Vega 表达式函数
            if y_timeunit == 'date':
                y_nums = ','.join([str(int(v)) for v in y_values])
                filters.append(f'indexof([{y_nums}], date(datum.{y_field})) >= 0')
            elif y_timeunit == 'month':
                # 提取月份，尝试转换月份名称为数字
                y_months = []
                for v in y_values:
                    if v in MONTH_MAP:
                        y_months.append(str(MONTH_MAP[v]))
                    else:
                        try:
                            y_months.append(str(int(v)))
                        except:
                            y_months.append(f'"{v}"')
                y_str = ','.join(y_months)
                filters.append(f'indexof([{y_str}], month(datum.{y_field})) >= 0')
            elif y_timeunit == 'year':
                y_nums = ','.join([str(int(v)) for v in y_values])
                filters.append(f'indexof([{y_nums}], year(datum.{y_field})) >= 0')
            else:
                # 其他 timeUnit，直接使用函数名
                y_str = ','.join([f'"{v}"' for v in y_values])
                filters.append(f'indexof([{y_str}], {y_timeunit}(datum.{y_field})) >= 0')
        else:
            # 没有 timeUnit，直接匹配字段值
            y_str = ','.join([f'"{v}"' for v in y_values])
            filters.append(f'indexof([{y_str}], datum.{y_field}) >= 0')
    
    if filters:
        new_spec['transform'].append({
            'filter': ' && '.join(filters)
        })
    
    return {
        'success': True,
        'operation': 'select_submatrix',
        'vega_spec': new_spec,
        'message': f'Selected submatrix with {len(x_values) if x_values else "all"} cols, {len(y_values) if y_values else "all"} rows'
    }


def find_extremes(vega_spec: Dict, top_n: int = 5, mode: str = "both") -> Dict[str, Any]:
    """
    标记极值点位置
    
    Args:
        vega_spec: Vega-Lite规范
        top_n: 标记前N个极值
        mode: "max" | "min" | "both"
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # 获取字段信息
    encoding = new_spec.get('encoding', {})
    x_field = encoding.get('x', {}).get('field')
    y_field = encoding.get('y', {}).get('field')
    color_field = encoding.get('color', {}).get('field')
    
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field for finding extremes'}
    
    # 获取数据
    data = new_spec.get('data', {}).get('values', [])
    if not data:
        return {'success': False, 'error': 'No data found'}
    
    # 聚合为格子值（按 x/y 分组）
    agg_op = (encoding.get('color', {}) or {}).get('aggregate')
    agg_op = str(agg_op).lower().strip() if agg_op else 'mean'
    allowed = {"mean", "sum", "max", "min", "median", "count"}
    if agg_op not in allowed:
        agg_op = 'mean'
    
    grouped = {}
    for d in data:
        if d.get(color_field) is None:
            continue
        x_val = d.get(x_field)
        y_val = d.get(y_field)
        if x_val is None or y_val is None:
            continue
        key = (x_val, y_val)
        grouped.setdefault(key, []).append(d.get(color_field))
    
    def _aggregate(vals: List[Any]) -> float:
        nums = [v for v in vals if isinstance(v, (int, float))]
        if not nums:
            return 0.0
        if agg_op == 'sum':
            return float(sum(nums))
        if agg_op == 'max':
            return float(max(nums))
        if agg_op == 'min':
            return float(min(nums))
        if agg_op == 'median':
            s = sorted(nums)
            mid = len(s) // 2
            return float(s[mid]) if len(s) % 2 == 1 else float((s[mid - 1] + s[mid]) / 2)
        if agg_op == 'count':
            return float(len(nums))
        # mean (default)
        return float(sum(nums) / len(nums))
    
    aggregated = [
        {x_field: k[0], y_field: k[1], color_field: _aggregate(v)}
        for k, v in grouped.items()
    ]
    if not aggregated:
        return {'success': False, 'error': 'No aggregated values found'}
    
    # 按聚合后的格子值排序找极值
    sorted_data = sorted(
        aggregated,
        key=lambda x: x.get(color_field, 0)
    )
    
    extremes = []
    if mode in ["max", "both"]:
        extremes.extend(sorted_data[-top_n:])
    if mode in ["min", "both"]:
        extremes.extend(sorted_data[:top_n])
    
    # 构建极值点的坐标条件（支持含空格的字段名）
    dx, dy = _datum_ref(x_field), _datum_ref(y_field)
    extreme_conditions = []
    for e in extremes:
        x_val = e.get(x_field)
        y_val = e.get(y_field)
        if x_val is not None and y_val is not None:
            xq = json.dumps(x_val) if isinstance(x_val, str) else x_val
            yq = json.dumps(y_val) if isinstance(y_val, str) else y_val
            extreme_conditions.append(f'({dx} === {xq} && {dy} === {yq})')
    
    if not extreme_conditions:
        return {'success': False, 'error': 'No extreme values found'}
    
    test_expr = ' || '.join(extreme_conditions)
    # 使用 transparent 替代 null，避免 Vega 信号名问题
    new_spec['encoding']['stroke'] = {
        'condition': {'test': test_expr, 'value': 'red'},
        'value': 'transparent'
    }
    new_spec['encoding']['strokeWidth'] = {
        'condition': {
            'test': test_expr,
            'value': 3
        },
        'value': 0
    }
    
    # 返回极值信息
    extreme_info = []
    for e in extremes:
        extreme_info.append({
            'x': e.get(x_field),
            'y': e.get(y_field),
            'value': e.get(color_field),
            'aggregate': agg_op
        })
    
    return {
        'success': True,
        'operation': 'find_extremes',
        'vega_spec': new_spec,
        'extremes': extreme_info,
        'message': f'Marked {len(extremes)} extreme points (mode: {mode})'
    }


def threshold_mask(
    vega_spec: Dict,
    min_value: float,
    max_value: float,
    outside_opacity: float = 0.1,
) -> Dict[str, Any]:
    """
    对不在阈值范围内的单元格做“遮罩”（变淡），但不删除数据。
    
    Args:
        vega_spec: Vega-Lite规范
        min_value: 下阈值（包含）
        max_value: 上阈值（包含）
        outside_opacity: 范围外的透明度
    """
    new_spec = copy.deepcopy(vega_spec)

    color_enc = new_spec.get('encoding', {}).get('color', {})
    color_field = color_enc.get('field')
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field'}

    # If color uses aggregation, the datum field is typically "<op>_<field>" unless "as" is set.
    agg = color_enc.get('aggregate')
    agg_as = color_enc.get('as')
    value_field = color_field
    if agg:
        if isinstance(agg_as, str) and agg_as.strip():
            value_field = agg_as.strip()
        else:
            value_field = f'{str(agg).lower()}_{color_field}'

    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}

    ref = _datum_ref(value_field)
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': f'{ref} >= {min_value} && {ref} <= {max_value}',
            'value': 1.0
        },
        'value': float(outside_opacity)
    }

    return {
        'success': True,
        'operation': 'threshold_mask',
        'vega_spec': new_spec,
        'message': f'Applied threshold mask on {value_field} in [{min_value}, {max_value}]'
    }


def drilldown_time(
    vega_spec: Dict,
    level: str,
    value: Union[int, str],
    parent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    时间热力图下钻：年 -> 月 -> 日
    
    约定：
    - 时间字段在 encoding.x.field，且 encoding.x.type == 'temporal'
    - 初始建议 timeUnit='year'（若缺省，也可被记录并在 reset 时恢复）
    
    Args:
        vega_spec: Vega-Lite规范
        level: 'year' | 'month' | 'date'
        value: 对应 level 的值（year=int；month=1-12；date=1-31）
        parent: 可选父级信息，如 {'year': 2012} 或 {'year':2012,'month':3}
    """
    new_spec = copy.deepcopy(vega_spec)

    encoding = new_spec.get('encoding', {})
    x_enc = encoding.get('x', {})
    time_field = x_enc.get('field')
    x_type = x_enc.get('type')

    if not time_field:
        return {'success': False, 'error': 'Cannot find temporal x field for drilldown'}
    if x_type and x_type != 'temporal':
        return {'success': False, 'error': f'Expected encoding.x.type=temporal, got {x_type}'}

    # init state
    state = new_spec.get('_heatmap_state')
    if not isinstance(state, dict):
        state = {}

    if 'original_x_encoding' not in state:
        state['original_x_encoding'] = copy.deepcopy(x_enc)

    if 'transform' not in new_spec:
        new_spec['transform'] = []

    # remove existing drilldown filters (idempotent drilldown)
    new_spec['transform'] = [
        t for t in new_spec['transform']
        if not (isinstance(t, dict) and t.get('_avs_tag') == 'heatmap_drilldown_time')
    ]

    # parent merge (explicit parent > stored state)
    p = {}
    if isinstance(state.get('parent'), dict):
        p.update(state.get('parent'))
    if isinstance(parent, dict):
        p.update(parent)

    # normalize level
    level = str(level).lower().strip()

    # build filter + new timeUnit
    filters: List[str] = []
    next_timeunit: Optional[str] = None

    if level == 'year':
        try:
            year_val = int(value)
        except Exception:
            return {'success': False, 'error': f'Invalid year value: {value}'}

        state['parent'] = {'year': year_val}
        next_timeunit = 'month'
        filters.append(f'year(datum.{time_field}) == {year_val}')

    elif level == 'month':
        # month drilldown needs year
        year_val = p.get('year')
        if year_val is None:
            return {'success': False, 'error': 'Month drilldown requires parent.year'}
        try:
            year_val = int(year_val)
            month_val = int(value)
        except Exception:
            return {'success': False, 'error': f'Invalid month drilldown values: year={year_val}, month={value}'}

        state['parent'] = {'year': year_val, 'month': month_val}
        next_timeunit = 'date'
        # Vega-Lite month() returns 0-11, so month 1-12 => compare month()==month-1
        filters.append(f'year(datum.{time_field}) == {year_val}')
        filters.append(f'month(datum.{time_field}) == {month_val - 1}')

    elif level == 'date':
        # date drilldown needs year+month
        year_val = p.get('year')
        month_val = p.get('month')
        if year_val is None or month_val is None:
            return {'success': False, 'error': 'Date drilldown requires parent.year and parent.month'}
        try:
            year_val = int(year_val)
            month_val = int(month_val)
            date_val = int(value)
        except Exception:
            return {'success': False, 'error': f'Invalid date drilldown values: {p}, date={value}'}

        state['parent'] = {'year': year_val, 'month': month_val, 'date': date_val}
        next_timeunit = 'date'
        filters.append(f'year(datum.{time_field}) == {year_val}')
        filters.append(f'month(datum.{time_field}) == {month_val - 1}')
        filters.append(f'date(datum.{time_field}) == {date_val}')

    else:
        return {'success': False, 'error': f'Unsupported level: {level}. Use year|month|date'}

    # apply timeUnit on x
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    if 'x' not in new_spec['encoding']:
        new_spec['encoding']['x'] = {}
    if next_timeunit:
        new_spec['encoding']['x']['timeUnit'] = next_timeunit
        new_spec['encoding']['x']['type'] = 'temporal'
        new_spec['encoding']['x']['field'] = time_field

    # add filter transform
    if filters:
        new_spec['transform'].append({
            'filter': ' && '.join(filters),
            '_avs_tag': 'heatmap_drilldown_time'
        })

    new_spec['_heatmap_state'] = state

    return {
        'success': True,
        'operation': 'drilldown_time',
        'vega_spec': new_spec,
        'message': f'Drilldown to {level}={value}',
        'state': state.get('parent')
    }


def reset_drilldown(vega_spec: Dict) -> Dict[str, Any]:
    """
    重置时间热力图下钻：移除 drilldown_time 添加的 filter，并恢复原始 x 编码（timeUnit 等）。
    """
    new_spec = copy.deepcopy(vega_spec)

    state = new_spec.get('_heatmap_state')
    original_x = None
    if isinstance(state, dict):
        original_x = state.get('original_x_encoding')

    if 'transform' in new_spec and isinstance(new_spec['transform'], list):
        new_spec['transform'] = [
            t for t in new_spec['transform']
            if not (isinstance(t, dict) and t.get('_avs_tag') == 'heatmap_drilldown_time')
        ]

    if original_x and isinstance(original_x, dict):
        if 'encoding' not in new_spec:
            new_spec['encoding'] = {}
        new_spec['encoding']['x'] = copy.deepcopy(original_x)

    # clear state
    if '_heatmap_state' in new_spec:
        del new_spec['_heatmap_state']

    return {
        'success': True,
        'operation': 'reset_drilldown',
        'vega_spec': new_spec,
        'message': 'Reset heatmap drilldown to original state'
    }


# ============================================================================
# Marginal bars (interaction necessity)
# - Add top (column) and right (row) marginal bar charts to a heatmap
# - Default aggregation: mean (as requested)
# ============================================================================

def add_marginal_bars(
    vega_spec: Dict,
    op: str = "mean",
    show_top: bool = True,
    show_right: bool = True,
    bar_size: int = 70,
    bar_color: str = "#666666",
) -> Dict[str, Any]:
    """
    为热力图添加边际条形图（行/列聚合），默认使用均值（mean）。

    实现逻辑：
    - 复用热力图的 data、transform、config；按 encoding.x / encoding.y、encoding.color 取 x 轴、y 轴、数值字段。
    - 顶部边际：按 x 分组，对 color 字段做 op 聚合（mean/sum/median 等），竖条；与主图 x 对齐（vconcat + resolve scale x shared）。
    - 右侧边际：按 y 分组，对 color 字段做 op 聚合，横条；与主图 y 对齐（hconcat + resolve scale y shared）。
    - 若 spec 未设置 width/height，则使用默认 400×300，避免 concat 子图尺寸缺失导致空白。

    说明：纯热力图难以快速比较“哪一行/列总体更高”；边际条提供行/列聚合视角，默认 mean。
    """
    if not show_top and not show_right:
        return {'success': False, 'error': 'At least one of show_top/show_right must be True'}

    new_spec = copy.deepcopy(vega_spec)
    encoding = new_spec.get("encoding", {}) or {}
    x_enc = encoding.get("x", {}) or {}
    y_enc = encoding.get("y", {}) or {}
    c_enc = encoding.get("color", {}) or {}

    x_field = x_enc.get("field")
    y_field = y_enc.get("field")
    value_field = c_enc.get("field")
    if not x_field or not y_field or not value_field:
        return {'success': False, 'error': 'Cannot find required fields in encoding.x/encoding.y/encoding.color'}

    agg = str(op).lower().strip()
    allowed = {"mean", "sum", "median", "max", "min", "count"}
    if agg not in allowed:
        return {'success': False, 'error': f'Unsupported op: {op}. Use one of {sorted(list(allowed))}'}

    main = copy.deepcopy(new_spec)
    title = main.pop("title", None)

    default_w, default_h = 400, 300
    mw = main.get("width") if isinstance(main.get("width"), (int, float)) else None
    mh = main.get("height") if isinstance(main.get("height"), (int, float)) else None
    if mw is None:
        main["width"] = default_w
        mw = default_w
    if mh is None:
        main["height"] = default_h
        mh = default_h

    base_data = main.get("data")
    base_transform = main.get("transform")
    base_config = main.get("config")

    def _base_block() -> Dict[str, Any]:
        block: Dict[str, Any] = {}
        if base_data is not None:
            block["data"] = copy.deepcopy(base_data)
        if base_transform is not None:
            block["transform"] = copy.deepcopy(base_transform)
        if base_config is not None:
            block["config"] = copy.deepcopy(base_config)
        return block

    top_spec = None
    if show_top:
        top_spec = {
            **_base_block(),
            "mark": {"type": "bar", "color": bar_color},
            "encoding": {
                "x": copy.deepcopy(x_enc),
                "y": {"aggregate": agg, "field": value_field, "type": "quantitative", "title": None},
                "tooltip": [
                    {"field": x_field, **({"timeUnit": x_enc.get("timeUnit")} if x_enc.get("timeUnit") else {}), "type": x_enc.get("type", "nominal"), "title": x_enc.get("title", x_field)},
                    {"aggregate": agg, "field": value_field, "type": "quantitative", "title": f"{agg}({value_field})"},
                ],
            },
            "height": int(bar_size),
            "width": int(mw),
        }
        top_spec["encoding"]["x"]["axis"] = {"labels": False, "ticks": False, "title": None, "domain": False}
        top_spec["encoding"]["y"]["axis"] = {"grid": False, "ticks": False, "title": None}

    right_spec = None
    if show_right:
        right_spec = {
            **_base_block(),
            "mark": {"type": "bar", "color": bar_color},
            "encoding": {
                "y": copy.deepcopy(y_enc),
                "x": {"aggregate": agg, "field": value_field, "type": "quantitative", "title": None},
                "tooltip": [
                    {"field": y_field, **({"timeUnit": y_enc.get("timeUnit")} if y_enc.get("timeUnit") else {}), "type": y_enc.get("type", "nominal"), "title": y_enc.get("title", y_field)},
                    {"aggregate": agg, "field": value_field, "type": "quantitative", "title": f"{agg}({value_field})"},
                ],
            },
            "width": int(bar_size),
            "height": int(mh),
        }
        right_spec["encoding"]["y"]["axis"] = {"labels": False, "ticks": False, "title": None, "domain": False}
        right_spec["encoding"]["x"]["axis"] = {"grid": False, "ticks": False, "title": None}

    # Compose: vconcat(top, hconcat(main, right))
    # We want x shared between top and main; y shared between main and right.
    row = {
        "hconcat": [main] + ([right_spec] if right_spec else []),
        "resolve": {"scale": {"y": "shared"}},
    }
    composed: Dict[str, Any] = {
        "$schema": new_spec.get("$schema", "https://vega.github.io/schema/vega-lite/v5.json"),
        "vconcat": ([] if not top_spec else [top_spec]) + [row],
        "resolve": {"scale": {"x": "shared"}},
    }
    if title is not None:
        composed["title"] = title

    composed["_marginal_bars_state"] = {
        "enabled": True,
        "op": agg,
        "show_top": bool(show_top),
        "show_right": bool(show_right),
        "value_field": value_field,
        "x_field": x_field,
        "y_field": y_field,
        "updated_at": datetime.now().isoformat(),
    }

    return {
        "success": True,
        "operation": "add_marginal_bars",
        "vega_spec": composed,
        "message": f"Added marginal bars (op={agg}, top={show_top}, right={show_right})"
    }


def transpose(vega_spec: Dict) -> Dict[str, Any]:
    """
    热力图行列转置：交换 x 轴和 y 轴。
    
    交互必要性：
    - 有时数据的行列安排不符合分析习惯，需要快速切换视角
    - 比如把"按月看各地区"变成"按地区看各月"
    
    Args:
        vega_spec: Vega-Lite 规范
    
    Returns:
        转置后的规格
    """
    new_spec = copy.deepcopy(vega_spec)
    
    encoding = new_spec.get('encoding', {})
    x_enc = encoding.get('x')
    y_enc = encoding.get('y')
    
    if not x_enc or not y_enc:
        return {
            'success': False,
            'error': 'Cannot find both x and y encoding for transpose'
        }
    
    # 交换 x 和 y 编码
    new_spec['encoding']['x'] = copy.deepcopy(y_enc)
    new_spec['encoding']['y'] = copy.deepcopy(x_enc)
    
    # 交换 width 和 height（如果定义）
    width = new_spec.get('width')
    height = new_spec.get('height')
    if width is not None and height is not None:
        new_spec['width'] = height
        new_spec['height'] = width
    
    # 记录转置状态（用于切换回来）
    state = new_spec.get('_transpose_state', {'transposed': False})
    state['transposed'] = not state.get('transposed', False)
    new_spec['_transpose_state'] = state
    
    status = "transposed" if state['transposed'] else "restored"
    return {
        'success': True,
        'operation': 'transpose',
        'vega_spec': new_spec,
        'message': f'Heatmap {status}: x and y axes swapped'
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
    'adjust_color_scale',
    'filter_cells',
    'filter_cells_by_region',
    'highlight_region',
    'highlight_region_by_value',
    'cluster_rows_cols',
    'select_submatrix',
    'find_extremes',
    'threshold_mask',
    'drilldown_time',
    'reset_drilldown',
    'add_marginal_bars',
    'transpose',
    'change_encoding',
]
