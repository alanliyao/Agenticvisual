"""
散点图专用工具
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import copy
import json
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr


def _datum_ref(field: str) -> str:
    """Vega expr: datum access for field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"



def identify_clusters(vega_spec: Dict, n_clusters: int = 3, method: str = "kmeans") -> Dict[str, Any]:
    """识别数据聚类"""
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required fields'}
    
    data = new_spec.get('data', {}).get('values', [])
    
    points = []
    valid_indices = []
    for i, row in enumerate(data):
        if row.get(x_field) is not None and row.get(y_field) is not None:
            points.append([row[x_field], row[y_field]])
            valid_indices.append(i)
    
    if len(points) < n_clusters:
        return {'success': False, 'error': f'Not enough points for {n_clusters} clusters'}
    
    points_array = np.array(points)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points_array)
    centers = kmeans.cluster_centers_
    
    cluster_field = f'cluster_{n_clusters}'
    for i, label in enumerate(labels):
        data[valid_indices[i]][cluster_field] = int(label)
    
    new_spec['data']['values'] = data
    new_spec['encoding']['color'] = {
        'field': cluster_field,
        'type': 'nominal',
        'scale': {'scheme': 'category10'},
        'legend': {'title': 'Cluster'}
    }
    
    cluster_stats = []
    for i in range(n_clusters):
        cluster_points = points_array[labels == i]
        cluster_stats.append({
            'cluster_id': i,
            'size': len(cluster_points),
            'center': centers[i].tolist()
        })
    
    return {
        'success': True,
        'operation': 'identify_clusters',
        'vega_spec': new_spec,
        'n_clusters': n_clusters,
        'cluster_statistics': cluster_stats,
        'message': f'Identified {n_clusters} clusters'
    }


def calculate_correlation(vega_spec: Dict, method: str = "pearson") -> Dict[str, Any]:
    """计算相关系数
    
    如果之前使用了 select_region 或 brush_region 选中了区域，
    则只计算选中区域内的数据的相关系数。
    """
    x_field = vega_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = vega_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required fields'}
    
    data = vega_spec.get('data', {}).get('values', [])
    
    # 检查是否有选中区域（来自 select_region 或 brush_region）
    selected = vega_spec.get('_selected_region')
    region_info = ""
    if selected:
        x_min, x_max = selected['x_range']
        y_min, y_max = selected['y_range']
        data = [row for row in data 
                if row.get(x_field) is not None and row.get(y_field) is not None
                and x_min <= row[x_field] <= x_max
                and y_min <= row[y_field] <= y_max]
        region_info = f" (in selected region: {len(data)} points)"
    
    x_values = [row[x_field] for row in data if row.get(x_field) is not None and row.get(y_field) is not None]
    y_values = [row[y_field] for row in data if row.get(x_field) is not None and row.get(y_field) is not None]
    
    if len(x_values) < 2:
        return {'success': False, 'error': f'Not enough data points{region_info}'}
    
    x_array = np.array(x_values)
    y_array = np.array(y_values)
    
    if method == "pearson":
        correlation, p_value = pearsonr(x_array, y_array)
    elif method == "spearman":
        correlation, p_value = spearmanr(x_array, y_array)
    else:
        return {'success': False, 'error': f'Unsupported method: {method}'}
    
    strength = "strong" if abs(correlation) >= 0.7 else "moderate" if abs(correlation) >= 0.4 else "weak"
    direction = "positive" if correlation > 0 else "negative"
    
    return {
        'success': True,
        'operation': 'calculate_correlation',
        'method': method,
        'correlation_coefficient': float(correlation),
        'p_value': float(p_value),
        'strength': strength,
        'direction': direction,
        'data_points': len(x_values),
        'selected_region': selected is not None,
        'message': f'{method} correlation: {correlation:.3f} ({strength} {direction}){region_info}'
    }


def zoom_dense_area(vega_spec: Dict, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    """Zooms the specified view to a particular area by filtering data and adjusting axis scales.
    
    This focuses the visualization on a specific rectangular region.
    
    Args:
        vega_spec: The Vega-Lite specification
        x_range: Tuple of (min, max) for x-axis range
        y_range: Tuple of (min, max) for y-axis range
        
    Returns:
        Dict containing success status, filtered vega_spec, and statistics
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # Get field names
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required x or y fields'}
    
    # Get original data
    data = new_spec.get('data', {}).get('values', [])
    if not data:
        return {'success': False, 'error': 'No data found in specification'}
    
    original_count = len(data)
    
    # Filter data: only keep points within the specified range
    filtered_data = [
        point for point in data
        if (point.get(x_field) is not None and 
            point.get(y_field) is not None and
            x_range[0] <= point[x_field] <= x_range[1] and
            y_range[0] <= point[y_field] <= y_range[1])
    ]
    
    filtered_count = len(filtered_data)
    
    if filtered_count == 0:
        return {
            'success': False,
            'error': f'No data points found in range x:[{x_range[0]}, {x_range[1]}], y:[{y_range[0]}, {y_range[1]}]',
            'original_count': original_count,
            'filtered_count': 0
        }
    
    # Update data in spec
    new_spec['data']['values'] = filtered_data
    
    # Adjust axis scales to the specified range
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    
    for axis, vals in [('x', x_range), ('y', y_range)]:
        if axis not in new_spec['encoding']:
            new_spec['encoding'][axis] = {}
        if 'scale' not in new_spec['encoding'][axis]:
            new_spec['encoding'][axis]['scale'] = {}
        new_spec['encoding'][axis]['scale']['domain'] = [vals[0], vals[1]]
    
    return {
        'success': True,
        'operation': 'zoom_dense_area',
        'vega_spec': new_spec,
        'original_count': original_count,
        'filtered_count': filtered_count,
        'zoom_range': {
            'x': list(x_range),
            'y': list(y_range)
        },
        'message': f'Zoomed to dense area: showing {filtered_count} out of {original_count} points ({filtered_count/original_count*100:.1f}%)'
    }


def filter_categorical(vega_spec: Dict, categories_to_remove: List[str], field: str = None) -> Dict[str, Any]:
    """
    过滤掉指定类别的数据点
    
    Args:
        vega_spec: Vega-Lite规范
        categories_to_remove: 要移除的类别列表
        field: 分类字段名（可选，自动探测 color 字段）
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # 自动探测分类字段
    if field is None:
        encoding = new_spec.get('encoding', {})
        color_enc = encoding.get('color', {})
        field = color_enc.get('field')
        
        if not field:
            # 尝试从 shape 字段获取
            shape_enc = encoding.get('shape', {})
            field = shape_enc.get('field')
    
    if not field:
        return {
            'success': False,
            'error': 'Cannot find categorical field. Please specify field parameter.'
        }
    
    # 添加 filter transform
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    categories_json = json.dumps(categories_to_remove)
    new_spec['transform'].append({
        'filter': f'indexof({categories_json}, {_datum_ref(field)}) < 0'
    })
    
    return {
        'success': True,
        'operation': 'filter_categorical',
        'vega_spec': new_spec,
        'message': f'Filtered out categories: {categories_to_remove} from field {field}'
    }


def select_region(vega_spec: Dict, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    """
    选中指定区域内的点，区域内高亮、区域外变淡。
    
    后续调用 calculate_correlation 将只计算选中区域内的数据。
    
    Args:
        vega_spec: Vega-Lite 规范
        x_range: X 轴范围 (min, max)
        y_range: Y 轴范围 (min, max)
    """
    new_spec = copy.deepcopy(vega_spec)
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required fields'}
    data = new_spec.get('data', {}).get('values', [])
    selected_count = sum(
        1 for row in data
        if row.get(x_field) is not None and row.get(y_field) is not None
        and x_range[0] <= row[x_field] <= x_range[1]
        and y_range[0] <= row[y_field] <= y_range[1]
    )
    xr, yr = _datum_ref(x_field), _datum_ref(y_field)
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': f'{xr} >= {x_range[0]} && {xr} <= {x_range[1]} && {yr} >= {y_range[0]} && {yr} <= {y_range[1]}',
            'value': 1.0
        },
        'value': 0.2
    }
    # 保存选中区域元数据，供后续 calculate_correlation 使用
    new_spec['_selected_region'] = {
        'x_range': list(x_range),
        'y_range': list(y_range),
        'x_field': x_field,
        'y_field': y_field
    }
    return {
        'success': True,
        'operation': 'select_region',
        'vega_spec': new_spec,
        'selected_count': selected_count,
        'message': f'Selected {selected_count} points'
    }


def brush_region(vega_spec: Dict, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    """
    刷选特定区域，区域外数据点变淡
    
    后续调用 calculate_correlation 将只计算刷选区域内的数据。
    
    Args:
        vega_spec: Vega-Lite规范
        x_range: X轴范围 (min, max)
        y_range: Y轴范围 (min, max)
    """
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x or y fields'}
    
    # 统计刷选区域内的数据点数量
    data = new_spec.get('data', {}).get('values', [])
    brushed_count = sum(
        1 for row in data
        if row.get(x_field) is not None and row.get(y_field) is not None
        and x_range[0] <= row[x_field] <= x_range[1]
        and y_range[0] <= row[y_field] <= y_range[1]
    )
    
    # 通过 opacity 条件编码实现刷选效果（支持含空格的字段名）
    xr, yr = _datum_ref(x_field), _datum_ref(y_field)
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': f'{xr} >= {x_range[0]} && {xr} <= {x_range[1]} && {yr} >= {y_range[0]} && {yr} <= {y_range[1]}',
            'value': 1.0
        },
        'value': 0.15
    }
    
    # 保存刷选区域元数据，供后续 calculate_correlation 使用
    new_spec['_selected_region'] = {
        'x_range': list(x_range),
        'y_range': list(y_range),
        'x_field': x_field,
        'y_field': y_field
    }
    
    return {
        'success': True,
        'operation': 'brush_region',
        'vega_spec': new_spec,
        'brushed_count': brushed_count,
        'message': f'Brushed region x:[{x_range[0]}, {x_range[1]}], y:[{y_range[0]}, {y_range[1]}] ({brushed_count} points)'
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


def show_regression(vega_spec: Dict, method: str = "linear") -> Dict[str, Any]:
    """
    叠加回归线
    
    Args:
        vega_spec: Vega-Lite规范
        method: 回归方法 ("linear", "log", "exp", "poly", "quad")
    """
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x or y fields'}
    
    # 如果原规范没有 layer，转换为 layer 结构
    if 'layer' not in new_spec:
        original_spec = copy.deepcopy(new_spec)
        new_spec['layer'] = [{
            'mark': original_spec.get('mark', 'point'),
            'encoding': original_spec.get('encoding', {})
        }]
        # 移除顶层的 mark 和 encoding
        if 'mark' in new_spec:
            del new_spec['mark']
        if 'encoding' in new_spec:
            del new_spec['encoding']
    
    # 添加回归线图层
    regression_transform = {
        'regression': y_field,
        'on': x_field
    }
    
    # 根据方法设置回归参数
    if method == "poly":
        regression_transform['method'] = 'poly'
        regression_transform['order'] = 3
    elif method == "quad":
        regression_transform['method'] = 'poly'
        regression_transform['order'] = 2
    elif method in ["log", "exp"]:
        regression_transform['method'] = method
    else:
        regression_transform['method'] = 'linear'
    
    new_spec['layer'].append({
        'mark': {
            'type': 'line',
            'color': 'red',
            'strokeWidth': 2
        },
        'transform': [regression_transform],
        'encoding': {
            'x': {'field': x_field, 'type': 'quantitative'},
            'y': {'field': y_field, 'type': 'quantitative'}
        }
    })
    
    return {
        'success': True,
        'operation': 'show_regression',
        'vega_spec': new_spec,
        'message': f'Added {method} regression line'
    }



def _infer_field_type(data: List[Dict], field: str) -> str:
    """推断字段的 Vega-Lite 类型"""
    if not data:
        return 'nominal'
    
    for row in data:
        value = row.get(field)
        if value is not None:
            if isinstance(value, bool):
                return 'nominal'
            elif isinstance(value, (int, float)):
                return 'quantitative'
            elif isinstance(value, str):
                # 检查是否是日期格式
                if any(sep in value for sep in ['-', '/', ':']):
                    return 'temporal'
                return 'nominal'
    return 'nominal'


__all__ = [
    'identify_clusters',
    'calculate_correlation',
    'zoom_dense_area',
    'filter_categorical',
    'brush_region',
    'change_encoding',
    'show_regression',
]
