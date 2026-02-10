"""
平行坐标图专用工具（简化版 - 使用 vega_spec）
"""

from typing import Dict, Any, List, Union
import copy
import json



def reorder_dimensions(vega_spec: Dict, dimension_order: List[str]) -> Dict[str, Any]:
    """重新排序维度（支持 fold 格式和预归一化长格式）"""
    new_spec = copy.deepcopy(vega_spec)
    
    # 方法1: 基于 fold transform
    fold_transform = None
    fold_index = -1
    transforms = new_spec.get('transform', [])
    for i, transform in enumerate(transforms):
        if isinstance(transform, dict) and 'fold' in transform:
            fold_transform = transform
            fold_index = i
            break
    
    if fold_transform is not None:
        # 标准 fold 格式的平行坐标图
        current_fold = fold_transform['fold']
        if not isinstance(current_fold, list):
            return {'success': False, 'error': 'Fold transform does not contain a list'}
        
        missing_dims = [dim for dim in dimension_order if dim not in current_fold]
        if missing_dims:
            return {'success': False, 'error': f'Dimensions not found in fold: {missing_dims}'}
        
        extra_dims = [dim for dim in current_fold if dim not in dimension_order]
        if extra_dims:
            return {'success': False, 'error': f'Missing dimensions in dimension_order: {extra_dims}'}
        
        new_spec['transform'][fold_index]['fold'] = dimension_order
        
        # 更新 x 轴编码的 scale.domain
        def update_x_encoding_scale(obj):
            if isinstance(obj, dict):
                if 'encoding' in obj and isinstance(obj['encoding'], dict):
                    x_encoding = obj['encoding'].get('x')
                    if isinstance(x_encoding, dict) and x_encoding.get('field') == 'key':
                        if 'scale' not in x_encoding:
                            x_encoding['scale'] = {}
                        x_encoding['scale']['domain'] = dimension_order
                for value in obj.values():
                    update_x_encoding_scale(value)
            elif isinstance(obj, list):
                for item in obj:
                    update_x_encoding_scale(item)
        
        update_x_encoding_scale(new_spec)
    else:
        # 方法2: 预归一化长格式（无 fold，用 x.sort 或 x.scale.domain）
        def update_x_sort(obj):
            """更新 dimension 字段的 x 编码 sort"""
            if isinstance(obj, dict):
                if 'encoding' in obj and isinstance(obj['encoding'], dict):
                    x_encoding = obj['encoding'].get('x')
                    if isinstance(x_encoding, dict) and x_encoding.get('field') in ['dimension', 'key', 'variable']:
                        x_encoding['sort'] = dimension_order
                        if 'scale' in x_encoding:
                            x_encoding['scale']['domain'] = dimension_order
                for value in obj.values():
                    update_x_sort(value)
            elif isinstance(obj, list):
                for item in obj:
                    update_x_sort(item)
        
        update_x_sort(new_spec)
    
    return {
        'success': True,
        'operation': 'reorder_dimensions',
        'vega_spec': new_spec,
        'message': f'Reordered dimensions to {dimension_order}'
    }

def filter_dimension(vega_spec: Dict, dimension: str, range: List[float]) -> Dict[str, Any]:
    """Filter by dimension"""
    new_spec = copy.deepcopy(vega_spec)
    
    min_val, max_val = range
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    # find fold operation position
    fold_index = -1
    for i, transform in enumerate(new_spec['transform']):
        if isinstance(transform, dict) and 'fold' in transform:
            fold_index = i
            break
    
    # build filter expression (adapted for wide format, using square brackets to access field names)
    filter_expr = f"datum['{dimension}'] >= {min_val} && datum['{dimension}'] <= {max_val}"
    
    if fold_index >= 0:
        # insert filter before fold (using wide format)
        new_spec['transform'].insert(fold_index, {
            'filter': filter_expr
        })
    else:
        # if no fold, insert at beginning of transform array
        new_spec['transform'].insert(0, {
            'filter': filter_expr
    })
    
    return {
        'success': True,
        'operation': 'filter_dimension',
        'vega_spec': new_spec,
        'message': f'Filtered {dimension} to [{min_val}, {max_val}]'
    }



def filter_by_category(vega_spec: Dict, field: str, values: Union[str, List[str]]) -> Dict[str, Any]:
    """
    按分类字段筛选（在 fold 之前，使用宽格式）
    
    Args:
        vega_spec: Vega-Lite规范
        field: 分类字段名（如 "Species", "product", "region"）
        values: 要保留的值列表
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if not isinstance(values, list):
        values = [values]
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    # 找到 fold 操作的位置
    fold_index = -1
    for i, transform in enumerate(new_spec['transform']):
        if isinstance(transform, dict) and 'fold' in transform:
            fold_index = i
            break
    
    # 构建 filter 表达式（使用方括号语法处理含空格的字段名）
    values_str = ','.join([f'"{v}"' for v in values])
    filter_expr = f"indexof([{values_str}], datum['{field}']) < 0"
    
    if fold_index >= 0:
        # 在 fold 之前插入 filter（使用宽格式）
        new_spec['transform'].insert(fold_index, {
            'filter': filter_expr
        })
    else:
        # 如果没有 fold，在 transform 数组开头插入
        new_spec['transform'].insert(0, {
            'filter': filter_expr
        })
    
    return {
        'success': True,
        'operation': 'filter_by_category',
        'vega_spec': new_spec,
        'message': f'Filtered {field} to: {values}'
    }



def highlight_category(vega_spec: Dict, field: str, values: Union[str, List[str]]) -> Dict[str, Any]:
    """
    高亮指定类别，其他变暗
    
    Args:
        vega_spec: Vega-Lite规范
        field: 分类字段名（如 "Species", "product", "region"）
        values: 要高亮的值列表
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if not isinstance(values, list):
        values = [values]
    
    # 检测是否有 layer 结构
    if 'layer' in new_spec and isinstance(new_spec['layer'], list):
        # 找到包含 mark: "line" 的 layer
        line_layer_index = -1
        for i, layer in enumerate(new_spec['layer']):
            if isinstance(layer, dict):
                mark = layer.get('mark')
                if (isinstance(mark, dict) and mark.get('type') == 'line') or mark == 'line':
                    line_layer_index = i
                    break
        
        if line_layer_index >= 0:
            # 在该 layer 的 encoding 中添加或更新 opacity
            layer = new_spec['layer'][line_layer_index]
            if 'encoding' not in layer:
                layer['encoding'] = {}
            
            # 构建 opacity 条件（使用方括号语法处理含空格的字段名）
            values_json = json.dumps(values)
            layer['encoding']['opacity'] = {
                'condition': {
                    'test': f"indexof({values_json}, datum['{field}']) >= 0",
                    'value': 1.0
                },
                'value': 0.1
            }
        else:
            return {
                'success': False,
                'error': 'No line layer found in vega_spec'
            }
    else:
        # 如果没有 layer，在顶层 encoding 中添加（向后兼容）
        if 'encoding' not in new_spec:
            new_spec['encoding'] = {}
        
        values_json = json.dumps(values)
        new_spec['encoding']['opacity'] = {
            'condition': {
                'test': f"indexof({values_json}, datum['{field}']) >= 0",
                'value': 1.0
            },
            'value': 0.1
        }
    
    return {
        'success': True,
        'operation': 'highlight_category',
        'vega_spec': new_spec,
        'message': f'Highlighted {field}: {values}'
    }


def hide_dimensions(
    vega_spec: Dict,
    dimensions: List[str],
    mode: str = "hide",
) -> Dict[str, Any]:
    """
    隐藏/显示平行坐标图中的维度轴。
    
    交互必要性：
    - 平行坐标图维度太多时会很拥挤，需要临时隐藏不关心的维度
    - 可以按需恢复（show 模式）
    
    Args:
        vega_spec: Vega-Lite 规范
        dimensions: 要隐藏或显示的维度名称列表
        mode: "hide"（隐藏）或 "show"（显示），默认 hide
    
    Returns:
        修改后的规格
    """
    new_spec = copy.deepcopy(vega_spec)
    
    mode_lower = str(mode).lower().strip()
    if mode_lower not in ("hide", "show"):
        return {
            'success': False,
            'error': f'Invalid mode: {mode}. Use "hide" or "show"'
        }
    
    if not dimensions:
        return {
            'success': False,
            'error': 'dimensions list cannot be empty'
        }
    
    # 初始化或获取隐藏状态
    state = new_spec.get('_pc_hidden_state')
    if not isinstance(state, dict):
        state = {'hidden': [], 'all_dimensions': None}
    
    hidden_set = set(state.get('hidden', []))
    
    # 查找 transform 中的 fold 操作
    transforms = new_spec.get('transform', [])
    fold_index = -1
    fold_transform = None
    
    for i, t in enumerate(transforms):
        if isinstance(t, dict) and 'fold' in t:
            fold_index = i
            fold_transform = t
            break
    
    def _find_dimension_field(spec: Dict) -> Union[str, None]:
        enc = spec.get('encoding', {})
        x_enc = enc.get('x') if isinstance(enc, dict) else None
        if isinstance(x_enc, dict):
            field = x_enc.get('field')
            if isinstance(field, str) and field:
                return field
        for layer in spec.get('layer', []) if isinstance(spec.get('layer'), list) else []:
            if isinstance(layer, dict):
                layer_enc = layer.get('encoding', {})
                x_layer = layer_enc.get('x') if isinstance(layer_enc, dict) else None
                if isinstance(x_layer, dict):
                    field = x_layer.get('field')
                    if isinstance(field, str) and field:
                        return field
        return None

    def _collect_dimensions_from_values(values: List[Dict], field: str) -> List[str]:
        seen = set()
        ordered = []
        for row in values:
            if not isinstance(row, dict):
                continue
            val = row.get(field)
            if isinstance(val, str) and val not in seen:
                seen.add(val)
                ordered.append(val)
        return ordered

    def _find_all_dimensions(spec: Dict, field: str) -> List[str]:
        # 优先 x.sort 或 x.scale.domain
        enc = spec.get('encoding', {})
        x_enc = enc.get('x') if isinstance(enc, dict) else None
        if isinstance(x_enc, dict):
            sort_vals = x_enc.get('sort')
            if isinstance(sort_vals, list) and sort_vals:
                return list(sort_vals)
            scale_domain = (x_enc.get('scale') or {}).get('domain')
            if isinstance(scale_domain, list) and scale_domain:
                return list(scale_domain)
        # 其次从 layer data.values 中提取
        for layer in spec.get('layer', []) if isinstance(spec.get('layer'), list) else []:
            if isinstance(layer, dict):
                layer_data = layer.get('data', {})
                layer_values = layer_data.get('values') if isinstance(layer_data, dict) else None
                if isinstance(layer_values, list) and layer_values:
                    dims = _collect_dimensions_from_values(layer_values, field)
                    if dims:
                        return dims
        # 最后从主数据中提取
        data = spec.get('data', {})
        values = data.get('values') if isinstance(data, dict) else None
        if isinstance(values, list) and values:
            return _collect_dimensions_from_values(values, field)
        return []

    def _update_x_encodings(obj: Any, field: str, visible: List[str]) -> None:
        if isinstance(obj, dict):
            enc = obj.get('encoding')
            if isinstance(enc, dict):
                x_enc = enc.get('x')
                if isinstance(x_enc, dict) and x_enc.get('field') == field:
                    x_enc['sort'] = list(visible)
                    scale = x_enc.get('scale')
                    if not isinstance(scale, dict):
                        scale = {}
                    scale['domain'] = list(visible)
                    x_enc['scale'] = scale
            for value in obj.values():
                _update_x_encodings(value, field, visible)
        elif isinstance(obj, list):
            for item in obj:
                _update_x_encodings(item, field, visible)

    def _pick_dimension_field_from_data(spec: Dict) -> Union[str, None]:
        data = spec.get('data', {})
        values = data.get('values') if isinstance(data, dict) else None
        if isinstance(values, list) and values:
            sample = values[0]
            if isinstance(sample, dict):
                for candidate in ['dimension', 'key', 'variable']:
                    if candidate in sample:
                        return candidate
        return None

    def _filter_layer_dimension_values(spec: Dict, field: str, visible: List[str]) -> None:
        for layer in spec.get('layer', []) if isinstance(spec.get('layer'), list) else []:
            if not isinstance(layer, dict):
                continue
            layer_data = layer.get('data', {})
            layer_values = layer_data.get('values') if isinstance(layer_data, dict) else None
            if isinstance(layer_values, list) and layer_values:
                filtered = [
                    row for row in layer_values
                    if isinstance(row, dict) and row.get(field) in visible
                ]
                if filtered:
                    layer['data']['values'] = filtered

    if fold_index < 0 or fold_transform is None:
        # 长表结构：无 fold，基于 dimension 字段过滤
        dim_field = _find_dimension_field(new_spec)
        if not dim_field:
            dim_field = _pick_dimension_field_from_data(new_spec)
        all_dims = _find_all_dimensions(new_spec, dim_field) if dim_field else []
        if not dim_field or not all_dims:
            return {
                'success': False,
                'error': 'Cannot find dimension field or dimension list for non-fold parallel coordinates.'
            }
        if state.get('all_dimensions') is None:
            state['all_dimensions'] = list(all_dims)
        all_dims = state['all_dimensions']
    else:
        current_fold = list(fold_transform.get('fold', []))
        if state.get('all_dimensions') is None:
            state['all_dimensions'] = list(current_fold)
        all_dims = state['all_dimensions']
    
    if mode_lower == "hide":
        # 隐藏指定维度
        for dim in dimensions:
            hidden_set.add(dim)
        # 新的可见维度 = 原始维度 - 隐藏集合
        visible_dims = [d for d in all_dims if d not in hidden_set]
    else:
        # 显示指定维度（从隐藏集合中移除）
        for dim in dimensions:
            hidden_set.discard(dim)
        visible_dims = [d for d in all_dims if d not in hidden_set]
    
    if not visible_dims:
        return {
            'success': False,
            'error': 'Cannot hide all dimensions. At least one dimension must remain visible.'
        }
    
    if fold_index >= 0 and fold_transform is not None:
        # 更新 fold transform
        fold_transform['fold'] = visible_dims
        new_spec['transform'][fold_index] = fold_transform
    else:
        # 无 fold：更新 transform 过滤
        if 'transform' not in new_spec:
            new_spec['transform'] = []
        visible_json = json.dumps(visible_dims)
        ff = dim_field.replace("\\", "\\\\").replace("'", "\\'")
        filter_expr = f"indexof({visible_json}, datum['{ff}']) >= 0"
        updated = False
        for t in new_spec['transform']:
            if isinstance(t, dict) and t.get('_pc_hide_dimensions'):
                t['filter'] = filter_expr
                updated = True
                break
        if not updated:
            new_spec['transform'].insert(0, {
                'filter': filter_expr,
                '_pc_hide_dimensions': True
            })
        _update_x_encodings(new_spec, dim_field, visible_dims)
        _filter_layer_dimension_values(new_spec, dim_field, visible_dims)
    
    # 保存状态
    state['hidden'] = list(hidden_set)
    new_spec['_pc_hidden_state'] = state
    
    action = "Hidden" if mode_lower == "hide" else "Shown"
    return {
        'success': True,
        'operation': 'hide_dimensions',
        'vega_spec': new_spec,
        'hidden_dimensions': list(hidden_set),
        'visible_dimensions': visible_dims,
        'message': f'{action} dimensions: {dimensions}. Currently hidden: {list(hidden_set)}'
    }


def reset_hidden_dimensions(vega_spec: Dict) -> Dict[str, Any]:
    """
    重置所有隐藏的维度，恢复到全部可见状态。
    """
    new_spec = copy.deepcopy(vega_spec)
    
    state = new_spec.get('_pc_hidden_state')
    if not isinstance(state, dict) or state.get('all_dimensions') is None:
        return {
            'success': True,
            'operation': 'reset_hidden_dimensions',
            'vega_spec': new_spec,
            'message': 'No hidden dimensions to reset'
        }
    
    all_dims = state['all_dimensions']
    
    # 查找 fold transform
    transforms = new_spec.get('transform', [])
    for i, t in enumerate(transforms):
        if isinstance(t, dict) and 'fold' in t:
            t['fold'] = list(all_dims)
            new_spec['transform'][i] = t
            break
    
    # 清除状态
    if '_pc_hidden_state' in new_spec:
        del new_spec['_pc_hidden_state']
    
    return {
        'success': True,
        'operation': 'reset_hidden_dimensions',
        'vega_spec': new_spec,
        'message': f'Reset to show all {len(all_dims)} dimensions'
    }


__all__ = [
    'highlight_cluster',
    'reorder_dimensions',
    'filter_by_category',
    'highlight_category',
    'hide_dimensions',
    'reset_hidden_dimensions',
]
