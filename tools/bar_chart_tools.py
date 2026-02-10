"""
Bar chart tools
"""

from typing import List, Dict, Any, Optional, Tuple
import copy
import json
from pathlib import Path
from datetime import datetime


def _datum_ref(field: str) -> str:
    """Vega expr: datum access for field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"


import json
import copy
import pandas as pd

def sort_bars(vega_spec: dict, order: str = "descending", by_subcategory: str = None) -> dict:
    """
    针对带有聚合（如mean）和颜色分组的条形图进行排序的增强版函数。
    统一使用 Pandas 预计算 + 显式数组排序，兼容所有渲染环境。
    """
    new_spec = copy.deepcopy(vega_spec)
    enc = new_spec.get('encoding', {})

    # 1. 自动识别 X 和 Y 轴
    x_type = enc.get('x', {}).get('type', 'nominal')
    y_type = enc.get('y', {}).get('type', 'quantitative')

    if y_type in ['nominal', 'ordinal'] and x_type == 'quantitative':
        cat_channel = 'y'
        val_channel = 'x'
    else:
        cat_channel = 'x'
        val_channel = 'y'

    cat_field = enc.get(cat_channel, {}).get('field')
    val_field = enc.get(val_channel, {}).get('field')
    color_field = enc.get('color', {}).get('field')
    val_agg = enc.get(val_channel, {}).get('aggregate', 'mean')

    if not cat_field or not val_field:
        return {'success': False, 'error': '无法识别分类或数值字段'}

    # 2. 统一加载数据到 Pandas DataFrame（所有场景都走这条路）
    data_vals = new_spec.get('data', {}).get('values')
    if not data_vals:
        return {'success': False, 'error': 'Spec中未找到values数据，无法在后端计算排序'}

    df = pd.DataFrame(data_vals)

    # 3. 计算排序指标
    sort_scores = {}

    def apply_agg(series):
        if val_agg == 'mean': return series.mean()
        if val_agg == 'sum': return series.sum()
        if val_agg == 'count': return series.count()
        if val_agg == 'median': return series.median()
        if val_agg == 'min': return series.min()
        if val_agg == 'max': return series.max()
        return series.mean()

    if by_subcategory:
        # 【场景 A：按特定子类排序】
        if not color_field:
            return {'success': False, 'error': '未定义颜色字段，无法按子类排序'}

        df['color_str'] = df[color_field].astype(str)
        sub_df = df[df['color_str'] == str(by_subcategory)]
        grouped = sub_df.groupby(cat_field)[val_field].apply(apply_agg)
        sort_scores = grouped.to_dict()

    elif color_field:
        # 【场景 B：有颜色分组，按总值排序（堆叠高度）】
        g1 = df.groupby([cat_field, color_field])[val_field].apply(apply_agg)
        grouped = g1.groupby(cat_field).sum()
        sort_scores = grouped.to_dict()

    else:
        # 【场景 C：简单条形图（无颜色分组）】
        # ★ 修复点：不再用 EncodingSortField 对象做 early return，
        #   统一走 Pandas 预计算 + 显式数组，兼容所有渲染环境。
        grouped = df.groupby(cat_field)[val_field].apply(apply_agg)
        sort_scores = grouped.to_dict()

    # 4. 生成排序列表
    all_categories = df[cat_field].unique()
    is_descending = order.lower() in ['descending', 'desc']

    def get_score(cat):
        return sort_scores.get(cat, 0)

    sorted_categories = sorted(all_categories, key=get_score, reverse=is_descending)
    sorted_categories = [x.item() if hasattr(x, "item") else x for x in sorted_categories]

    # 5. 统一使用显式数组赋值给 sort（最稳定的方式）
    enc[cat_channel]['sort'] = sorted_categories

    return {'success': True, 'vega_spec': new_spec}




def filter_categories(vega_spec: Dict, categories: List[str]) -> Dict[str, Any]:
    """Filter specific categories"""
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    
    if not x_field:
        return {'success': False, 'error': 'Cannot find category field'}
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    category_str = ','.join([f'"{c}"' for c in categories])
    new_spec['transform'].append({
        'filter': f'indexof([{category_str}], {_datum_ref(x_field)}) < 0'
    })
    
    return {
        'success': True,
        'operation': 'filter_categories',
        'vega_spec': new_spec,
        'message': f'Filtered to {len(categories)} categories'
    }


def highlight_top_n(vega_spec: Dict, n: int = 5, order: str = "descending") -> Dict[str, Any]:
    """Highlight top N bars by aggregated value (supports stacked/grouped charts)"""
    from collections import defaultdict
    new_spec = copy.deepcopy(vega_spec)
    
    data = new_spec.get('data', {}).get('values', [])
    encoding = new_spec.get('encoding', {})
    y_enc = encoding.get('y', {})
    x_enc = encoding.get('x', {})
    y_field = y_enc.get('field')
    x_field = x_enc.get('field')
    
    if not y_field or not x_field or not data:
        return {'success': False, 'error': 'Missing data or x/y field'}
    
    # Aggregate by x_field (category) to handle stacked/grouped charts
    category_totals = defaultdict(float)
    for row in data:
        cat = row.get(x_field)
        val = row.get(y_field, 0)
        if cat is not None and isinstance(val, (int, float)):
            category_totals[cat] += val
    
    # Sort categories by aggregated total
    order_value = str(order).lower()
    descending_aliases = {"descending", "desc", "top", "high", "max", "maximum"}
    descending = order_value in descending_aliases
    sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=descending)
    top_categories = [cat for cat, _ in sorted_cats[:n]]
    
    if not top_categories:
        return {'success': False, 'error': 'No categories found'}
    
    # Build condition using x_field (category) instead of y_field (value)
    # Use bracket notation for field names (handles spaces and special chars)
    def quote(v):
        return f"'{v}'" if isinstance(v, str) else str(v)
    test_expr = ' || '.join([f"datum['{x_field}'] == {quote(c)}" for c in top_categories])
    
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': test_expr,
            'value': 1.0
        },
        'value': 0.3
    }
    
    return {
        'success': True,
        'operation': 'highlight_top_n',
        'vega_spec': new_spec,
        'message': f'Highlighted top {n} categories: {top_categories}'
    }




def expand_stack(vega_spec: Dict, category: str) -> Dict[str, Any]:
    """
    Expand stacked bar chart of a specific category into a parallel bar chart.
    
    Deconstruct the stacked parts of the bar chart into independent bars, making it easier to compare the values of each subcategory within the same category.
    This is a "physical interaction necessity" tool - the baseline of the middle layer of the stacked chart is different, making it difficult to compare sizes directly.
    
    Parameters:
        vega_spec: Vega spec
        category: x axis category name to expand (e.g. "East China")
    
    Returns:
        expanded parallel bar chart spec
    """
    new_spec = copy.deepcopy(vega_spec)
    encoding = new_spec.get('encoding', {})
    
    # get x axis field and color field
    x_field = encoding.get('x', {}).get('field')
    color_enc = encoding.get('color', {})
    color_field = color_enc.get('field')
    y_enc = encoding.get('y', {})
    
    if not x_field:
        return {'success': False, 'error': 'Cannot find x axis field'}
    
    if not color_field:
        return {'success': False, 'error': 'This is not a stacked bar chart (missing color encoding)'}
    
    # 1. add filter to only keep data for specified category (support spaces in field names)
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    q = json.dumps(category)
    new_spec['transform'].append({
        'filter': f'{_datum_ref(x_field)} == {q}'
    })
    
    # 2. move original color field to x axis
    original_x_enc = encoding.get('x', {})
    new_spec['encoding']['x'] = {
        'field': color_field,
        'type': 'nominal',
        'title': color_enc.get('title', color_field),
        'axis': {'labelAngle': -45}
    }
    # keep original x axis scale settings if any
    if 'scale' in color_enc:
        new_spec['encoding']['x']['sort'] = color_enc['scale'].get('domain')
    
    # 3. remove y axis stack settings
    if 'stack' in y_enc:
        del new_spec['encoding']['y']['stack']
    
    # 4. keep color encoding to maintain visual consistency
    # color field remains the same, but now each bar is displayed independently
    
    return {
        'success': True,
        'operation': 'expand_stack',
        'vega_spec': new_spec,
        'message': f'Expanded stacked bars for category "{category}" into parallel bars'
    }


def toggle_stack_mode(vega_spec: Dict, mode: str = "grouped") -> Dict[str, Any]:
    """
    Toggle stacked/grouped display mode globally.
    
    - grouped: convert stacked bar chart to grouped bar chart (all subcategories displayed side by side)
    - stacked: restore to stacked display
    
    Parameters:
        vega_spec: Vega spec
        mode: "grouped" or "stacked"
    
    Returns:
        modified spec
    """
    new_spec = copy.deepcopy(vega_spec)
    encoding = new_spec.get('encoding', {})
    
    color_enc = encoding.get('color', {})
    color_field = color_enc.get('field')
    
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field, cannot toggle stack mode'}
    
    if mode == "grouped":
        # add xOffset encoding to implement grouped bar chart
        new_spec['encoding']['xOffset'] = {
            'field': color_field
        }
        # remove stack settings
        if 'stack' in encoding.get('y', {}):
            del new_spec['encoding']['y']['stack']
        
        message = 'Switched to grouped mode: all subcategories displayed side by side,便于跨类别对比'
        
    elif mode == "stacked":
        # remove xOffset, restore stacked
        if 'xOffset' in new_spec['encoding']:
            del new_spec['encoding']['xOffset']
        # restore stack settings
        if 'y' in new_spec['encoding']:
            new_spec['encoding']['y']['stack'] = 'zero'
        
        message = 'Switched to stacked mode: all subcategories displayed stacked,便于查看总量'
        
    else:
        return {'success': False, 'error': f'Invalid mode: {mode}, please use "grouped" or "stacked"'}
    
    # store mode state
    new_spec['_stack_mode'] = mode
    
    return {
        'success': True,
        'operation': 'toggle_stack_mode',
        'vega_spec': new_spec,
        'message': message
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


# ============================================================================
# Bar visibility tools (interaction necessity)
# - Add/remove whole bars by x category (stacked or grouped)
# - Add/remove individual bar items by (x, sub) pair
# - Optionally load missing data from vega_spec._metadata.full_data_path
# ============================================================================

_VIS_FILTER_TAG = "bar_visibility"


def _ensure_transform_list(spec: Dict[str, Any]) -> None:
    if "transform" not in spec or spec["transform"] is None:
        spec["transform"] = []


def _find_or_create_visibility_filter_transform(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the single transform node inserted by this module; create if absent.
    We tag it with an extra key so Vega-Lite ignores it but we can find it.
    """
    _ensure_transform_list(spec)
    for t in spec["transform"]:
        if isinstance(t, dict) and t.get("__avs_tool") == _VIS_FILTER_TAG:
            return t
    node = {"__avs_tool": _VIS_FILTER_TAG, "filter": "true"}
    spec["transform"].append(node)
    return node


def _detect_x_category_field(spec: Dict[str, Any], x_field: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect which encoding channel is the category axis for bars.
    Returns (field_name, channel_name) where channel_name is "x" or "y".
    """
    enc = spec.get("encoding", {}) or {}
    if x_field:
        return x_field, "x"

    x_enc = enc.get("x", {}) or {}
    y_enc = enc.get("y", {}) or {}
    x_f = x_enc.get("field")
    y_f = y_enc.get("field")
    x_t = str(x_enc.get("type", "")).lower()
    y_t = str(y_enc.get("type", "")).lower()

    nominal_types = {"nominal", "ordinal"}
    quantitative_types = {"quantitative"}

    # Typical vertical bar: x nominal, y quantitative
    if x_f and (x_t in nominal_types or (y_t in quantitative_types and x_t != "quantitative")):
        return x_f, "x"
    # Horizontal bar: y nominal, x quantitative
    if y_f and (y_t in nominal_types or (x_t in quantitative_types and y_t != "quantitative")):
        return y_f, "y"

    # Fallback: prefer x.field
    return x_f or y_f, ("x" if x_f else ("y" if y_f else None))


def _detect_sub_field(spec: Dict[str, Any], sub_field: Optional[str] = None) -> Optional[str]:
    enc = spec.get("encoding", {}) or {}
    if sub_field:
        return sub_field
    xoff = (enc.get("xOffset", {}) or {}).get("field")
    if xoff:
        return xoff
    color = (enc.get("color", {}) or {}).get("field")
    return color


def _get_values_from_data_obj(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("values"), list):
        return obj["values"]
    return []


def _load_full_values_if_available(spec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    meta = spec.get("_metadata") or {}
    full_path = meta.get("full_data_path")
    if not full_path:
        return None

    p = Path(full_path)
    if not p.is_absolute():
        # Resolve relative to repo root (tools/..)
        repo_root = Path(__file__).resolve().parent.parent
        p = (repo_root / full_path).resolve()

    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Accept multiple shapes:
    # 1) {"values":[...]}
    # 2) {"data":{"values":[...]}}
    # 3) full vega spec containing {"data":{"values":[...]}}
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        return data["values"]
    if isinstance(data, dict):
        dv = (data.get("data") or {}).get("values")
        if isinstance(dv, list):
            return dv
    return None


def _merge_rows(target: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> int:
    """Merge rows into target with naive dedup; returns number of added rows."""
    def row_key(r: Dict[str, Any]) -> Tuple:
        # stable key; ok for small demo data
        return tuple(sorted((str(k), json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in r.items()))

    existing = {row_key(r) for r in target if isinstance(r, dict)}
    added = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = row_key(r)
        if k in existing:
            continue
        target.append(r)
        existing.add(k)
        added += 1
    return added


def _maybe_expand_data_from_full(spec: Dict[str, Any], predicate) -> Dict[str, Any]:
    """
    If predicate(row) matches rows missing in current data, try to load from full_data_path and merge.
    Returns dict with keys: loaded(bool), added(int)
    """
    data_obj = spec.get("data", {}) or {}
    current_values = _get_values_from_data_obj(data_obj)
    full_values = _load_full_values_if_available(spec)
    if not full_values:
        return {"loaded": False, "added": 0}

    needed_rows = [r for r in full_values if isinstance(r, dict) and predicate(r)]
    added = _merge_rows(current_values, needed_rows)
    # Ensure spec.data.values points to our list if it was missing
    if "data" not in spec or spec["data"] is None:
        spec["data"] = {"values": current_values}
    else:
        spec["data"]["values"] = current_values
    return {"loaded": True, "added": added}


def add_bars(vega_spec: Dict, values: List[Any], x_field: Optional[str] = None) -> Dict[str, Any]:
    """
    Add whole bars (x categories). Works for stacked or grouped bars.
    If the requested category isn't present in current data, tries to load from _metadata.full_data_path.
    """
    new_spec = copy.deepcopy(vega_spec)
    field, channel = _detect_x_category_field(new_spec, x_field=x_field)
    if not field or not channel:
        return {"success": False, "error": "Cannot find category axis field (x/y)."}

    data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    requested = list(dict.fromkeys(values or []))

    # Base visible set should reflect current view BEFORE any full-data merge.
    existing_categories_before = {r.get(field) for r in data_values if isinstance(r, dict)}
    state = new_spec.get("_bar_visibility_state") or {}
    visible = (
        set(state.get("visible_x", []))
        if state.get("mode") == "x" and state.get("x_field") == field
        else set(existing_categories_before)
    )

    # If missing, attempt full data merge for those categories
    missing = [v for v in requested if v not in existing_categories_before]
    load_info = {"loaded": False, "added": 0}
    if missing:
        load_info = _maybe_expand_data_from_full(new_spec, lambda r: r.get(field) in set(missing))
        data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    existing_categories = {r.get(field) for r in data_values if isinstance(r, dict)}

    actually_added = []
    still_missing = []
    for v in requested:
        if v in existing_categories:
            if v not in visible:
                visible.add(v)
                actually_added.append(v)
        else:
            still_missing.append(v)

    # Update filter
    filt = _find_or_create_visibility_filter_transform(new_spec)
    category_str = ",".join([json.dumps(v, ensure_ascii=False) for v in sorted(visible, key=lambda x: str(x))])
    filt["filter"] = f"indexof([{category_str}], {_datum_ref(field)}) >= 0"

    new_spec["_bar_visibility_state"] = {
        "mode": "x",
        "x_field": field,
        "visible_x": list(visible),
        "updated_at": datetime.now().isoformat(),
    }

    msg = f"Added {len(actually_added)} bars on {field}"
    if load_info.get("loaded"):
        msg += f"; loaded {load_info.get('added', 0)} rows from full_data_path"
    if still_missing:
        msg += f"; missing categories not found: {still_missing}"

    return {"success": True, "operation": "add_bars", "vega_spec": new_spec, "message": msg}


def remove_bars(vega_spec: Dict, values: List[Any], x_field: Optional[str] = None) -> Dict[str, Any]:
    """Remove whole bars (x categories) by hiding them via a single managed transform filter."""
    new_spec = copy.deepcopy(vega_spec)
    field, channel = _detect_x_category_field(new_spec, x_field=x_field)
    if not field or not channel:
        return {"success": False, "error": "Cannot find category axis field (x/y)."}

    data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    existing_categories = {r.get(field) for r in data_values if isinstance(r, dict)}

    state = new_spec.get("_bar_visibility_state") or {}
    visible = set(state.get("visible_x", [])) if state.get("mode") == "x" and state.get("x_field") == field else set(existing_categories)

    requested = set(values or [])
    actually_removed = []
    for v in list(requested):
        if v in visible:
            visible.remove(v)
            actually_removed.append(v)

    filt = _find_or_create_visibility_filter_transform(new_spec)
    category_str = ",".join([json.dumps(v, ensure_ascii=False) for v in sorted(visible, key=lambda x: str(x))])
    filt["filter"] = f"indexof([{category_str}], {_datum_ref(field)}) >= 0"

    new_spec["_bar_visibility_state"] = {
        "mode": "x",
        "x_field": field,
        "visible_x": list(visible),
        "updated_at": datetime.now().isoformat(),
    }

    return {
        "success": True,
        "operation": "remove_bars",
        "vega_spec": new_spec,
        "message": f"Removed {len(actually_removed)} bars on {field}"
    }


def add_bar_items(
    vega_spec: Dict,
    items: List[Dict[str, Any]],
    x_field: Optional[str] = None,
    sub_field: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add individual bar items by (x, sub) pair. Works for stacked (color) and grouped (xOffset) bars.
    items: [{"x": <x_value>, "sub": <sub_value>}, ...]
    """
    new_spec = copy.deepcopy(vega_spec)
    x_f, _ = _detect_x_category_field(new_spec, x_field=x_field)
    sub_f = _detect_sub_field(new_spec, sub_field=sub_field)
    if not x_f:
        return {"success": False, "error": "Cannot find x category field."}
    if not sub_f:
        return {"success": False, "error": "Cannot find sub field (encoding.xOffset.field or encoding.color.field)."}

    data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    existing_pairs_before = {(r.get(x_f), r.get(sub_f)) for r in data_values if isinstance(r, dict)}

    requested = []
    for it in items or []:
        if not isinstance(it, dict) or "x" not in it or "sub" not in it:
            continue
        requested.append((it["x"], it["sub"]))
    requested = list(dict.fromkeys(requested))

    # Base visible pairs should reflect current view BEFORE any full-data merge.
    state = new_spec.get("_bar_visibility_state") or {}
    visible_pairs = (
        set(tuple(p) for p in state.get("visible_items", []))
        if state.get("mode") == "item" and state.get("x_field") == x_f and state.get("sub_field") == sub_f
        else set(existing_pairs_before)
    )

    missing_pairs = [p for p in requested if p not in existing_pairs_before]
    load_info = {"loaded": False, "added": 0}
    if missing_pairs:
        missing_set = set(missing_pairs)
        load_info = _maybe_expand_data_from_full(
            new_spec,
            lambda r: (r.get(x_f), r.get(sub_f)) in missing_set
        )
        data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    existing_pairs = {(r.get(x_f), r.get(sub_f)) for r in data_values if isinstance(r, dict)}

    actually_added = []
    still_missing = []
    for p in requested:
        if p in existing_pairs:
            if p not in visible_pairs:
                visible_pairs.add(p)
                actually_added.append(p)
        else:
            still_missing.append(p)

    xr, sr = _datum_ref(x_f), _datum_ref(sub_f)
    filt = _find_or_create_visibility_filter_transform(new_spec)
    parts = []
    for xv, sv in sorted(visible_pairs, key=lambda t: (str(t[0]), str(t[1]))):
        parts.append(f"({xr} == {json.dumps(xv, ensure_ascii=False)} && {sr} == {json.dumps(sv, ensure_ascii=False)})")
    filt["filter"] = " || ".join(parts) if parts else "false"

    new_spec["_bar_visibility_state"] = {
        "mode": "item",
        "x_field": x_f,
        "sub_field": sub_f,
        "visible_items": [list(p) for p in visible_pairs],
        "updated_at": datetime.now().isoformat(),
    }

    msg = f"Added {len(actually_added)} bar items by ({x_f}, {sub_f})"
    if load_info.get("loaded"):
        msg += f"; loaded {load_info.get('added', 0)} rows from full_data_path"
    if still_missing:
        msg += f"; missing items not found: {still_missing}"

    return {"success": True, "operation": "add_bar_items", "vega_spec": new_spec, "message": msg}


def remove_bar_items(
    vega_spec: Dict,
    items: List[Dict[str, Any]],
    x_field: Optional[str] = None,
    sub_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove individual bar items by (x, sub) pair by updating the managed filter."""
    new_spec = copy.deepcopy(vega_spec)
    x_f, _ = _detect_x_category_field(new_spec, x_field=x_field)
    sub_f = _detect_sub_field(new_spec, sub_field=sub_field)
    if not x_f:
        return {"success": False, "error": "Cannot find x category field."}
    if not sub_f:
        return {"success": False, "error": "Cannot find sub field (encoding.xOffset.field or encoding.color.field)."}

    data_values = _get_values_from_data_obj(new_spec.get("data", {}) or {})
    existing_pairs = {(r.get(x_f), r.get(sub_f)) for r in data_values if isinstance(r, dict)}

    state = new_spec.get("_bar_visibility_state") or {}
    visible_pairs = (
        set(tuple(p) for p in state.get("visible_items", []))
        if state.get("mode") == "item" and state.get("x_field") == x_f and state.get("sub_field") == sub_f
        else set(existing_pairs)
    )

    requested = []
    for it in items or []:
        if not isinstance(it, dict) or "x" not in it or "sub" not in it:
            continue
        requested.append((it["x"], it["sub"]))
    requested = set(requested)

    actually_removed = []
    for p in list(requested):
        if p in visible_pairs:
            visible_pairs.remove(p)
            actually_removed.append(p)

    xr, sr = _datum_ref(x_f), _datum_ref(sub_f)
    filt = _find_or_create_visibility_filter_transform(new_spec)
    parts = []
    for xv, sv in sorted(visible_pairs, key=lambda t: (str(t[0]), str(t[1]))):
        parts.append(f"({xr} == {json.dumps(xv, ensure_ascii=False)} && {sr} == {json.dumps(sv, ensure_ascii=False)})")
    filt["filter"] = " || ".join(parts) if parts else "false"

    new_spec["_bar_visibility_state"] = {
        "mode": "item",
        "x_field": x_f,
        "sub_field": sub_f,
        "visible_items": [list(p) for p in visible_pairs],
        "updated_at": datetime.now().isoformat(),
    }

    return {
        "success": True,
        "operation": "remove_bar_items",
        "vega_spec": new_spec,
        "message": f"Removed {len(actually_removed)} bar items by ({x_f}, {sub_f})"
    }


def filter_subcategories(vega_spec: Dict, subcategories_to_remove: List[Any], sub_field: Optional[str] = None) -> Dict[str, Any]:
    """
    Filter subcategories (color or xOffset encoded categories) from bar charts.
    
    Works for both stacked charts (color encoding) and grouped charts (xOffset encoding).
    Automatically detects the subcategory field from xOffset.field or color.field.
    
    Parameters:
        vega_spec: Vega-Lite specification
        subcategories_to_remove: List of subcategory values to remove (e.g., [1, 2] or ["Type1", "Type2"])
        sub_field: Subcategory field name (optional, auto-detected from xOffset.field or color.field)
    
    Returns:
        Modified vega_spec with filtered subcategories
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # Auto-detect subcategory field
    sub_f = _detect_sub_field(new_spec, sub_field=sub_field)
    
    if not sub_f:
        return {'success': False, 'error': 'Cannot find subcategory field (xOffset.field or color.field)'}
    
    if not subcategories_to_remove:
        return {'success': False, 'error': 'subcategories_to_remove cannot be empty'}
    
    # Add filter transform to exclude specified subcategories
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    # Build filter expression: exclude subcategories in the removal list
    sub_ref = _datum_ref(sub_f)
    exclusion_parts = []
    for sub_val in subcategories_to_remove:
        exclusion_parts.append(f"{sub_ref} != {json.dumps(sub_val, ensure_ascii=False)}")
    
    filter_expr = " && ".join(exclusion_parts) if exclusion_parts else "true"
    new_spec['transform'].append({
        'filter': filter_expr
    })
    
    # Update color.scale.domain if color encoding exists to maintain visual consistency
    encoding = new_spec.get('encoding', {})
    color_enc = encoding.get('color', {})
    if color_enc:
        scale_config = color_enc.get('scale', {})
        if scale_config and 'domain' in scale_config:
            original_domain = scale_config['domain']
            if isinstance(original_domain, list):
                filtered_domain = [d for d in original_domain if d not in subcategories_to_remove]
                if 'encoding' not in new_spec:
                    new_spec['encoding'] = {}
                if 'color' not in new_spec['encoding']:
                    new_spec['encoding']['color'] = {}
                if 'scale' not in new_spec['encoding']['color']:
                    new_spec['encoding']['color']['scale'] = {}
                new_spec['encoding']['color']['scale']['domain'] = filtered_domain
    
    return {
        'success': True,
        'operation': 'filter_subcategories',
        'vega_spec': new_spec,
        'message': f'Filtered out {len(subcategories_to_remove)} subcategories: {subcategories_to_remove}'
    }


__all__ = [
    'sort_bars',
    'filter_categories',
    'highlight_top_n',
    'expand_stack',
    'toggle_stack_mode',
    'add_bars',
    'remove_bars',
    'add_bar_items',
    'remove_bar_items',
    'filter_subcategories',
]
