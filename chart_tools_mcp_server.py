"""
chart tools MCP Server
wraps all visualization tools, called by Claude、GPT、Qwen etc.

usage:
    1. install dependencies: pip install mcp numpy scipy scikit-learn
    2. run server: python chart_tools_mcp_server.py
    3. or use npm tool: npx @modelcontextprotocol/inspector python chart_tools_mcp_server.py
"""

import json
import copy
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Union
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from mcp.server.fastmcp import FastMCP

# Constants
_VIS_FILTER_TAG = "bar_visibility"


def _datum_ref(field: str) -> str:
    """Vega expr: datum access that supports field names with spaces/special chars."""
    if not field:
        return "datum"
    s = str(field).replace("\\", "\\\\").replace("'", "\\'")
    return f"datum['{s}']"


# ============================================================
# initialize MCP Server
# ============================================================
mcp = FastMCP("chart_tools_mcp_server")


# ============================================================
# common tools (Common Tools)
# ============================================================

#helper function
def _get_encoding_field(vega_spec: Dict, channel: str) -> Optional[str]:
    """获取编码字段"""
    return vega_spec.get('encoding', {}).get(channel, {}).get('field')

#scatter plot
def _infer_field_type(data: List[Dict], field: str) -> str:
    """infer field type of Vega-Lite"""
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
                # check if is date format
                if any(sep in value for sep in ['-', '/', ':']):
                    return 'temporal'
                return 'nominal'
    return 'nominal'


@mcp.tool()
def change_encoding(vega_spec: Dict, channel: str, field: str) -> Dict[str, Any]:
    """
    Modify the field mapping of the specified encoding channel.
    Supports bar_chart, line_chart, heatmap, scatter_plot (Vega-Lite charts).
    
    Args:
        vega_spec: Vega-Lite spec
        channel: encoding channel ("x", "y", "color", "size", "shape", "opacity")
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

#bar chart
def _detect_sub_field(spec: Dict[str, Any], sub_field: Optional[str] = None) -> Optional[str]:
    enc = spec.get("encoding", {}) or {}
    if sub_field:
        return sub_field
    xoff = (enc.get("xOffset", {}) or {}).get("field")
    if xoff:
        return xoff
    color = (enc.get("color", {}) or {}).get("field")
    return color

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


def _get_values_from_data_obj(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("values"), list):
        return obj["values"]
    return []

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

def _ensure_transform_list(spec: Dict[str, Any]) -> None:
    if "transform" not in spec or spec["transform"] is None:
        spec["transform"] = []

# ==================== perception APIs ====================
@mcp.tool()
def get_data_summary(vega_spec: Dict, scope: str = 'all') -> Dict[str, Any]:
    """
    return data summary
    
    Args:
        vega_spec: Vega-Lite spec
        scope: 'visible' or 'all' - return visible data or all data summary
        
    Returns:
        data summary dictionary
    """
    # extract data from vega_spec
    data = vega_spec.get('data', {}).get('values', [])
    
    if not data:
        return {'success': False, 'error': 'No data available'}
    
    # calculate summary
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
        'summary': summary
    }

@mcp.tool()
def get_tooltip_data(vega_spec: Dict, position: Tuple[float, float]) -> Dict[str, Any]:
    """get tooltip data at specified position"""
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


# ==================== bar chart tools ====================

def _get_time_field(vega_spec: Dict) -> Optional[str]:
    """get time field (support layer structure)"""
    # 检查是否有 layer 结构
    if 'layer' in vega_spec and len(vega_spec['layer']) > 0:
        encoding = vega_spec['layer'][0].get('encoding', {})
    else:
        encoding = vega_spec.get('encoding', {})
    
    # time field is usually on x axis, but may be on y axis
    x_encoding = encoding.get('x', {})
    y_encoding = encoding.get('y', {})
    
    if x_encoding.get('type') == 'temporal':
        return x_encoding.get('field')
    elif y_encoding.get('type') == 'temporal':
        return y_encoding.get('field')
    
    # if type is not specified, try to infer from field (usually time is on x axis)
    return x_encoding.get('field') or y_encoding.get('field')

@mcp.tool()
def sort_bars(vega_spec: Dict, order: str = "descending", by_subcategory: Optional[str] = None) -> Dict[str, Any]:
    """Sort bars. Stacked: use by_subcategory to sort x by that subcategory's value, or omit to sort stack layers by value. Grouped/simple: sort x by total."""
    from tools import bar_chart_tools
    return bar_chart_tools.sort_bars(vega_spec, order=order, by_subcategory=by_subcategory)

@mcp.tool()
def filter_categories(vega_spec: Dict, categories: List[str]) -> Dict[str, Any]:
    """filter specific categories"""
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    
    if not x_field:
        return {'success': False, 'error': 'Cannot find category field'}
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    category_str = ','.join([f'"{c}"' for c in categories])
    new_spec['transform'].append({
        'filter': f'indexof([{category_str}], datum.{x_field}) >= 0'
    })
    
    return {
        'success': True,
        'operation': 'filter_categories',
        'vega_spec': new_spec,
        'message': f'Filtered to {len(categories)} categories'
    }

@mcp.tool()
def highlight_top_n(vega_spec: Dict, n: int = 5, order: str = "descending") -> Dict[str, Any]:
    """highlight top N bars by aggregated value (supports stacked/grouped charts)"""
    new_spec = copy.deepcopy(vega_spec)
    
    data = new_spec.get('data', {}).get('values', [])
    encoding = new_spec.get('encoding', {})
    y_enc = encoding.get('y', {})
    x_enc = encoding.get('x', {})
    y_field = y_enc.get('field')
    x_field = x_enc.get('field')
    agg = y_enc.get('aggregate', None)
    
    if not y_field or not x_field or not data:
        return {'success': False, 'error': 'Missing data or x/y field'}
    
    # Aggregate by x_field (category) to handle stacked/grouped charts
    from collections import defaultdict
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

@mcp.tool() 
def expand_stack(vega_spec: Dict, category: str) -> Dict[str, Any]:
    """
    expand stacked bars in a category to a parallel bars chart.
    
    This is an "interaction necessity" tool - the baseline of the stacked chart is different, making it difficult to compare the values of each sub-category directly.
    
    Args:
        vega_spec: Vega-Lite spec
        category: the category name to expand (e.g. "East China")
    
    Returns:
        expanded parallel bars chart spec
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
    
    # 1. add filter, only keep data of the specified category
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    new_spec['transform'].append({
        'filter': f'datum.{x_field} == "{category}"'
    })
    
    # 2. move original color field to x axis
    original_x_enc = encoding.get('x', {})
    new_spec['encoding']['x'] = {
        'field': color_field,
        'type': 'nominal',
        'title': color_enc.get('title', color_field),
        'axis': {'labelAngle': -45}
    }
    # keep original x axis scale setting if there is any
    if 'scale' in color_enc:
        new_spec['encoding']['x']['sort'] = color_enc['scale'].get('domain')
    
    # 3. remove y axis stack setting
    if 'stack' in y_enc:
        del new_spec['encoding']['y']['stack']
    
    # 4. keep color encoding to maintain visual consistency
    # color field remains the same, but now each bar is displayed independently
    
    # 5. do not modify title (keep original title or no title)
    # remove title to avoid hard coding
    
    # 6. store expanded state for restoration
    new_spec['_expand_state'] = {
        'expanded': True,
        'category': category,
        'original_x_field': x_field
    }
    
    return {
        'success': True,
        'operation': 'expand_stack',
        'vega_spec': new_spec,
        'message': f'Expanded stacked bars of "{category}", now可以直接比较各产品类别的销售额'
    }

@mcp.tool()
def toggle_stack_mode(vega_spec: Dict, mode: str = "grouped") -> Dict[str, Any]:
    """
    globally toggle stacked/grouped display mode.
    
    - grouped: convert stacked bars to grouped bars (all sub-categories displayed side by side)
    - stacked: restore to stacked display
    
    Args:
        vega_spec: Vega-Lite spec
        mode: "grouped" or "stacked"
    
    Returns:
        spec after toggling mode
    """
    new_spec = copy.deepcopy(vega_spec)
    encoding = new_spec.get('encoding', {})
    
    color_enc = encoding.get('color', {})
    color_field = color_enc.get('field')
    
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field, cannot toggle stacked mode'}
    
    if mode == "grouped":
        # add xOffset encoding, implement grouped bars
        new_spec['encoding']['xOffset'] = {
            'field': color_field
        }
        # remove stack setting
        if 'stack' in encoding.get('y', {}):
            del new_spec['encoding']['y']['stack']
        
        message = 'Switched to grouped mode: all sub-categories displayed side by side,便于跨类别对比'
        
    elif mode == "stacked":
        # remove xOffset, restore stacked
        if 'xOffset' in new_spec['encoding']:
            del new_spec['encoding']['xOffset']
        # restore stack setting
        if 'y' in new_spec['encoding']:
            new_spec['encoding']['y']['stack'] = 'zero'
        
        message = 'Switched to stacked mode: all sub-categories stacked,便于查看总量'
        
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
@mcp.tool()
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
    filt["filter"] = f"indexof([{category_str}], datum.{field}) >= 0"

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

@mcp.tool()
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

@mcp.tool()
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

    filt = _find_or_create_visibility_filter_transform(new_spec)
    parts = []
    for xv, sv in sorted(visible_pairs, key=lambda t: (str(t[0]), str(t[1]))):
        parts.append(f"(datum.{x_f} == {json.dumps(xv, ensure_ascii=False)} && datum.{sub_f} == {json.dumps(sv, ensure_ascii=False)})")
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

@mcp.tool()
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

    filt = _find_or_create_visibility_filter_transform(new_spec)
    parts = []
    for xv, sv in sorted(visible_pairs, key=lambda t: (str(t[0]), str(t[1]))):
        parts.append(f"(datum.{x_f} == {json.dumps(xv, ensure_ascii=False)} && datum.{sub_f} == {json.dumps(sv, ensure_ascii=False)})")
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



# ==================== 折线图专用工具 (Line Chart Tools) ====================
@mcp.tool()
def zoom_time_range(vega_spec: Dict, start: str, end: str) -> Dict[str, Any]:
    """zoom time range -放大视图到特定时间段（不删除数据）"""
    new_spec = copy.deepcopy(vega_spec)
    
    # get time field name (support layer structure)
    time_field = _get_time_field(new_spec)
    
    if not time_field:
        return {
            'success': False,
            'error': 'Cannot find time field'
        }
    
    # determine which axis the time field is on
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    x_field = encoding.get('x', {}).get('field')
    time_axis = 'x' if x_field == time_field else 'y'
    
    # directly use VLM provided date format, no conversion
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
    
    # ensure mark has clip: true, clip points outside the range
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


@mcp.tool()
def highlight_trend(vega_spec: Dict, trend_type: str = "increasing") -> Dict[str, Any]:
    """highlight trend - add regression trend line"""
    new_spec = copy.deepcopy(vega_spec)
    
    # get x and y fields (support layer structure)
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
    
    # if original spec does not have layer, convert to layer structure
    if 'layer' not in new_spec:
        original_layer = copy.deepcopy(new_spec)
        # remove top level mark and encoding, because they are now in layer
        for key in ['mark', 'encoding']:
            if key in original_layer:
                del original_layer[key]
        
        new_spec = original_layer
        new_spec['layer'] = [{
            'mark': vega_spec.get('mark', 'line'),
            'encoding': vega_spec.get('encoding', {})
        }]
    
    # add trend line layer
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



@mcp.tool()
def detect_anomalies(vega_spec: Dict, threshold: float = 2.0) -> Dict[str, Any]:
    """detect anomalies - detect and highlight anomalies in the view"""
    import numpy as np
    
    data = vega_spec.get('data', {}).get('values', [])
    
    # get fields (support layer structure)
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
    
    # calculate statistical features
    mean = np.mean(values)
    std = np.std(values)
    
    # identify anomalies
    anomaly_data = []
    for row in data:
        val = row.get(y_field)
        if val is not None and abs(val - mean) > threshold * std:
            anomaly_data.append(row)
    
    new_spec = copy.deepcopy(vega_spec)
    
    # if anomalies are detected, mark them in the view
    if anomaly_data:
        # convert to layer structure
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
        
        # add anomaly point marking layer
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

@mcp.tool()
def bold_lines(vega_spec: Dict, line_names: List[str], line_field: str = None) -> Dict[str, Any]:
    """
    bold specified lines
    
    Args:
        vega_spec: Vega-Lite spec
        line_names: list of line names to bold
        line_field: line grouping field name (optional, auto-detect from color/detail)
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # automatically detect grouping field (priority color, then detail)
    if line_field is None:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            encoding = new_spec['layer'][0].get('encoding', {})
        else:
            encoding = new_spec.get('encoding', {})
        
        color_enc = encoding.get('color', {})
        line_field = color_enc.get('field')
        
        if not line_field:
            # try to get from detail field
            detail_enc = encoding.get('detail', {})
            line_field = detail_enc.get('field')
    
    if not line_field:
        return {
            'success': False,
            'error': 'Cannot find line grouping field. Please specify line_field parameter.'
        }
    
    # build strokeWidth condition encoding (bracket notation for field names with spaces)
    lines_json = json.dumps(line_names)
    stroke_width_encoding = {
        'condition': {
            'test': f'indexof({lines_json}, {_datum_ref(line_field)}) >= 0',
            'value': 4
        },
        'value': 1
    }
    
    # apply to spec
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

@mcp.tool() 
def filter_lines(vega_spec: Dict, lines_to_remove: List[str], line_field: str = None) -> Dict[str, Any]:
    """
    filter out specified lines
    
    Args:
        vega_spec: Vega-Lite spec
        lines_to_remove: list of line names to remove
        line_field: line grouping field name (optional, auto-detect from color/detail)
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # automatically detect grouping field (priority color, then detail)
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
    
    # add filter transform to exclude specified series (bracket notation for field names with spaces)
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

@mcp.tool()
def show_moving_average(vega_spec: Dict, window_size: int = 3) -> Dict[str, Any]:
    """
    add moving average line
    
    Args:
        vega_spec: Vega-Lite spec
        window_size: moving average window size
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # get fields
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
    
    # if original spec does not have layer, convert to layer structure
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
    
    # detect grouping field (multi-series)
    color_field = encoding.get('color', {}).get('field')
    detail_field = encoding.get('detail', {}).get('field')
    group_field = color_field or detail_field
    
    # moving average field name
    ma_field = f'{y_field}_ma'
    
    # build window transform
    window_transform = {
        'window': [{
            'op': 'mean',
            'field': y_field,
            'as': ma_field
        }],
        'frame': [-(window_size - 1), 0],
        'sort': [{'field': x_field, 'order': 'ascending'}]
    }
    if group_field:
        window_transform['groupby'] = [group_field]
    
    # add moving average line layer
    ma_encoding = {
        'x': {'field': x_field, 'type': encoding['x'].get('type', 'temporal')},
        'y': {'field': ma_field, 'type': 'quantitative'}
    }
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

@mcp.tool()
def focus_lines(
    vega_spec: Dict,
    lines: List[str],
    line_field: Optional[str] = None,
    mode: str = "dim",
    dim_opacity: float = 0.08,
) -> Dict[str, Any]:
    """
    cognitive interaction necessity: focus on few lines, others dim or hide.
    
    Args:
        vega_spec: Vega-Lite spec
        lines: list of line names to focus
        line_field: line grouping field name (optional, auto-detect from color/detail)
        mode: 'dim' (others dim)
        dim_opacity: opacity of non-focused series when mode='dim'
    """
    import json

    new_spec = copy.deepcopy(vega_spec)

    if not isinstance(lines, list) or not lines:
        return {'success': False, 'error': 'lines must be a non-empty list'}

    # automatically detect grouping field (priority color, then detail)
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

    if mode == "hide":
        if 'transform' not in new_spec:
            new_spec['transform'] = []
        new_spec['transform'].append({
            'filter': f'indexof({lines_json}, {_datum_ref(line_field)}) >= 0',
            '_avs_tag': 'focus_lines'
        })
    else:
        # default dim (bracket notation for field names with spaces)
        opacity_encoding = {
            'condition': {
                'test': f'indexof({lines_json}, {_datum_ref(line_field)}) >= 0',
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

@mcp.tool()
def drilldown_line_time(
    vega_spec: Dict,
    level: str,
    value: int,
    parent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Time line chart drilldown: year → month → day
    
    cognitive interaction necessity: many years of daily data is huge, initially only the annual aggregate trend can be displayed.
    Through drilldown, we can go deeper and find more granular patterns.
    
    Initial view (year) only shows one aggregate point per year, after drilldown:
    - Drilldown to year: display monthly aggregate data for that year
    - Drilldown to month: display daily data for that month
    
    Args:
        vega_spec: Vega-Lite spec
        level: 'year' | 'month' | 'date'
        value: value corresponding to level
               - year: 4-digit year like 2023
               - month: 1-12 (1=January, 12=December)
               - date: 1-31
        parent: parent information, like {'year': 2023} or {'year': 2023, 'month': 3}
    
    Returns:
        drilldown view 
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # initialize or get drilldown state
    state = new_spec.get('_line_drilldown_state')
    if not isinstance(state, dict):
        state = {}
    
    # on first drilldown, save original transform and encoding, and detect field names
    if 'original_transform' not in state:
        state['original_transform'] = copy.deepcopy(new_spec.get('transform', []))
        state['original_encoding'] = copy.deepcopy(new_spec.get('encoding', {}))
        state['original_title'] = new_spec.get('title', '')
        
        # dynamically detect original time field
        # priority from timeUnit in transform
        raw_time_field = None
        for t in state['original_transform']:
            if isinstance(t, dict) and 'timeUnit' in t and 'field' in t:
                raw_time_field = t['field']
                break
        
        # if not found, infer from x encoding
        if not raw_time_field:
            x_enc = state['original_encoding'].get('x', {})
            raw_time_field = x_enc.get('field', 'date')
            # if x field is aggregated (e.g. year_date), try to infer original field
            if raw_time_field and '_' in raw_time_field:
                # year_date, month_date -> possible original field is date
                possible_raw = raw_time_field.split('_')[-1]
                # check if the field exists in the data
                data = new_spec.get('data', {}).get('values', [])
                if data and possible_raw in data[0]:
                    raw_time_field = possible_raw
        
        state['raw_time_field'] = raw_time_field or 'date'
        
        # dynamically detect value field
        # priority from aggregate in transform
        raw_value_field = None
        for t in state['original_transform']:
            if isinstance(t, dict) and 'aggregate' in t:
                aggs = t['aggregate']
                if isinstance(aggs, list) and len(aggs) > 0:
                    raw_value_field = aggs[0].get('field')
                    break
        
        # if not found, infer from y encoding
        if not raw_value_field:
            y_enc = state['original_encoding'].get('y', {})
            y_field = y_enc.get('field', '')
            # if aggregated field (e.g. total_sales), try to infer original field from data
            if y_field.startswith('total_') or y_field.startswith('sum_'):
                possible_raw = y_field.replace('total_', '').replace('sum_', '')
                data = new_spec.get('data', {}).get('values', [])
                if data and possible_raw in data[0]:
                    raw_value_field = possible_raw
            else:
                raw_value_field = y_field
        
        state['raw_value_field'] = raw_value_field or 'value'
        
        # detect grouping field (color encoding)
        color_enc = state['original_encoding'].get('color', {})
        state['group_field'] = color_enc.get('field')
    
    # use stored field names
    raw_date_field = state.get('raw_time_field', 'date')
    raw_value_field = state.get('raw_value_field', 'value')
    group_field = state.get('group_field')
    
    # merge parent information (explicit parent > stored state)
    p = {}
    if isinstance(state.get('parent'), dict):
        p.update(state.get('parent'))
    if isinstance(parent, dict):
        p.update(parent)
    
    # normalize level
    level = str(level).lower().strip()
    
    try:
        value = int(value)
    except (TypeError, ValueError):
        return {'success': False, 'error': f'Invalid value: {value}, should be an integer'}
    
    # rebuild transform: first filter, then aggregate
    new_transforms = []
    title_suffix = ""
    
    # build groupby list
    groupby_fields = ['_time_field_']  # 占位，后面替换
    if group_field:
        groupby_fields.append(group_field)
    
    if level == 'year':
        # drilldown to specified year, display monthly aggregate data
        if value < 1900 or value > 2100:
            return {'success': False, 'error': f'Invalid year: {value}'}
        
        # 1. filter to specified year
        new_transforms.append({
            'filter': f'year(datum.{raw_date_field}) == {value}',
            '_avs_tag': 'line_drilldown_time'
        })
        # 2. aggregate by month
        new_transforms.append({
            'timeUnit': 'yearmonth',
            'field': raw_date_field,
            'as': 'month_date',
            '_avs_tag': 'line_drilldown_time'
        })
        
        # build groupby
        month_groupby = ['month_date']
        if group_field:
            month_groupby.append(group_field)
        
        new_transforms.append({
            'aggregate': [{'op': 'sum', 'field': raw_value_field, 'as': 'total_value'}],
            'groupby': month_groupby,
            '_avs_tag': 'line_drilldown_time'
        })
        
        state['parent'] = {'year': value}
        title_suffix = f'{value} year monthly trend'
        
        # update encoding
        new_spec['encoding'] = {
            'x': {
                'field': 'month_date',
                'type': 'temporal',
                'title': 'month',
                'axis': {'format': '%Y-%m'}
            },
            'y': {
                'field': 'total_value',
                'type': 'quantitative',
                'title': f'monthly total {raw_value_field}'
            },
            'color': state['original_encoding'].get('color', {})
        }
        
    elif level == 'month':
        # drilldown to specified month, display daily data
        year_val = p.get('year')
        if not year_val:
            return {'success': False, 'error': 'Drilldown to month requires parent.year'}
        
        if value < 1 or value > 12:
            return {'success': False, 'error': f'Invalid month: {value}, should be 1-12'}
        
        # Vega's month() returns 0-11
        vega_month = value - 1
        
        # 1. filter to specified year and month
        new_transforms.append({
            'filter': f'year(datum.{raw_date_field}) == {year_val} && month(datum.{raw_date_field}) == {vega_month}',
            '_avs_tag': 'line_drilldown_time'
        })
        # 2. aggregate by day (if there are multiple records on the same day)
        new_transforms.append({
            'timeUnit': 'yearmonthdate',
            'field': raw_date_field,
            'as': 'day_date',
            '_avs_tag': 'line_drilldown_time'
        })
        
        # build groupby
        day_groupby = ['day_date']
        if group_field:
            day_groupby.append(group_field)
        
        new_transforms.append({
            'aggregate': [{'op': 'sum', 'field': raw_value_field, 'as': 'total_value'}],
            'groupby': day_groupby,
            '_avs_tag': 'line_drilldown_time'
        })
        
        state['parent'] = {'year': year_val, 'month': value}
        title_suffix = f'{year_val} year {value} month daily trend'
        
        # update encoding
        new_spec['encoding'] = {
            'x': {
                'field': 'day_date',
                'type': 'temporal',
                'title': 'date',
                'axis': {'format': '%m-%d'}
            },
            'y': {
                'field': 'total_value',
                'type': 'quantitative',
                'title': f'daily total {raw_value_field}'
            },
            'color': state['original_encoding'].get('color', {})
        }
        
    elif level == 'date':
        # usually no further drilldown to specific date
        return {'success': False, 'error': 'Daily data is already the most granular, cannot drill further'}
        
    else:
        return {'success': False, 'error': f'Invalid level: {level}, should be year/month/date'}
    
    # apply new transform (replace existing aggregate transform)
    new_spec['transform'] = new_transforms
    
    # update title
    new_spec['title'] = title_suffix
    
    # keep tooltip
    if 'tooltip' in state['original_encoding']:
        # simplify tooltip
        tooltip_list = [
            {'field': new_spec['encoding']['x']['field'], 'type': 'temporal', 'title': 'time'},
        ]
        if group_field:
            tooltip_list.append({'field': group_field, 'type': 'nominal', 'title': group_field})
        tooltip_list.append({'field': 'total_value', 'type': 'quantitative', 'title': raw_value_field, 'format': ',.0f'})
        new_spec['encoding']['tooltip'] = tooltip_list
    
    # save state
    new_spec['_line_drilldown_state'] = state
    
    return {
        'success': True,
        'operation': 'drilldown_line_time',
        'vega_spec': new_spec,
        'message': f'下钻到 {title_suffix}',
        'current_level': level,
        'parent': state.get('parent', {})
    }

@mcp.tool()
def reset_line_drilldown(vega_spec: Dict) -> Dict[str, Any]:
    """
    Reset line chart time drilldown, restore to initial annual view.
    
    Args:
        vega_spec: Vega-Lite spec
    
    Returns:
        restored view spec
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # get drilldown state
    state = new_spec.get('_line_drilldown_state')
    if not isinstance(state, dict):
        return {
            'success': True,
            'operation': 'reset_line_drilldown',
            'vega_spec': new_spec,
            'message': 'No drilldown performed, no need to reset'
        }
    
    # restore original transform
    original_transform = state.get('original_transform')
    if original_transform is not None:
        new_spec['transform'] = copy.deepcopy(original_transform)
    else:
        # if original transform is not saved, remove drilldown related transform
        if 'transform' in new_spec:
            new_spec['transform'] = [
                t for t in new_spec['transform']
                if not (isinstance(t, dict) and t.get('_avs_tag') == 'line_drilldown_time')
            ]
    
    # restore original encoding
    original_encoding = state.get('original_encoding')
    if original_encoding:
        new_spec['encoding'] = copy.deepcopy(original_encoding)
    
    # restore original title
    original_title = state.get('original_title')
    if original_title:
        new_spec['title'] = original_title
    
    # clear state
    if '_line_drilldown_state' in new_spec:
        del new_spec['_line_drilldown_state']
    
    return {
        'success': True,
        'operation': 'reset_line_drilldown',
        'vega_spec': new_spec,
        'message': 'Reset to initial annual view'
    }

@mcp.tool()
def resample_time(
    vega_spec: Dict,
    granularity: str,
    agg: str = "mean",
) -> Dict[str, Any]:
    """
    Time granularity switch (resampling): aggregate time series from fine to coarse granularity.
    
    Cognitive interaction necessity:
    - Daily data is too dense, need to aggregate to week/month to see trend
    - Conversely, annual data needs to drill down to month/day to see details
    
    Args:
        vega_spec: Vega-Lite spec
        granularity: target time granularity ("day" | "week" | "month" | "quarter" | "year")
        agg: aggregation method ("mean" | "sum" | "max" | "min" | "median"), default mean
    
    Returns:
        resampled view spec
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # supported granularity mapping to Vega-Lite timeUnit
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
    
    # supported aggregation methods
    ALLOWED_AGG = {"mean", "sum", "max", "min", "median", "count"}
    agg_lower = str(agg).lower().strip()
    if agg_lower not in ALLOWED_AGG:
        return {
            'success': False,
            'error': f'Unsupported agg: {agg}. Use one of {sorted(list(ALLOWED_AGG))}'
        }
    
    # get time field
    time_field = _get_time_field(new_spec)
    if not time_field:
        return {'success': False, 'error': 'Cannot find temporal field in encoding'}
    
    # save original state for restoration
    state = new_spec.get('_resample_state')
    if not isinstance(state, dict):
        state = {}
    if 'original_encoding' not in state:
        if 'layer' in new_spec and len(new_spec['layer']) > 0:
            state['original_encoding'] = copy.deepcopy(new_spec['layer'][0].get('encoding', {}))
        else:
            state['original_encoding'] = copy.deepcopy(new_spec.get('encoding', {}))
    
    # determine time axis and value axis
    if 'layer' in new_spec and len(new_spec['layer']) > 0:
        encoding = new_spec['layer'][0].get('encoding', {})
    else:
        encoding = new_spec.get('encoding', {})
    
    x_enc = encoding.get('x', {})
    y_enc = encoding.get('y', {})
    
    # determine time on which axis
    if x_enc.get('field') == time_field or x_enc.get('type') == 'temporal':
        time_axis = 'x'
        value_axis = 'y'
    else:
        time_axis = 'y'
        value_axis = 'x'
    
    # modify time axis timeUnit
    def _update_encoding(enc: Dict) -> None:
        if time_axis in enc:
            enc[time_axis]['timeUnit'] = target_timeunit
            enc[time_axis]['type'] = 'temporal'
        
        # add aggregation on value axis
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
    
    # save state
    state['current_granularity'] = granularity_lower
    state['current_agg'] = agg_lower
    new_spec['_resample_state'] = state
    
    return {
        'success': True,
        'operation': 'resample_time',
        'vega_spec': new_spec,
        'message': f'Resampled time to {granularity} with {agg} aggregation'
    }

@mcp.tool()
def reset_resample(vega_spec: Dict) -> Dict[str, Any]:
    """
    Reset time resampling, restore to original granularity.
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


    
# ==================== Scatter Plot Tools ====================
@mcp.tool()
def select_region(vega_spec: Dict, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    """Select points in a specific region"""
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required fields'}
    
    data = new_spec.get('data', {}).get('values', [])
    selected_count = sum(1 for row in data if 
                         row.get(x_field) is not None and row.get(y_field) is not None and
                         x_range[0] <= row[x_field] <= x_range[1] and
                         y_range[0] <= row[y_field] <= y_range[1])
    
    # add highlight (bracket notation for field names with spaces)
    xr, yr = _datum_ref(x_field), _datum_ref(y_field)
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': (f'{xr} >= {x_range[0]} && {xr} <= {x_range[1]} && '
                     f'{yr} >= {y_range[0]} && {yr} <= {y_range[1]}'),
            'value': 1.0
        },
        'value': 0.2
    }
    
    return {
        'success': True,
        'operation': 'select_region',
        'vega_spec': new_spec,
        'selected_count': selected_count,
        'message': f'Selected {selected_count} points'
    }

@mcp.tool()
def identify_clusters(vega_spec: Dict, n_clusters: int = 3, method: str = "kmeans") -> Dict[str, Any]:
    """Identify data clusters"""
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

@mcp.tool()
def calculate_correlation(vega_spec: Dict, method: str = "pearson") -> Dict[str, Any]:
    """Calculate correlation coefficient"""
    x_field = vega_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = vega_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find required fields'}
    
    data = vega_spec.get('data', {}).get('values', [])
    
    x_values = [row[x_field] for row in data if row.get(x_field) is not None and row.get(y_field) is not None]
    y_values = [row[y_field] for row in data if row.get(x_field) is not None and row.get(y_field) is not None]
    
    if len(x_values) < 2:
        return {'success': False, 'error': 'Not enough data points'}
    
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
        'message': f'{method} correlation: {correlation:.3f} ({strength} {direction})'
    }

@mcp.tool()
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

@mcp.tool()
def filter_categorical(vega_spec: Dict, categories_to_remove: List[str], field: str = None) -> Dict[str, Any]:
    """
    Filter out data points of specified categories
    
    Args:
        vega_spec: Vega-Lite spec
        categories_to_remove: list of categories to remove
        field: category field name (optional, automatically detect color field)
    """
    import json
    new_spec = copy.deepcopy(vega_spec)
    
    # automatically detect category field
    if field is None:
        encoding = new_spec.get('encoding', {})
        color_enc = encoding.get('color', {})
        field = color_enc.get('field')
        
        if not field:
            # try to get from shape field
            shape_enc = encoding.get('shape', {})
            field = shape_enc.get('field')
    
    if not field:
        return {
            'success': False,
            'error': 'Cannot find categorical field. Please specify field parameter.'
        }
    
    # add filter transform
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


@mcp.tool()
def brush_region(vega_spec: Dict, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Dict[str, Any]:
    """
    Brush select a specific region, data points outside the region become fainter
    
    Args:
        vega_spec: Vega-Lite spec
        x_range: x-axis range (min, max)
        y_range: y-axis range (min, max)
    """
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x or y fields'}
    
    # implement brush selection effect (bracket notation for field names with spaces)
    xr, yr = _datum_ref(x_field), _datum_ref(y_field)
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': (f'{xr} >= {x_range[0]} && {xr} <= {x_range[1]} && '
                     f'{yr} >= {y_range[0]} && {yr} <= {y_range[1]}'),
            'value': 1.0
        },
        'value': 0.15
    }
    
    return {
        'success': True,
        'operation': 'brush_region',
        'vega_spec': new_spec,
        'message': f'Brushed region x:[{x_range[0]}, {x_range[1]}], y:[{y_range[0]}, {y_range[1]}]'
    }


@mcp.tool()
def show_regression(vega_spec: Dict, method: str = "linear") -> Dict[str, Any]:
    """
    Add regression line
    
    Args:
        vega_spec: Vega-Lite spec
        method: regression method ("linear", "log", "exp", "poly", "quad")
    """
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x or y fields'}
    
    # if original spec has no layer, convert to layer structure
    if 'layer' not in new_spec:
        original_spec = copy.deepcopy(new_spec)
        new_spec['layer'] = [{
            'mark': original_spec.get('mark', 'point'),
            'encoding': original_spec.get('encoding', {})
        }]
        # remove top-level mark and encoding
        if 'mark' in new_spec:
            del new_spec['mark']
        if 'encoding' in new_spec:
            del new_spec['encoding']
    
    # add regression line layer
    regression_transform = {
        'regression': y_field,
        'on': x_field
    }
    
    # set regression parameters according to method
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


# ==================== Heatmap Tools ====================
@mcp.tool()
def adjust_color_scale(vega_spec: Dict, scheme: str = "viridis") -> Dict[str, Any]:
    """Adjust color scale"""
    new_spec = copy.deepcopy(vega_spec)
    
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    if 'color' not in new_spec['encoding']:
        new_spec['encoding']['color'] = {}
    
    new_spec['encoding']['color']['scale'] = {'scheme': scheme}
    
    return {
        'success': True,
        'operation': 'adjust_color_scale',
        'vega_spec': new_spec,
        'message': f'Changed color scheme to {scheme}'
    }

@mcp.tool()
def filter_cells(
    vega_spec: Dict,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Filter heatmap cells by value. Provide at least one of min_value or max_value (single-sided interval).
    Only min: keep cells >= min_value. Only max: keep cells <= max_value. Both: keep [min, max].
    """
    if min_value is None and max_value is None:
        return {'success': False, 'error': 'Must provide at least one of min_value or max_value'}

    new_spec = copy.deepcopy(vega_spec)

    color_enc = new_spec.get('encoding', {}).get('color', {}) or {}
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
    parts = []
    if min_value is not None:
        parts.append(f'{ref} >= {float(min_value)}')
    if max_value is not None:
        parts.append(f'{ref} <= {float(max_value)}')
    filter_expr = ' && '.join(parts)

    new_spec['transform'].append({'filter': filter_expr})

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

@mcp.tool()
def highlight_region(
    vega_spec: Dict,
    x_values: Optional[List] = None,
    y_values: Optional[List] = None,
) -> Dict[str, Any]:
    """Highlight region. Provide at least one of x_values or y_values (only x=highlight columns, only y=highlight rows, both=highlight intersection)."""
    new_spec = copy.deepcopy(vega_spec)
    
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x/y fields'}
    
    x_vals = x_values if x_values is not None else []
    y_vals = y_values if y_values is not None else []
    if not x_vals and not y_vals:
        return {'success': False, 'error': 'Must provide at least one of x_values or y_values'}
    
    parts = []
    if x_vals:
        x_str = ','.join([json.dumps(v, ensure_ascii=False) for v in x_vals])
        xr = _datum_ref(x_field)
        parts.append(f'indexof([{x_str}], {xr}) >= 0')
    if y_vals:
        y_str = ','.join([json.dumps(v, ensure_ascii=False) for v in y_vals])
        yr = _datum_ref(y_field)
        parts.append(f'indexof([{y_str}], {yr}) >= 0')
    test_expr = ' && '.join(parts)
    
    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    
    new_spec['encoding']['opacity'] = {
        'condition': {
            'test': test_expr,
            'value': 1.0
        },
        # lower opacity for non-selected cells for clearer contrast
        'value': 0.15
    }
    
    return {
        'success': True,
        'operation': 'highlight_region',
        'vega_spec': new_spec,
        'message': 'Highlighted specified region'
    }

@mcp.tool()
def highlight_region_by_value(
    vega_spec: Dict,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    outside_opacity: float = 0.12,
) -> Dict[str, Any]:
    """
    Highlight cells by displayed cell value (typically the color-encoded value / aggregation).
    This is visual-only: dims outside range without deleting data (no transform.filter).
    Supports one-sided threshold by providing only min_value or max_value.
    """
    if min_value is None and max_value is None:
        return {'success': False, 'error': 'Must provide at least one of min_value or max_value'}

    new_spec = copy.deepcopy(vega_spec)

    color_enc = new_spec.get('encoding', {}).get('color', {}) or {}
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

    ref = _datum_ref(value_field)
    parts = []
    if min_value is not None:
        parts.append(f'{ref} >= {float(min_value)}')
    if max_value is not None:
        parts.append(f'{ref} <= {float(max_value)}')
    test_expr = ' && '.join(parts) if parts else 'true'

    if 'encoding' not in new_spec:
        new_spec['encoding'] = {}
    new_spec['encoding']['opacity'] = {
        'condition': {'test': test_expr, 'value': 1.0},
        'value': float(outside_opacity)
    }

    return {
        'success': True,
        'operation': 'highlight_region_by_value',
        'vega_spec': new_spec,
        'message': f'Highlighted cells by value (min={min_value}, max={max_value}); outside_opacity={outside_opacity}'
    }

@mcp.tool()
def filter_cells_by_region(
    vega_spec: Dict,
    x_value: Any = None,
    y_value: Any = None,
    x_values: Optional[List[Any]] = None,
    y_values: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Filter out heatmap cells by x/y coordinates (adds transform.filter to exclude matching cells).
    Provide at least one of x or y (only x=filter columns, only y=filter rows, both=filter intersection).
    Single cell: x_value + y_value. Multiple (cartesian product): x_values + y_values.
    """
    new_spec = copy.deepcopy(vega_spec)
    x_field = new_spec.get('encoding', {}).get('x', {}).get('field')
    y_field = new_spec.get('encoding', {}).get('y', {}).get('field')
    x_timeunit = new_spec.get('encoding', {}).get('x', {}).get('timeUnit')
    y_timeunit = new_spec.get('encoding', {}).get('y', {}).get('timeUnit')
    if not x_field or not y_field:
        return {'success': False, 'error': 'Cannot find x/y fields'}

    if x_values is None and x_value is not None:
        x_values = [x_value]
    if y_values is None and y_value is not None:
        y_values = [y_value]
    if (not x_values or len(x_values) == 0) and (not y_values or len(y_values) == 0):
        return {'success': False, 'error': 'Must provide at least one of x_value/x_values or y_value/y_values'}

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

    exclude_parts = []
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

@mcp.tool()
def cluster_rows_cols(vega_spec: Dict, cluster_rows: bool = True, 
                     cluster_cols: bool = True, method: str = "sum") -> Dict[str, Any]:
    """
    Reorder heatmap rows/cols by aggregate (sum/mean/max) of color field.
    Sets encoding.y.sort / encoding.x.sort; higher aggregate rows/cols move to top/left.
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

@mcp.tool()
def select_submatrix(vega_spec: Dict, x_values: List = None, 
                    y_values: List = None) -> Dict[str, Any]:
    """Select submatrix"""
    if not x_values and not y_values:
        return {'success': False, 'error': 'Must specify x_values or y_values'}
    
    new_spec = copy.deepcopy(vega_spec)
    
    # month name to number mapping (Vega month starts from 0: 0=Jan, 11=Dec)
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
    
    # process X axis filtering
    if x_values and x_field:
        if x_timeunit:
            # has timeUnit, use Vega expression function
            if x_timeunit == 'date':
                # extract date (1-31)
                x_nums = ','.join([str(int(v)) for v in x_values])
                filters.append(f'indexof([{x_nums}], date(datum.{x_field})) >= 0')
            elif x_timeunit == 'month':
                # extract month, try to convert month name to number
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
                # other timeUnit, use function name directly
                x_str = ','.join([f'"{v}"' for v in x_values])
                filters.append(f'indexof([{x_str}], {x_timeunit}(datum.{x_field})) >= 0')
        else:
            # no timeUnit, match field value directly
            x_str = ','.join([f'"{v}"' for v in x_values])
            filters.append(f'indexof([{x_str}], datum.{x_field}) >= 0')
    
    # process Y axis filtering
    if y_values and y_field:
        if y_timeunit:
            # has timeUnit, use Vega expression function
            if y_timeunit == 'date':
                y_nums = ','.join([str(int(v)) for v in y_values])
                filters.append(f'indexof([{y_nums}], date(datum.{y_field})) >= 0')
            elif y_timeunit == 'month':
                # extract month, try to convert month name to number
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
                # other timeUnit, use function name directly
                y_str = ','.join([f'"{v}"' for v in y_values])
                filters.append(f'indexof([{y_str}], {y_timeunit}(datum.{y_field})) >= 0')
        else:
            # no timeUnit, match field value directly
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

@mcp.tool()
def find_extremes(vega_spec: Dict, top_n: int = 5, mode: str = "both") -> Dict[str, Any]:
    """
    Mark extreme points
    
    Args:
        vega_spec: Vega-Lite spec
        top_n: mark top N extreme values
        mode: "max" | "min" | "both"
    """
    new_spec = copy.deepcopy(vega_spec)
    
    # get field information
    encoding = new_spec.get('encoding', {})
    x_field = encoding.get('x', {}).get('field')
    y_field = encoding.get('y', {}).get('field')
    color_field = encoding.get('color', {}).get('field')
    
    if not color_field:
        return {'success': False, 'error': 'Cannot find color field for finding extremes'}
    
    # get data
    data = new_spec.get('data', {}).get('values', [])
    if not data:
        return {'success': False, 'error': 'No data found'}
    
    # sort by value to find extreme values
    sorted_data = sorted(
        [d for d in data if d.get(color_field) is not None],
        key=lambda x: x.get(color_field, 0)
    )
    
    extremes = []
    if mode in ["max", "both"]:
        extremes.extend(sorted_data[-top_n:])
    if mode in ["min", "both"]:
        extremes.extend(sorted_data[:top_n])
    
    # remove duplicates
    extremes = list({id(e): e for e in extremes}.values())
    
    # build extreme point coordinate conditions (bracket notation for field names with spaces)
    dx, dy = _datum_ref(x_field), _datum_ref(y_field)
    extreme_conditions = []
    for e in extremes:
        x_val = e.get(x_field)
        y_val = e.get(y_field)
        if x_val is not None and y_val is not None:
            if isinstance(x_val, str):
                xq, yq = json.dumps(x_val), json.dumps(y_val)
                extreme_conditions.append(f'({dx} === {xq} && {dy} === {yq})')
            else:
                extreme_conditions.append(f'({dx} === {x_val} && {dy} === {y_val})')
    
    if not extreme_conditions:
        return {'success': False, 'error': 'No extreme values found'}
    
    # use strokeWidth to mark extreme points; transparent avoids Vega signal issues
    test_expr = ' || '.join(extreme_conditions)
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
    
    # return extreme information
    extreme_info = []
    for e in extremes:
        extreme_info.append({
            'x': e.get(x_field),
            'y': e.get(y_field),
            'value': e.get(color_field)
    })
    
    return {
        'success': True,
        'operation': 'find_extremes',
        'vega_spec': new_spec,
        'extremes': extreme_info,
        'message': f'Marked {len(extremes)} extreme points (mode: {mode})'
    }

@mcp.tool()
def threshold_mask(
    vega_spec: Dict,
    min_value: float,
    max_value: float,
    outside_opacity: float = 0.1,
) -> Dict[str, Any]:
    """
    Mask cells that are not in the threshold range (dimmed) but do not delete data.
    
    Args:
        vega_spec: Vega-Lite spec
        min_value: lower threshold (inclusive)
        max_value: upper threshold (inclusive)
        outside_opacity: opacity outside the range
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

@mcp.tool()
def drilldown_time(
    vega_spec: Dict,
    level: str,
    value: Union[int, str],
    parent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Time heatmap drilldown: year -> month -> date
    
    Conventions:
    - Time field in encoding.x.field, and encoding.x.type == 'temporal'
    - Initial suggestion timeUnit='year' (if not specified, can be recorded and restored in reset)
    
    Args:
        vega_spec: Vega-Lite spec
        level: 'year' | 'month' | 'date'
        value: value corresponding to level (year=int; month=1-12; date=1-31)
        parent: optional parent information, e.g. {'year': 2012} or {'year':2012,'month':3}
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

@mcp.tool()
def reset_drilldown(vega_spec: Dict) -> Dict[str, Any]:
    """
    Reset time heatmap drilldown: remove filters added by drilldown_time, and restore original x encoding (timeUnit, etc.).
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

@mcp.tool()
def add_marginal_bars(
    vega_spec: Dict,
    op: str = "mean",
    show_top: bool = True,
    show_right: bool = True,
    bar_size: int = 70,
    bar_color: str = "#666666",
) -> Dict[str, Any]:
    """
    Add marginal bar charts (row/col aggregation) to heatmap. Top = aggregate by x, right = by y.
    Uses shared x/y scales with main heatmap. Defaults width/height to 400x300 when missing to avoid blank layout.
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

@mcp.tool()
def transpose(vega_spec: Dict) -> Dict[str, Any]:
    """
    Transpose heatmap: swap x and y axes.
    
    Necessity of interaction:
    - Sometimes the arrangement of data rows and columns does not conform to analysis habits, so quick switching of perspective is needed
    - For example, change "view by month" to "view by region"
    
    Args:
        vega_spec: Vega-Lite spec
    
    Returns:
        transposed spec
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
    
    # swap x and y encoding
    new_spec['encoding']['x'] = copy.deepcopy(y_enc)
    new_spec['encoding']['y'] = copy.deepcopy(x_enc)
    
    # swap width and height (if defined)
    width = new_spec.get('width')
    height = new_spec.get('height')
    if width is not None and height is not None:
        new_spec['width'] = height
        new_spec['height'] = width
    
    # record transpose state (for switching back)
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

# ==================== Sankey Tools ====================
@mcp.tool()
def _find_sankey_data(vega_spec: Dict) -> Tuple[Optional[List], Optional[List], Optional[int], Optional[int]]:
    """
    Find nodes and links data from Vega spec
    
    Returns:
        (nodes_values, links_values, nodes_index, links_index)
    """
    data = vega_spec.get("data", [])
    if not isinstance(data, list):
        return None, None, None, None
    
    nodes_data = None
    links_data = None
    nodes_idx = None
    links_idx = None
    
    for i, d in enumerate(data):
        if not isinstance(d, dict):
            continue
        name = d.get("name", "")
        if name == "nodes":
            nodes_data = d.get("values", [])
            nodes_idx = i
        elif name == "links":
            links_data = d.get("values", [])
            links_idx = i
    
    return nodes_data, links_data, nodes_idx, links_idx

@mcp.tool()
def _recalculate_layout(nodes: List[Dict], links: List[Dict], 
                        width: int = 900, height: int = 400) -> Tuple[List[Dict], List[Dict]]:
    """
    Recalculate node positions and connection coordinates
    Simplified version: sort by depth, evenly distribute each layer
    """
    if not nodes or not links:
        return nodes, links
    
    # group by depth
    depth_groups = {}
    for node in nodes:
        depth = node.get("depth", 0)
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node)
    
    max_depth = max(depth_groups.keys()) if depth_groups else 0
    
    # calculate total flow for each node
    node_values = {}
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        node_values[src] = node_values.get(src, 0) + val
        node_values[tgt] = node_values.get(tgt, 0) + val
    
    # recalculate node positions
    x_step = width / (max_depth + 1) if max_depth > 0 else width
    
    new_nodes = []
    for depth, group in sorted(depth_groups.items()):
        x = depth * x_step
        total_height = sum(node_values.get(n["name"], 10) for n in group)
        scale = (height - 50) / total_height if total_height > 0 else 1
        
        y = 10
        for node in group:
            dy = max(10, node_values.get(node["name"], 10) * scale)
            new_node = copy.deepcopy(node)
            new_node["x"] = x
            new_node["y"] = y
            new_node["dy"] = dy
            new_nodes.append(new_node)
            y += dy + 5
    
    # create node position mapping
    node_pos = {n["name"]: n for n in new_nodes}
    
    # recalculate connection coordinates
    # track current y position used by each node
    node_out_y = {n["name"]: n["y"] for n in new_nodes}
    node_in_y = {n["name"]: n["y"] for n in new_nodes}
    
    new_links = []
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        if src not in node_pos or tgt not in node_pos:
            continue
        
        src_node = node_pos[src]
        tgt_node = node_pos[tgt]
        
        # calculate connection height
        total_out = node_values.get(src, 1)
        total_in = node_values.get(tgt, 1)
        link_height_src = (val / total_out) * src_node["dy"] if total_out > 0 else 10
        link_height_tgt = (val / total_in) * tgt_node["dy"] if total_in > 0 else 10
        
        new_link = copy.deepcopy(link)
        new_link["sy0"] = node_out_y[src]
        new_link["sy1"] = node_out_y[src] + link_height_src
        new_link["ty0"] = node_in_y[tgt]
        new_link["ty1"] = node_in_y[tgt] + link_height_tgt
        
        node_out_y[src] += link_height_src
        node_in_y[tgt] += link_height_tgt
        
        new_links.append(new_link)
    
    return new_nodes, new_links

@mcp.tool()
def filter_flow(vega_spec: Dict, min_value: float) -> Dict[str, Any]:
    """
    Filter flow: only show connections with value >= min_value
    
    Args:
        vega_spec: Vega spec
        min_value: minimum flow threshold
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if links is None or links_idx is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # filter connections
    filtered_links = [l for l in links if l.get("value", 0) >= min_value]
    
    if not filtered_links:
        return {
            'success': False,
            'error': f'No links with value >= {min_value}'
        }
    
    # find nodes still in use
    used_nodes = set()
    for link in filtered_links:
        used_nodes.add(link.get("source"))
        used_nodes.add(link.get("target"))
    
    # filter nodes
    if nodes is not None and nodes_idx is not None:
        filtered_nodes = [n for n in nodes if n.get("name") in used_nodes]
        # recalculate layout
        filtered_nodes, filtered_links = _recalculate_layout(filtered_nodes, filtered_links)
        new_spec["data"][nodes_idx]["values"] = filtered_nodes
    
    new_spec["data"][links_idx]["values"] = filtered_links
    
    return {
        'success': True,
        'operation': 'filter_flow',
        'vega_spec': new_spec,
        'message': f'Filtered to {len(filtered_links)} links with value >= {min_value}'
    }

@mcp.tool()     
def highlight_path(vega_spec: Dict, path: List[str]) -> Dict[str, Any]:
    """
    Highlight multi-step path: reduce transparency of other connections
    
    Args:
        vega_spec: Vega spec
        path: node path list, e.g. ["Google Ads 1", "Homepage", "Electronics", "Purchase"]
              highlight all adjacent nodes between connections
    """
    if not path or len(path) < 2:
        return {
            'success': False,
            'error': 'Path must contain at least 2 nodes'
        }
    
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # build set of edges to highlight
    highlight_edges = set()
    missing_edges = []
    
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]
        # check if this edge exists
        edge_exists = any(
            l.get("source") == src and l.get("target") == tgt 
            for l in links
        )
        if edge_exists:
            highlight_edges.add((src, tgt))
        else:
            missing_edges.append(f"{src} -> {tgt}")
    
    if not highlight_edges:
        return {
            'success': False,
            'error': f'No valid edges found in path. Missing: {missing_edges}'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # build highlight condition expression
    conditions = [f"(parent.source == '{src}' && parent.target == '{tgt}')" 
                  for src, tgt in highlight_edges]
    highlight_condition = " || ".join(conditions)
    
    # modify fillOpacity of path in marks
    marks = new_spec.get("marks", [])
    for mark in marks:
        if mark.get("type") == "group":
            inner_marks = mark.get("marks", [])
            for inner in inner_marks:
                if inner.get("type") == "path":
                    encode = inner.setdefault("encode", {})
                    enter = encode.setdefault("enter", {})
                    enter["fillOpacity"] = {
                        "signal": f"({highlight_condition}) ? 0.9 : 0.15"
                    }
    
    # build path description
    path_desc = " -> ".join(path)
    warning = f" (Note: edges not found: {missing_edges})" if missing_edges else ""
    
    return {
        'success': True,
        'operation': 'highlight_path',
        'vega_spec': new_spec,
        'message': f'Highlighted path: {path_desc}{warning}',
        'highlighted_edges': len(highlight_edges),
        'total_edges_in_path': len(path) - 1
    }


@mcp.tool()
def calculate_conversion_rate(vega_spec: Dict, node_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate conversion rate: analyze inflow, outflow, and conversion rate for each node
    
    Args:
        vega_spec: Vega spec
        node_name: specified node name (optional). If not specified, return conversion rate for all nodes
    
    Returns:
        dictionary containing conversion rate analysis results
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # calculate inflow and outflow for each node
    node_inflow = {}   # inflow to this node
    node_outflow = {}  # outflow from this node
    
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        # outflow from source node
        node_outflow[src] = node_outflow.get(src, 0) + val
        # inflow to target node
        node_inflow[tgt] = node_inflow.get(tgt, 0) + val
    
    # get all nodes
    all_nodes = set(node_inflow.keys()) | set(node_outflow.keys())
    
    # calculate conversion rate for each node
    conversions = []
    for name in sorted(all_nodes):
        inflow = node_inflow.get(name, 0)
        outflow = node_outflow.get(name, 0)
        
        # calculate conversion rate
        if inflow > 0:
            rate = outflow / inflow
        elif outflow > 0:
            rate = float('inf')  # 源节点（只有出流）
        else:
            rate = 0
        
        # determine node type
        if inflow == 0 and outflow > 0:
            node_type = "source"  # source node (start)
        elif outflow == 0 and inflow > 0:
            node_type = "sink"    # sink node (end)
        else:
            node_type = "intermediate"  # intermediate node
        
        conversion = {
            "node": name,
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
            "rate": round(rate, 4) if rate != float('inf') else "source",
            "type": node_type
        }
        
        # calculate loss
        if node_type == "intermediate" and inflow > 0:
            loss = inflow - outflow
            loss_rate = loss / inflow if inflow > 0 else 0
            conversion["loss"] = round(loss, 2)
            conversion["loss_rate"] = round(loss_rate, 4)
        
        conversions.append(conversion)
    
    # if specified node, only return that node
    if node_name:
        target_conversion = [c for c in conversions if c["node"] == node_name]
        if not target_conversion:
            return {
                'success': False,
                'error': f'Node "{node_name}" not found'
            }
        
        # return upstream and downstream of the node
        upstream = [link for link in links if link.get("target") == node_name]
        downstream = [link for link in links if link.get("source") == node_name]
        
        return {
            'success': True,
            'operation': 'calculate_conversion_rate',
            'node': node_name,
            'conversion': target_conversion[0],
            'upstream': [{"from": l["source"], "value": l["value"]} for l in upstream],
            'downstream': [{"to": l["target"], "value": l["value"]} for l in downstream],
            'message': f'Conversion analysis for {node_name}'
        }
    
    # group by type
    sources = [c for c in conversions if c["type"] == "source"]
    sinks = [c for c in conversions if c["type"] == "sink"]
    intermediates = [c for c in conversions if c["type"] == "intermediate"]
    
    # find nodes with highest loss
    high_loss_nodes = sorted(
        [c for c in intermediates if "loss_rate" in c and c["loss_rate"] > 0],
        key=lambda x: x["loss_rate"],
        reverse=True
    )[:5]
    
    return {
        'success': True,
        'operation': 'calculate_conversion_rate',
        'summary': {
            'total_nodes': len(all_nodes),
            'source_nodes': len(sources),
            'sink_nodes': len(sinks),
            'intermediate_nodes': len(intermediates)
        },
        'conversions': conversions,
        'high_loss_nodes': high_loss_nodes,
        'message': f'Calculated conversion rates for {len(all_nodes)} nodes'
    }

@mcp.tool()
def trace_node(vega_spec: Dict, node_name: str) -> Dict[str, Any]:
    """
    Trace node: highlight all connections connected to the node
    
    Args:
        vega_spec: Vega spec
        node_name: node name
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # check if node exists
    node_exists = any(
        l.get("source") == node_name or l.get("target") == node_name 
        for l in links
    )
    
    if not node_exists:
        return {
            'success': False,
            'error': f'Node "{node_name}" not found in links'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # modify fillOpacity of path in marks
    marks = new_spec.get("marks", [])
    for mark in marks:
        if mark.get("type") == "group":
            inner_marks = mark.get("marks", [])
            for inner in inner_marks:
                if inner.get("type") == "path":
                    encode = inner.setdefault("encode", {})
                    enter = encode.setdefault("enter", {})
                    enter["fillOpacity"] = {
                        "signal": f"parent.source == '{node_name}' || parent.target == '{node_name}' ? 0.9 : 0.15"
                    }
    
    # highlight node itself
    for mark in marks:
        if mark.get("type") == "rect":
            encode = mark.setdefault("encode", {})
            enter = encode.setdefault("enter", {})
            enter["fillOpacity"] = {
                "signal": f"datum.name == '{node_name}' ? 1.0 : 0.4"
    }
    
    return {
        'success': True,
        'operation': 'trace_node',
        'vega_spec': new_spec,
        'message': f'Traced all connections of node: {node_name}'
    }

@mcp.tool()
def collapse_nodes(vega_spec: Dict, nodes_to_collapse: List[str], 
                   aggregate_name: str = "Other") -> Dict[str, Any]:
    """
    Collapse multiple nodes: merge specified nodes into a single aggregate node
    
    Args:
        vega_spec: Vega spec
        nodes_to_collapse: list of node names to collapse
        aggregate_name: name of aggregate node (default "Other")
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if nodes is None or links is None:
        return {
            'success': False,
            'error': 'Cannot find nodes or links data in Sankey spec'
        }
    
    collapse_set = set(nodes_to_collapse)
    
    # check if nodes exist
    existing_names = {n.get("name") for n in nodes}
    missing = collapse_set - existing_names
    if missing:
        return {
            'success': False,
            'error': f'Nodes not found: {list(missing)}'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # save original state for expand
    if "_sankey_state" not in new_spec:
        new_spec["_sankey_state"] = {
            "original_nodes": copy.deepcopy(nodes),
            "original_links": copy.deepcopy(links),
            "collapsed_groups": {}
        }
    
    # record collapsed groups
    new_spec["_sankey_state"]["collapsed_groups"][aggregate_name] = list(nodes_to_collapse)
    
    # find depth of collapsed nodes (take first depth)
    collapse_depth = None
    for node in nodes:
        if node.get("name") in collapse_set:
            collapse_depth = node.get("depth", 0)
            break
    
    # create new node list
    new_nodes = [n for n in nodes if n.get("name") not in collapse_set]
    
    # add aggregate node
    aggregate_node = {
        "name": aggregate_name,
        "depth": collapse_depth or 0,
        "x": 0,
        "y": 0,
        "dy": 50,
        "_is_aggregate": True,
        "_collapsed_nodes": list(nodes_to_collapse)
    }
    new_nodes.append(aggregate_node)
    
    # redirect connections
    new_links = []
    link_aggregates = {}  # (source, target) -> total_value
    
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        # replace collapsed nodes
        new_src = aggregate_name if src in collapse_set else src
        new_tgt = aggregate_name if tgt in collapse_set else tgt
        
        # skip connections inside aggregate node
        if new_src == aggregate_name and new_tgt == aggregate_name:
            continue
        
        key = (new_src, new_tgt)
        if key in link_aggregates:
            link_aggregates[key] += val
        else:
            link_aggregates[key] = val
    
    for (src, tgt), val in link_aggregates.items():
        new_links.append({
            "source": src,
            "target": tgt,
            "value": val
        })
    
    # recalculate layout
    new_nodes, new_links = _recalculate_layout(new_nodes, new_links)
    
    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = new_links
    
    return {
        'success': True,
        'operation': 'collapse_nodes',
        'vega_spec': new_spec,
        'message': f'Collapsed {len(nodes_to_collapse)} nodes into "{aggregate_name}"'
    }

@mcp.tool()
def expand_node(vega_spec: Dict, aggregate_name: str) -> Dict[str, Any]:
    """
    Expand aggregate node: restore collapsed original nodes
    
    Args:
        vega_spec: Vega spec
        aggregate_name: name of aggregate node to expand
    """
    state = vega_spec.get("_sankey_state", {})
    collapsed_groups = state.get("collapsed_groups", {})
    
    if aggregate_name not in collapsed_groups:
        return {
            'success': False,
            'error': f'"{aggregate_name}" is not a collapsed group'
        }
    
    original_nodes = state.get("original_nodes")
    original_links = state.get("original_links")
    
    if not original_nodes or not original_links:
        return {
            'success': False,
            'error': 'Original data not found, cannot expand'
        }
    
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if nodes is None or links is None:
        return {
            'success': False,
            'error': 'Cannot find nodes or links data'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # get collapsed nodes
    collapsed_node_names = set(collapsed_groups[aggregate_name])
    
    # remove aggregate node, add back original nodes
    new_nodes = [n for n in nodes if n.get("name") != aggregate_name]
    for orig_node in original_nodes:
        if orig_node.get("name") in collapsed_node_names:
            new_nodes.append(copy.deepcopy(orig_node))
    
    # restore original connections
    new_links = []
    current_node_names = {n.get("name") for n in new_nodes}
    
    for orig_link in original_links:
        src = orig_link.get("source")
        tgt = orig_link.get("target")
        if src in current_node_names and tgt in current_node_names:
            new_links.append(copy.deepcopy(orig_link))
    
    # keep other (non-expand related) connections
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        if src != aggregate_name and tgt != aggregate_name:
            if src not in collapsed_node_names and tgt not in collapsed_node_names:
                # check if already exists
                exists = any(
                    l.get("source") == src and l.get("target") == tgt 
                    for l in new_links
                )
                if not exists:
                    new_links.append(copy.deepcopy(link))
    
    # recalculate layout
    new_nodes, new_links = _recalculate_layout(new_nodes, new_links)
    
    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = new_links
    
    # remove from collapsed groups
    del new_spec["_sankey_state"]["collapsed_groups"][aggregate_name]
    
    return {
        'success': True,
        'operation': 'expand_node',
        'vega_spec': new_spec,
        'message': f'Expanded "{aggregate_name}" back to {len(collapsed_node_names)} nodes'
    }

@mcp.tool()
def auto_collapse_by_rank(vega_spec: Dict, top_n: int = 5) -> Dict[str, Any]:
    """
    Auto collapse by rank: keep top N nodes per layer, collapse others to "Others (Layer X)"
    
    This is a core tool for physical interaction necessity:
    - Large Sankey charts (100+ nodes) cannot be rendered at once
    - Initially only show top N nodes per layer
    - Users must expand nodes to see collapsed nodes
    
    Args:
        vega_spec: Vega spec
        top_n: number of top nodes to keep per layer (default 5)
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if nodes is None or links is None:
        return {
            'success': False,
            'error': 'Cannot find nodes or links data in Sankey spec'
        }
    
    new_spec = copy.deepcopy(vega_spec)
    
    # save original state for expand
    if "_sankey_state" not in new_spec:
        new_spec["_sankey_state"] = {
            "original_nodes": copy.deepcopy(nodes),
            "original_links": copy.deepcopy(links),
            "collapsed_groups": {}
        }
    
    # calculate total flow for each node
    node_values = {}
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        node_values[src] = node_values.get(src, 0) + val
        node_values[tgt] = node_values.get(tgt, 0) + val
    
    # group by depth
    depth_groups = {}
    for node in nodes:
        depth = node.get("depth", 0)
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(node)
    
    # rank by flow for each layer
    nodes_to_keep = set()
    collapsed_by_layer = {}
    
    for depth, group in depth_groups.items():
        # sort by flow (descending)
        sorted_nodes = sorted(
            group, 
            key=lambda n: node_values.get(n.get("name"), 0), 
            reverse=True
        )
        
        # keep top N
        for node in sorted_nodes[:top_n]:
            nodes_to_keep.add(node.get("name"))
        
        # record collapsed nodes
        collapsed = [n.get("name") for n in sorted_nodes[top_n:]]
        if collapsed:
            aggregate_name = f"Others (Layer {depth})"
            collapsed_by_layer[depth] = {
                "aggregate_name": aggregate_name,
                "collapsed_nodes": collapsed
            }
            new_spec["_sankey_state"]["collapsed_groups"][aggregate_name] = collapsed
    
    # build new node list
    new_nodes = [n for n in nodes if n.get("name") in nodes_to_keep]
    
    # add aggregate nodes
    for depth, info in collapsed_by_layer.items():
        aggregate_node = {
            "name": info["aggregate_name"],
            "depth": depth,
            "x": 0,
            "y": 0,
            "dy": 30,
            "_is_aggregate": True,
            "_collapsed_nodes": info["collapsed_nodes"]
        }
        new_nodes.append(aggregate_node)
    
    # redirect connections
    new_links = []
    link_aggregates = {}
    
    # create node name to aggregate name mapping
    node_to_aggregate = {}
    for depth, info in collapsed_by_layer.items():
        for node_name in info["collapsed_nodes"]:
            node_to_aggregate[node_name] = info["aggregate_name"]
    
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        # replace collapsed nodes
        new_src = node_to_aggregate.get(src, src)
        new_tgt = node_to_aggregate.get(tgt, tgt)
        
        key = (new_src, new_tgt)
        if key in link_aggregates:
            link_aggregates[key] += val
        else:
            link_aggregates[key] = val
    
    for (src, tgt), val in link_aggregates.items():
        new_links.append({
            "source": src,
            "target": tgt,
            "value": val
        })
    
    # recalculate layout
    new_nodes, new_links = _recalculate_layout(new_nodes, new_links)
    
    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = new_links
    
    # count
    total_collapsed = sum(len(info["collapsed_nodes"]) for info in collapsed_by_layer.values())
    
    return {
        'success': True,
        'operation': 'auto_collapse_by_rank',
        'vega_spec': new_spec,
        'message': f'Kept top {top_n} nodes per layer, collapsed {total_collapsed} nodes into {len(collapsed_by_layer)} groups',
        'collapsed_groups': {info["aggregate_name"]: info["collapsed_nodes"] for info in collapsed_by_layer.values()}
    }

@mcp.tool()
def color_flows(vega_spec: Dict, nodes: List[str], color: str = "#e74c3c") -> Dict[str, Any]:
    """
    Color flows connected to specified nodes
    
    Args:
        vega_spec: Vega spec
        nodes: list of node names, flows connected to these nodes will be colored
        color: color to use for coloring (default red)
    """
    new_spec = copy.deepcopy(vega_spec)
    
    nodes_data, links, nodes_idx, links_idx = _find_sankey_data(new_spec)
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # find indices of edges connected to specified nodes
    colored_indices = []
    for i, link in enumerate(links):
        src = link.get("source")
        tgt = link.get("target")
        if src in nodes or tgt in nodes:
            colored_indices.append(i)
    
    if not colored_indices:
        return {
            'success': False,
            'error': f'No flows connected to nodes: {nodes}'
        }
    
    # find path mark in marks, add condition fill
    for mark in new_spec.get('marks', []):
        if mark.get('name') == 'links' or mark.get('type') == 'path':
            if 'encode' not in mark:
                mark['encode'] = {}
            if 'update' not in mark['encode']:
                mark['encode']['update'] = {}
            
            # use signal expression to set color
            indices_str = ','.join(str(i) for i in colored_indices)
            mark['encode']['update']['fill'] = [
                {
                    'test': f'indexof([{indices_str}], datum.index) >= 0',
                    'value': color
                },
                {'value': '#aaa'}
            ]
            break
    
    return {
        'success': True,
        'operation': 'color_flows',
        'vega_spec': new_spec,
        'colored_count': len(colored_indices),
        'message': f'Colored {len(colored_indices)} flows connected to nodes: {nodes}'
    }

@mcp.tool()
def find_bottleneck(vega_spec: Dict, top_n: int = 3) -> Dict[str, Any]:
    """
    Identify nodes with the most severe loss
    
    Args:
        vega_spec: Vega spec
        top_n: return top N nodes with the most severe loss
    """
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # calculate inflow and outflow for each node
    node_inflow = {}
    node_outflow = {}
    
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        node_outflow[src] = node_outflow.get(src, 0) + val
        node_inflow[tgt] = node_inflow.get(tgt, 0) + val
    
    # calculate loss for each intermediate node
    bottlenecks = []
    all_nodes = set(node_inflow.keys()) | set(node_outflow.keys())
    
    for node in all_nodes:
        inflow = node_inflow.get(node, 0)
        outflow = node_outflow.get(node, 0)
        
        # only consider intermediate nodes (nodes with both inflow and outflow)
        if inflow > 0 and outflow > 0:
            loss = inflow - outflow
            loss_rate = loss / inflow if inflow > 0 else 0
            
            if loss > 0:  # only count nodes with loss
                bottlenecks.append({
                    'node': node,
                    'inflow': round(inflow, 2),
                    'outflow': round(outflow, 2),
                    'loss': round(loss, 2),
                    'loss_rate': round(loss_rate, 4)
                })
    
    # sort by loss rate
    bottlenecks.sort(key=lambda x: x['loss_rate'], reverse=True)
    top_bottlenecks = bottlenecks[:top_n]
    
    if not top_bottlenecks:
        return {
            'success': True,
            'operation': 'find_bottleneck',
            'bottlenecks': [],
            'message': 'No bottlenecks found (no intermediate nodes with loss)'
        }
    
    return {
        'success': True,
        'operation': 'find_bottleneck',
        'bottlenecks': top_bottlenecks,
        'total_bottleneck_nodes': len(bottlenecks),
        'message': f'Found top {len(top_bottlenecks)} bottleneck nodes with highest loss rates'
    }

@mcp.tool()
def reorder_nodes_in_layer(
    vega_spec: Dict,
    depth: int,
    order: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reorder nodes in a layer of Sankey chart
    
    Physical interaction necessity:
    - Sankey chart is usually ordered by some default order, but users may want to reorder by importance, name, etc.
    - Reasonable node order can reduce edge crossings, improve readability
    
    Args:
        vega_spec: Vega spec
        depth: layer to reorder (0, 1, 2, ...)
        order: list of node names, in order from top to bottom. If specified, will be ordered by this order
        sort_by: sorting method (mutually exclusive with order):
            - "value_desc": sort by flow descending
            - "value_asc": sort by flow ascending
            - "name": sort by name alphabetically
    
    Returns:
        reordered spec
    """
    if order is None and sort_by is None:
        return {
            'success': False,
            'error': 'Must specify either "order" or "sort_by"'
        }
    
    if order is not None and sort_by is not None:
        return {
            'success': False,
            'error': 'Cannot specify both "order" and "sort_by". Use one or the other.'
        }
    
    nodes, links, nodes_idx, links_idx = _find_sankey_data(vega_spec)
    
    if nodes is None:
        return {
            'success': False,
            'error': 'Cannot find nodes data in Sankey spec'
        }
    
    if links is None:
        return {
            'success': False,
            'error': 'Cannot find links data in Sankey spec'
        }
    
    # find nodes at specified depth
    layer_nodes = [n for n in nodes if n.get("depth") == depth]
    if not layer_nodes:
        return {
            'success': False,
            'error': f'No nodes found at depth {depth}'
        }
    
    # calculate flow value for each node
    node_values = {}
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        node_values[src] = node_values.get(src, 0) + val
        node_values[tgt] = node_values.get(tgt, 0) + val
    
    # determine sorting method
    if order is not None:
        # sort by user specified order
        order_map = {name: i for i, name in enumerate(order)}
        sorted_layer = sorted(
            layer_nodes,
            key=lambda n: order_map.get(n.get("name"), 999999)
        )
    else:
        # sort by sort_by
        sort_by_lower = str(sort_by).lower().strip()
        if sort_by_lower == "value_desc":
            sorted_layer = sorted(
                layer_nodes,
                key=lambda n: node_values.get(n.get("name"), 0),
                reverse=True
            )
        elif sort_by_lower == "value_asc":
            sorted_layer = sorted(
                layer_nodes,
                key=lambda n: node_values.get(n.get("name"), 0)
            )
        elif sort_by_lower == "name":
            sorted_layer = sorted(
                layer_nodes,
                key=lambda n: n.get("name", "")
            )
        else:
            return {
                'success': False,
                'error': f'Invalid sort_by: {sort_by}. Use "value_desc", "value_asc", or "name"'
            }
    
    # get canvas size
    height = 400
    # try to get height from spec
    if isinstance(vega_spec.get("height"), (int, float)):
        height = int(vega_spec.get("height"))
    
    # calculate total height of this layer
    total_height = sum(node_values.get(n.get("name"), 10) for n in sorted_layer)
    scale = (height - 50) / total_height if total_height > 0 else 1
    
    # reassign y coordinates
    new_nodes = copy.deepcopy(nodes)
    layer_node_names = {n.get("name") for n in layer_nodes}
    
    y = 10
    name_to_new_y = {}
    name_to_new_dy = {}
    
    for node in sorted_layer:
        name = node.get("name")
        dy = max(10, node_values.get(name, 10) * scale)
        name_to_new_y[name] = y
        name_to_new_dy[name] = dy
        y += dy + 5
    
    # update node positions
    for node in new_nodes:
        name = node.get("name")
        if name in name_to_new_y:
            node["y"] = name_to_new_y[name]
            node["dy"] = name_to_new_dy[name]
    
    # create new node position mapping
    node_pos = {n.get("name"): n for n in new_nodes}
    
    # recalculate links coordinates related to this layer
    # need to track used y positions for each node (for stacking multiple links)
    node_out_y_offset = {n.get("name"): n.get("y", 0) for n in new_nodes}
    node_in_y_offset = {n.get("name"): n.get("y", 0) for n in new_nodes}
    
    new_links = []
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        val = link.get("value", 0)
        
        new_link = copy.deepcopy(link)
        
        # only update links related to nodes in this layer
        if src in layer_node_names or tgt in layer_node_names:
            src_node = node_pos.get(src)
            tgt_node = node_pos.get(tgt)
            
            if src_node and tgt_node:
                total_out = node_values.get(src, 1)
                total_in = node_values.get(tgt, 1)
                link_height_src = (val / total_out) * src_node.get("dy", 10) if total_out > 0 else 10
                link_height_tgt = (val / total_in) * tgt_node.get("dy", 10) if total_in > 0 else 10
                
                # update coordinates
                new_link["sy0"] = node_out_y_offset[src]
                new_link["sy1"] = node_out_y_offset[src] + link_height_src
                new_link["ty0"] = node_in_y_offset[tgt]
                new_link["ty1"] = node_in_y_offset[tgt] + link_height_tgt
                
                # accumulate offset (for next link)
                node_out_y_offset[src] += link_height_src
                node_in_y_offset[tgt] += link_height_tgt
        
        new_links.append(new_link)
    
    # build new spec
    new_spec = copy.deepcopy(vega_spec)
    data = new_spec.get("data", [])
    
    if nodes_idx is not None and isinstance(data, list) and nodes_idx < len(data):
        data[nodes_idx]["values"] = new_nodes
    
    if links_idx is not None and isinstance(data, list) and links_idx < len(data):
        data[links_idx]["values"] = new_links
    
    new_spec["data"] = data
    
    sorted_names = [n.get("name") for n in sorted_layer]
    method = f"order: {order}" if order else f"sort_by: {sort_by}"
    
    return {
        'success': True,
        'operation': 'reorder_nodes_in_layer',
        'vega_spec': new_spec,
        'reordered_nodes': sorted_names,
        'message': f'Reordered {len(sorted_layer)} nodes at depth {depth} ({method})'
    }


# ==================== 平行坐标图专用工具 (parallel_coordinates_tools) ====================
@mcp.tool()
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

@mcp.tool()
def reorder_dimensions(vega_spec: Dict, dimension_order: List[str]) -> Dict[str, Any]:
    """Reorder dimensions (supports both fold-based and pre-normalized long format)"""
    new_spec = copy.deepcopy(vega_spec)
    
    # Try method 1: fold transform based
    fold_transform = None
    fold_index = -1
    transforms = new_spec.get('transform', [])
    for i, transform in enumerate(transforms):
        if isinstance(transform, dict) and 'fold' in transform:
            fold_transform = transform
            fold_index = i
            break
    
    if fold_transform is not None:
        # Standard fold-based parallel coordinates
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
        
        # Update scale.domain for x encoding with field='key'
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
        # Method 2: Pre-normalized long format (no fold, use x.sort or x.scale.domain)
        def update_x_sort(obj):
            """Update x encoding sort or scale.domain for dimension field"""
            if isinstance(obj, dict):
                if 'encoding' in obj and isinstance(obj['encoding'], dict):
                    x_encoding = obj['encoding'].get('x')
                    if isinstance(x_encoding, dict) and x_encoding.get('field') in ['dimension', 'key', 'variable']:
                        # Update sort array
                        x_encoding['sort'] = dimension_order
                        # Also update scale.domain if exists
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

@mcp.tool()
def filter_by_category(vega_spec: Dict, field: str, values: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Filter by category (before fold, using wide format)
    
    Args:
        vega_spec: Vega spec
        field: category field name (e.g. "Species", "product", "region")
        values: list of values to keep
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if not isinstance(values, list):
        values = [values]
    
    if 'transform' not in new_spec:
        new_spec['transform'] = []
    
    # find fold operation position
    fold_index = -1
    for i, transform in enumerate(new_spec['transform']):
        if isinstance(transform, dict) and 'fold' in transform:
            fold_index = i
            break
    
    # build filter expression (use bracket notation for field names with spaces)
    values_str = ','.join([f'"{v}"' for v in values])
    filter_expr = f"indexof([{values_str}], datum['{field}']) >= 0"
    
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
        'operation': 'filter_by_category',
        'vega_spec': new_spec,
        'message': f'Filtered {field} to: {values}'
    }


@mcp.tool()
def highlight_category(vega_spec: Dict, field: str, values: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Highlight specified category, dim other categories
    
    Args:
        vega_spec: Vega spec
        field: category field name (e.g. "Species", "product", "region")
        values: list of values to highlight
    """
    new_spec = copy.deepcopy(vega_spec)
    
    if not isinstance(values, list):
        values = [values]
    
    # check if there is layer structure
    if 'layer' in new_spec and isinstance(new_spec['layer'], list):
        # find layer containing mark: "line"
        line_layer_index = -1
        for i, layer in enumerate(new_spec['layer']):
            if isinstance(layer, dict):
                mark = layer.get('mark')
                if (isinstance(mark, dict) and mark.get('type') == 'line') or mark == 'line':
                    line_layer_index = i
                    break
        
        if line_layer_index >= 0:
            # add or update opacity in encoding of this layer
            layer = new_spec['layer'][line_layer_index]
            if 'encoding' not in layer:
                layer['encoding'] = {}
            
            # build opacity condition (use bracket notation for field names with spaces)
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
        # if no layer, add at top level encoding (backward compatible)
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


@mcp.tool()
def hide_dimensions(
    vega_spec: Dict,
    dimensions: List[str],
    mode: str = "hide",
) -> Dict[str, Any]:
    """
    Hide/show dimensions in parallel coordinates chart.
    
    Physical interaction necessity:
    - Parallel coordinates chart has too many dimensions, need to temporarily hide uninterested dimensions
    - Can be restored by show mode
    
    Args:
        vega_spec: Vega spec
        dimensions: list of dimension names to hide or show
        mode: "hide" (hide) or "show" (show), default hide
    
    Returns:
        modified spec
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
    
    # initialize or get hidden state
    state = new_spec.get('_pc_hidden_state')
    if not isinstance(state, dict):
        state = {'hidden': [], 'all_dimensions': None}
    
    hidden_set = set(state.get('hidden', []))
    
    # find fold operation in transform
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
        # prefer x.sort or x.scale.domain
        enc = spec.get('encoding', {})
        x_enc = enc.get('x') if isinstance(enc, dict) else None
        if isinstance(x_enc, dict):
            sort_vals = x_enc.get('sort')
            if isinstance(sort_vals, list) and sort_vals:
                return list(sort_vals)
            scale_domain = (x_enc.get('scale') or {}).get('domain')
            if isinstance(scale_domain, list) and scale_domain:
                return list(scale_domain)
        # next from layer data.values
        for layer in spec.get('layer', []) if isinstance(spec.get('layer'), list) else []:
            if isinstance(layer, dict):
                layer_data = layer.get('data', {})
                layer_values = layer_data.get('values') if isinstance(layer_data, dict) else None
                if isinstance(layer_values, list) and layer_values:
                    dims = _collect_dimensions_from_values(layer_values, field)
                    if dims:
                        return dims
        # finally from main data.values
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
        # long format: no fold, use dimension field filtering
        dim_field = _find_dimension_field(new_spec)
        if not dim_field:
            dim_field = _pick_dimension_field_from_data(new_spec)
        all_dims = _find_all_dimensions(new_spec, dim_field) if dim_field else []
        if not dim_field or not all_dims:
            return {
                'success': False,
                'error': 'Cannot find dimension field or dimension list for non-fold parallel coordinates.'
            }
        # save original all dimensions (first call)
        if state.get('all_dimensions') is None:
            state['all_dimensions'] = list(all_dims)
        all_dims = state['all_dimensions']
    else:
        current_fold = list(fold_transform.get('fold', []))
        # save original all dimensions (first call)
        if state.get('all_dimensions') is None:
            state['all_dimensions'] = list(current_fold)
        all_dims = state['all_dimensions']
    
    if mode_lower == "hide":
        # hide specified dimensions
        for dim in dimensions:
            hidden_set.add(dim)
        # new visible dimensions = original dimensions - hidden set
        visible_dims = [d for d in all_dims if d not in hidden_set]
    else:
        # show specified dimensions (remove from hidden set)
        for dim in dimensions:
            hidden_set.discard(dim)
        visible_dims = [d for d in all_dims if d not in hidden_set]
    
    if not visible_dims:
        return {
            'success': False,
            'error': 'Cannot hide all dimensions. At least one dimension must remain visible.'
        }
    
    if fold_index >= 0 and fold_transform is not None:
        # update fold transform
        fold_transform['fold'] = visible_dims
        new_spec['transform'][fold_index] = fold_transform
    else:
        # no fold: update transform filter
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
    
    # save state
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

@mcp.tool()
def reset_hidden_dimensions(vega_spec: Dict) -> Dict[str, Any]:
    """
    Reset all hidden dimensions, restore to all visible state.
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
    
    # find fold transform
    transforms = new_spec.get('transform', [])
    for i, t in enumerate(transforms):
        if isinstance(t, dict) and 'fold' in t:
            t['fold'] = list(all_dims)
            new_spec['transform'][i] = t
            break
    
    # clear state
    if '_pc_hidden_state' in new_spec:
        del new_spec['_pc_hidden_state']
    
    return {
        'success': True,
        'operation': 'reset_hidden_dimensions',
        'vega_spec': new_spec,
        'message': f'Reset to show all {len(all_dims)} dimensions'
    }


# ============================================================
# run server
# ============================================================

if __name__ == "__main__":
    print(" Starting Chart Tools MCP Server...")
    mcp.run()
