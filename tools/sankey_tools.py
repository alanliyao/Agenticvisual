"""
桑基图专用工具（适配 Vega 数据驱动架构）

数据结构约定（新架构）：
- data 数组中有 name="rawLinks"  的数据源，包含连接信息: {source, target, value}
- data 数组中有 name="nodeConfig" 的数据源，包含节点配置: {name, depth, order}
- data 数组中有 name="depthLabelsData" 的数据源，包含列标签: {depth, label}
- 布局由 Vega transform pipeline 自动计算（stack + window + lookup），无需手动管理坐标
- 交互由 signals 驱动：threshold, selectedNode, nodeHover, edgeHover

前端 UI 集成说明：
    1. 页面加载或 spec 变更后，先调用 get_node_options(spec) 获取完整节点元数据。
    2. 每个工具函数的返回值中也包含 _ui_hints 字段（与 get_node_options 结构相同），
       前端可用它刷新 UI 控件（因为 collapse/expand/filter 等操作会改变可选节点）。
    3. _ui_hints 中的 adjacency 字段对 highlight_path 特别有用：
       用户选了一个节点后，前端可以用 adjacency[node].downstream 只显示可达的下一步节点。
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import copy
import json


# ═══════════════════════════════════════════════════════════
#  内部工具函数
# ═══════════════════════════════════════════════════════════

def _parse_path_arg(path: Union[str, List[str]]) -> List[str]:
    """将 path 参数规范为节点名列表。支持 JSON 数组或逗号分隔字符串。"""
    if isinstance(path, list):
        return [str(x).strip() for x in path if str(x).strip()]
    raw = (path or "").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.rstrip().endswith("]"):
        try:
            out = json.loads(raw)
            return [str(x).strip() for x in (out if isinstance(out, list) else [out]) if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in raw.split(",") if x.strip()]


def _escape_vega_str(name: str) -> str:
    """Vega 信号表达式内的字符串转义（单引号）。"""
    return str(name).replace("\\", "\\\\").replace("'", "\\'")


def _find_data_source(vega_spec: Dict, name: str) -> Tuple[Optional[List], Optional[int]]:
    """从 Vega spec 的 data 数组中找到指定 name 的数据源。"""
    data = vega_spec.get("data", [])
    if not isinstance(data, list):
        return None, None
    for i, d in enumerate(data):
        if isinstance(d, dict) and d.get("name") == name:
            return d.get("values", []), i
    return None, None


def _get_raw_links(vega_spec: Dict) -> Tuple[Optional[List[Dict]], Optional[int]]:
    return _find_data_source(vega_spec, "rawLinks")


def _get_node_config(vega_spec: Dict) -> Tuple[Optional[List[Dict]], Optional[int]]:
    return _find_data_source(vega_spec, "nodeConfig")


def _get_depth_labels(vega_spec: Dict) -> Tuple[Optional[List[Dict]], Optional[int]]:
    return _find_data_source(vega_spec, "depthLabelsData")


def _compute_node_flows(links: List[Dict]) -> Dict[str, Dict[str, float]]:
    """计算每个节点的入流和出流。"""
    flows: Dict[str, Dict[str, float]] = {}

    def _ensure(name: str):
        if name not in flows:
            flows[name] = {"inflow": 0.0, "outflow": 0.0, "total": 0.0}

    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        val = float(link.get("value", 0))
        _ensure(src)
        _ensure(tgt)
        flows[src]["outflow"] += val
        flows[tgt]["inflow"] += val

    for info in flows.values():
        info["total"] = max(info["inflow"], info["outflow"])

    return flows


def _find_signal(vega_spec: Dict, signal_name: str) -> Tuple[Optional[Dict], Optional[int]]:
    signals = vega_spec.get("signals", [])
    for i, sig in enumerate(signals):
        if isinstance(sig, dict) and sig.get("name") == signal_name:
            return sig, i
    return None, None


def _find_mark(vega_spec: Dict, mark_name: str) -> Optional[Dict]:
    for mark in vega_spec.get("marks", []):
        if mark.get("name") == mark_name:
            return mark
        if mark.get("type") == "group":
            for inner in mark.get("marks", []):
                if inner.get("name") == mark_name:
                    return inner
    return None


def _update_x_scale_domain(vega_spec: Dict, max_depth: int):
    for scale in vega_spec.get("scales", []):
        if scale.get("name") == "x":
            scale["domain"] = list(range(max_depth + 1))
            return


def _make_error(msg: str) -> Dict[str, Any]:
    return {"success": False, "error": msg}


def _make_success(operation: str, message: str, vega_spec: Dict = None, **extra) -> Dict[str, Any]:
    result = {"success": True, "operation": operation, "message": message}
    if vega_spec is not None:
        result["vega_spec"] = vega_spec
    result.update(extra)
    return result


def _build_ui_hints(vega_spec: Dict) -> Dict[str, Any]:
    """
    从 vega_spec 中提取前端 UI 控件需要的全部元数据。

    返回结构示例:
    {
        "all_nodes": ["Google Ads", "Facebook", ...],

        "nodes_by_depth": {
            "0": {
                "label": "Traffic Source",
                "nodes": [
                    {"name": "Google Ads", "order": 0, "total": 9000.0},
                    {"name": "Facebook",   "order": 1, "total": 7000.0},
                    ...
                ]
            },
            "1": { ... },
            ...
        },

        "depth_count": 5,
        "depth_labels": {"0": "Traffic Source", "1": "Entry Point", ...},

        "edges": [
            {"source": "Google Ads", "target": "Landing Page", "value": 5000},
            ...
        ],

        "adjacency": {
            "Google Ads": {
                "upstream": [],
                "downstream": ["Landing Page", "Product List"]
            },
            "Landing Page": {
                "upstream": ["Google Ads", "Facebook", "Organic"],
                "downstream": ["Product Detail", "Exit"]
            },
            ...
        },

        "collapsed_groups": {"Others (Layer 0)": ["Email", "Organic"], ...},

        "value_range": {"min": 1000, "max": 10000}
    }

    前端使用指南:
    ┌──────────────────────┬──────────────────────────────────────────────────────────┐
    │ 工具                  │ 用哪些字段填充 UI                                         │
    ├──────────────────────┼──────────────────────────────────────────────────────────┤
    │ highlight_path       │ nodes_by_depth 逐层选节点；adjacency.downstream 限制下一步  │
    │ trace_node           │ all_nodes 填充单选下拉框                                   │
    │ collapse_nodes       │ nodes_by_depth 同层多选框                                  │
    │ expand_node          │ collapsed_groups 的 keys 填充下拉框                         │
    │ calculate_conversion │ all_nodes 填充可选单选下拉框（可留空=全部）                    │
    │ color_flows          │ all_nodes 填充多选框                                       │
    │ reorder_nodes        │ nodes_by_depth 选层后展示该层节点                            │
    │ filter_flow          │ value_range 设置滑块 min/max                               │
    │ find_bottleneck      │ 无需选择（直接执行）                                        │
    │ auto_collapse        │ 无需选择（输入 top_n 数字即可）                              │
    └──────────────────────┴──────────────────────────────────────────────────────────┘
    """
    hints: Dict[str, Any] = {
        "all_nodes": [],
        "nodes_by_depth": {},
        "depth_count": 0,
        "depth_labels": {},
        "edges": [],
        "adjacency": {},
        "collapsed_groups": {},
        "value_range": {"min": 0, "max": 0}
    }

    nodes, _ = _get_node_config(vega_spec)
    links, _ = _get_raw_links(vega_spec)
    depth_labels_data, _ = _get_depth_labels(vega_spec)

    if not nodes or not links:
        return hints

    node_flows = _compute_node_flows(links)

    # depth labels
    label_map: Dict[int, str] = {}
    if depth_labels_data:
        for dl in depth_labels_data:
            label_map[dl.get("depth", -1)] = dl.get("label", "")

    # nodes_by_depth + all_nodes
    depth_groups: Dict[int, List[Dict]] = {}
    all_names: List[str] = []
    for node in sorted(nodes, key=lambda n: (n.get("depth", 0), n.get("order", 0))):
        name = node.get("name", "")
        depth = node.get("depth", 0)
        all_names.append(name)
        flow = node_flows.get(name, {})
        entry: Dict[str, Any] = {
            "name": name,
            "order": node.get("order", 0),
            "total": round(flow.get("total", 0), 2)
        }
        if node.get("_is_aggregate"):
            entry["is_aggregate"] = True
            entry["collapsed_nodes"] = node.get("_collapsed_nodes", [])
        depth_groups.setdefault(depth, []).append(entry)

    hints["all_nodes"] = all_names
    hints["depth_count"] = len(depth_groups)
    # 用字符串 key 方便 JSON 序列化
    hints["nodes_by_depth"] = {
        str(depth): {
            "label": label_map.get(depth, f"Layer {depth}"),
            "nodes": node_list
        }
        for depth, node_list in sorted(depth_groups.items())
    }
    hints["depth_labels"] = {str(k): v for k, v in label_map.items()}

    # edges + adjacency + value_range
    edge_list = []
    adjacency: Dict[str, Dict[str, List[str]]] = {
        name: {"upstream": [], "downstream": []} for name in all_names
    }
    values = []

    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        val = float(link.get("value", 0))
        edge_list.append({"source": src, "target": tgt, "value": val})
        values.append(val)
        if src in adjacency and tgt not in adjacency[src]["downstream"]:
            adjacency[src]["downstream"].append(tgt)
        if tgt in adjacency and src not in adjacency[tgt]["upstream"]:
            adjacency[tgt]["upstream"].append(src)

    hints["edges"] = edge_list
    hints["adjacency"] = adjacency
    if values:
        hints["value_range"] = {"min": round(min(values), 2), "max": round(max(values), 2)}

    # collapsed groups
    state = vega_spec.get("_sankey_state", {})
    hints["collapsed_groups"] = state.get("collapsed_groups", {})

    return hints


# ═══════════════════════════════════════════════════════════
#  元数据提取工具（前端首先调用这个来填充 UI）
# ═══════════════════════════════════════════════════════════

def get_node_options(vega_spec: Dict) -> Dict[str, Any]:
    """
    从 vega_spec 中提取完整的节点元数据，供前端填充 UI 控件。

    前端应在以下时机调用：
    - 页面首次加载 sankey spec 时
    - 任何工具执行后 spec 发生变化时（也可以直接用返回值中的 _ui_hints）

    Args:
        vega_spec: Vega 规范

    Returns:
        包含 all_nodes, nodes_by_depth, adjacency, edges, collapsed_groups, value_range 等字段的字典。
    """
    nodes, _ = _get_node_config(vega_spec)
    if not nodes:
        return _make_error("Cannot find nodeConfig data source")

    hints = _build_ui_hints(vega_spec)

    return {
        "success": True,
        "operation": "get_node_options",
        "message": f"Extracted {len(hints['all_nodes'])} nodes across {hints['depth_count']} layers",
        **hints
    }


# ═══════════════════════════════════════════════════════════
#  数据操作类工具
# ═══════════════════════════════════════════════════════════

def filter_flow(vega_spec: Dict, min_value: float) -> Dict[str, Any]:
    """
    过滤流量：只显示 value >= min_value 的连接。

    Args:
        vega_spec: Vega 规范
        min_value:  最小流量阈值
    """
    links, links_idx = _get_raw_links(vega_spec)
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    if not any(link.get("value", 0) >= min_value for link in links):
        return _make_error(f"No links with value >= {min_value}")

    new_spec = copy.deepcopy(vega_spec)

    sig, sig_idx = _find_signal(new_spec, "threshold")
    if sig is not None:
        sig["value"] = min_value
        bind = sig.get("bind", {})
        if isinstance(bind, dict) and bind.get("input") == "range":
            if min_value > bind.get("max", 0):
                bind["max"] = min_value * 1.5
        filtered_count = sum(1 for l in links if l.get("value", 0) >= min_value)
        result = _make_success(
            "filter_flow",
            f"Set threshold signal to {min_value}. {filtered_count}/{len(links)} links visible.",
            vega_spec=new_spec
        )
        result["_ui_hints"] = _build_ui_hints(new_spec)
        return result

    filtered_links = [l for l in links if l.get("value", 0) >= min_value]
    used_nodes = set()
    for link in filtered_links:
        used_nodes.add(link["source"])
        used_nodes.add(link["target"])
    new_spec["data"][links_idx]["values"] = filtered_links
    nodes, nodes_idx = _get_node_config(new_spec)
    if nodes is not None and nodes_idx is not None:
        new_spec["data"][nodes_idx]["values"] = [
            n for n in nodes if n.get("name") in used_nodes
        ]
    result = _make_success(
        "filter_flow",
        f"Filtered to {len(filtered_links)} links with value >= {min_value}",
        vega_spec=new_spec
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def collapse_nodes(
    vega_spec: Dict,
    nodes_to_collapse: List[str],
    aggregate_name: str = "Other"
) -> Dict[str, Any]:
    """
    折叠多个节点：将指定节点合并为一个聚合节点。

    Args:
        vega_spec:          Vega 规范
        nodes_to_collapse:  要折叠的节点名称列表
        aggregate_name:     聚合后的节点名称（默认 "Other"）
    """
    links, links_idx = _get_raw_links(vega_spec)
    nodes, nodes_idx = _get_node_config(vega_spec)
    if links is None or nodes is None:
        return _make_error("Cannot find rawLinks or nodeConfig data source")

    collapse_set = set(nodes_to_collapse)
    existing_names = {n.get("name") for n in nodes}
    missing = collapse_set - existing_names
    if missing:
        return _make_error(f"Nodes not found: {sorted(missing)}")

    new_spec = copy.deepcopy(vega_spec)

    if "_sankey_state" not in new_spec:
        new_spec["_sankey_state"] = {
            "original_nodes": copy.deepcopy(nodes),
            "original_links": copy.deepcopy(links),
            "collapsed_groups": {}
        }
    state = new_spec["_sankey_state"]
    state.setdefault("collapsed_groups", {})
    state["collapsed_groups"][aggregate_name] = list(nodes_to_collapse)

    collapse_depth = 0
    max_order = 0
    for n in nodes:
        if n.get("name") in collapse_set:
            collapse_depth = n.get("depth", 0)
        if n.get("depth") == collapse_depth:
            max_order = max(max_order, n.get("order", 0))

    new_nodes = [n for n in nodes if n.get("name") not in collapse_set]
    new_nodes.append({
        "name": aggregate_name,
        "depth": collapse_depth,
        "order": max_order + 1,
        "_is_aggregate": True,
        "_collapsed_nodes": list(nodes_to_collapse)
    })

    link_agg: Dict[Tuple[str, str], float] = {}
    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        val = float(link.get("value", 0))
        new_src = aggregate_name if src in collapse_set else src
        new_tgt = aggregate_name if tgt in collapse_set else tgt
        if new_src == aggregate_name and new_tgt == aggregate_name:
            continue
        key = (new_src, new_tgt)
        link_agg[key] = link_agg.get(key, 0) + val

    new_links = [{"source": s, "target": t, "value": v} for (s, t), v in link_agg.items()]

    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = new_links

    result = _make_success(
        "collapse_nodes",
        f'Collapsed {len(nodes_to_collapse)} nodes into "{aggregate_name}"',
        vega_spec=new_spec
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def expand_node(vega_spec: Dict, aggregate_name: str) -> Dict[str, Any]:
    """
    展开聚合节点：恢复被折叠的原始节点。

    Args:
        vega_spec:       Vega 规范
        aggregate_name:  要展开的聚合节点名称
    """
    state = vega_spec.get("_sankey_state", {})
    collapsed_groups = state.get("collapsed_groups", {})

    if not state:
        return _make_error("No _sankey_state found. The chart has no collapsed nodes.")
    if aggregate_name not in collapsed_groups:
        available = list(collapsed_groups.keys())
        return _make_error(f'"{aggregate_name}" is not a collapsed group. Available: {available}')

    original_nodes = state.get("original_nodes")
    original_links = state.get("original_links")
    if not original_nodes or not original_links:
        return _make_error("Original data lost, cannot expand")

    links, links_idx = _get_raw_links(vega_spec)
    nodes, nodes_idx = _get_node_config(vega_spec)
    if links is None or nodes is None:
        return _make_error("Cannot find rawLinks or nodeConfig data source")

    new_spec = copy.deepcopy(vega_spec)
    collapsed_node_names = set(collapsed_groups[aggregate_name])

    new_nodes = [n for n in nodes if n.get("name") != aggregate_name]
    for orig_node in original_nodes:
        if orig_node.get("name") in collapsed_node_names:
            new_nodes.append(copy.deepcopy(orig_node))

    current_node_names = {n.get("name") for n in new_nodes}
    restored_links = []
    for orig_link in original_links:
        src = orig_link.get("source")
        tgt = orig_link.get("target")
        if src in current_node_names and tgt in current_node_names:
            restored_links.append(copy.deepcopy(orig_link))

    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        if src == aggregate_name or tgt == aggregate_name:
            continue
        if src in collapsed_node_names or tgt in collapsed_node_names:
            continue
        exists = any(
            l.get("source") == src and l.get("target") == tgt
            for l in restored_links
        )
        if not exists:
            restored_links.append(copy.deepcopy(link))

    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = restored_links
    del new_spec["_sankey_state"]["collapsed_groups"][aggregate_name]

    result = _make_success(
        "expand_node",
        f'Expanded "{aggregate_name}" back to {len(collapsed_node_names)} nodes',
        vega_spec=new_spec
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def auto_collapse_by_rank(vega_spec: Dict, top_n: int = 5) -> Dict[str, Any]:
    """
    按流量排名自动折叠：每层只保留 top N 个节点，其余折叠到 "Others (Layer X)"。

    Args:
        vega_spec: Vega 规范
        top_n:     每层保留的 top 节点数量（默认 5）
    """
    links, links_idx = _get_raw_links(vega_spec)
    nodes, nodes_idx = _get_node_config(vega_spec)
    if links is None or nodes is None:
        return _make_error("Cannot find rawLinks or nodeConfig data source")

    new_spec = copy.deepcopy(vega_spec)

    if "_sankey_state" not in new_spec:
        new_spec["_sankey_state"] = {
            "original_nodes": copy.deepcopy(nodes),
            "original_links": copy.deepcopy(links),
            "collapsed_groups": {}
        }
    state = new_spec["_sankey_state"]
    state.setdefault("collapsed_groups", {})

    node_flows = _compute_node_flows(links)

    depth_groups: Dict[int, List[Dict]] = {}
    for node in nodes:
        depth = node.get("depth", 0)
        depth_groups.setdefault(depth, []).append(node)

    nodes_to_keep: set = set()
    collapsed_by_layer: Dict[int, Dict] = {}
    node_to_aggregate: Dict[str, str] = {}

    for depth, group in depth_groups.items():
        sorted_group = sorted(
            group,
            key=lambda n: node_flows.get(n.get("name"), {}).get("total", 0),
            reverse=True
        )
        for node in sorted_group[:top_n]:
            nodes_to_keep.add(node.get("name"))

        collapsed_names = [n.get("name") for n in sorted_group[top_n:]]
        if collapsed_names:
            agg_name = f"Others (Layer {depth})"
            collapsed_by_layer[depth] = {
                "aggregate_name": agg_name,
                "collapsed_nodes": collapsed_names
            }
            state["collapsed_groups"][agg_name] = collapsed_names
            for name in collapsed_names:
                node_to_aggregate[name] = agg_name

    if not collapsed_by_layer:
        result = _make_success(
            "auto_collapse_by_rank",
            f"All layers have <= {top_n} nodes, nothing to collapse",
            vega_spec=new_spec
        )
        result["_ui_hints"] = _build_ui_hints(new_spec)
        return result

    new_nodes = [n for n in nodes if n.get("name") in nodes_to_keep]
    for depth, info in collapsed_by_layer.items():
        max_order = max(
            (n.get("order", 0) for n in depth_groups.get(depth, [])),
            default=0
        )
        new_nodes.append({
            "name": info["aggregate_name"],
            "depth": depth,
            "order": max_order + 1,
            "_is_aggregate": True,
            "_collapsed_nodes": info["collapsed_nodes"]
        })

    link_agg: Dict[Tuple[str, str], float] = {}
    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        val = float(link.get("value", 0))
        new_src = node_to_aggregate.get(src, src)
        new_tgt = node_to_aggregate.get(tgt, tgt)
        key = (new_src, new_tgt)
        link_agg[key] = link_agg.get(key, 0) + val

    new_links = [{"source": s, "target": t, "value": v} for (s, t), v in link_agg.items()]

    new_spec["data"][nodes_idx]["values"] = new_nodes
    new_spec["data"][links_idx]["values"] = new_links

    total_collapsed = sum(len(info["collapsed_nodes"]) for info in collapsed_by_layer.values())
    result = _make_success(
        "auto_collapse_by_rank",
        f"Kept top {top_n} per layer, collapsed {total_collapsed} nodes into {len(collapsed_by_layer)} groups",
        vega_spec=new_spec,
        collapsed_groups={
            info["aggregate_name"]: info["collapsed_nodes"]
            for info in collapsed_by_layer.values()
        }
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def reorder_nodes_in_layer(
    vega_spec: Dict,
    depth: int,
    order: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    重排桑基图某一层内节点的上下顺序。

    Args:
        vega_spec: Vega 规范
        depth:     要重排的层（0, 1, 2, ...）
        order:     节点名称列表，从上到下。与 sort_by 互斥。
        sort_by:   排序方式："value_desc", "value_asc", "name"
    """
    if order is None and sort_by is None:
        return _make_error('Must specify either "order" or "sort_by"')
    if order is not None and sort_by is not None:
        return _make_error('Cannot specify both "order" and "sort_by"')

    links, _ = _get_raw_links(vega_spec)
    nodes, nodes_idx = _get_node_config(vega_spec)
    if nodes is None:
        return _make_error("Cannot find nodeConfig data source")
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    layer_nodes = [n for n in nodes if n.get("depth") == depth]
    if not layer_nodes:
        return _make_error(f"No nodes found at depth {depth}")

    if order is not None:
        order_map = {name: i for i, name in enumerate(order)}
        sorted_names = sorted(
            [n.get("name") for n in layer_nodes],
            key=lambda name: order_map.get(name, 999999)
        )
    else:
        sort_key = str(sort_by).lower().strip()
        node_flows = _compute_node_flows(links)
        if sort_key == "value_desc":
            sorted_names = sorted(
                [n.get("name") for n in layer_nodes],
                key=lambda name: node_flows.get(name, {}).get("total", 0),
                reverse=True
            )
        elif sort_key == "value_asc":
            sorted_names = sorted(
                [n.get("name") for n in layer_nodes],
                key=lambda name: node_flows.get(name, {}).get("total", 0)
            )
        elif sort_key == "name":
            sorted_names = sorted(n.get("name") for n in layer_nodes)
        else:
            return _make_error(f'Invalid sort_by: {sort_by}. Use "value_desc", "value_asc", or "name"')

    name_to_new_order = {name: i for i, name in enumerate(sorted_names)}

    new_spec = copy.deepcopy(vega_spec)
    for node in new_spec["data"][nodes_idx]["values"]:
        if node.get("depth") == depth and node.get("name") in name_to_new_order:
            node["order"] = name_to_new_order[node["name"]]

    method = f"order: {order}" if order else f"sort_by: {sort_by}"
    result = _make_success(
        "reorder_nodes_in_layer",
        f"Reordered {len(sorted_names)} nodes at depth {depth} ({method})",
        vega_spec=new_spec,
        reordered_nodes=sorted_names
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


# ═══════════════════════════════════════════════════════════
#  视觉交互类工具
# ═══════════════════════════════════════════════════════════

def highlight_path(vega_spec: Dict, path: Union[str, List[str]]) -> Dict[str, Any]:
    """
    高亮多步路径：强调指定边，弱化其他边和节点。

    Args:
        vega_spec: Vega 规范
        path:      节点路径列表。支持 ["A","B","C"]、'["A","B","C"]'、'A,B,C'
    """
    path = _parse_path_arg(path)
    if not path or len(path) < 2:
        return _make_error("Path must contain at least 2 nodes")

    links, _ = _get_raw_links(vega_spec)
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    link_set = {(l.get("source"), l.get("target")) for l in links}
    highlight_edges = []
    missing_edges = []
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge in link_set:
            highlight_edges.append(edge)
        else:
            missing_edges.append(f"{edge[0]} → {edge[1]}")

    if not highlight_edges:
        return _make_error(f"No valid edges in path. Missing: {missing_edges}")

    new_spec = copy.deepcopy(vega_spec)

    path_nodes = set(path)
    edge_conditions = [
        f"(datum.source === '{_escape_vega_str(s)}' && datum.target === '{_escape_vega_str(t)}')"
        for s, t in highlight_edges
    ]
    is_on_path = " || ".join(edge_conditions)

    node_conditions = [f"datum.name === '{_escape_vega_str(n)}'" for n in path_nodes]
    is_path_node = " || ".join(node_conditions)

    edge_mark = _find_mark(new_spec, "edgeMark")
    if edge_mark:
        update = edge_mark.setdefault("encode", {}).setdefault("update", {})
        update["fillOpacity"] = {"signal": f"({is_on_path}) ? 0.75 : 0.06"}
        update["strokeOpacity"] = {"signal": f"({is_on_path}) ? 0.5 : 0.02"}

    node_mark = _find_mark(new_spec, "nodeRect")
    if node_mark:
        update = node_mark.setdefault("encode", {}).setdefault("update", {})
        update["fillOpacity"] = {"signal": f"({is_path_node}) ? 1.0 : 0.15"}
        update["strokeWidth"] = {"signal": f"({is_path_node}) ? 2.5 : 0.5"}

    path_desc = " → ".join(path)
    warning = f" (Note: edges not found: {missing_edges})" if missing_edges else ""
    result = _make_success(
        "highlight_path",
        f"Highlighted path: {path_desc}{warning}",
        vega_spec=new_spec,
        highlighted_edges=len(highlight_edges),
        total_edges_in_path=len(path) - 1
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def trace_node(vega_spec: Dict, node_name: str) -> Dict[str, Any]:
    """
    追踪节点：高亮与该节点直接相连的所有连接。

    Args:
        vega_spec: Vega 规范
        node_name: 节点名称
    """
    links, _ = _get_raw_links(vega_spec)
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    node_exists = any(
        l.get("source") == node_name or l.get("target") == node_name
        for l in links
    )
    if not node_exists:
        return _make_error(f'Node "{node_name}" not found in links')

    new_spec = copy.deepcopy(vega_spec)

    sig, sig_idx = _find_signal(new_spec, "selectedNode")
    if sig is not None:
        sig["value"] = node_name
        result = _make_success(
            "trace_node",
            f"Set selectedNode signal to '{node_name}'. Connected flows highlighted.",
            vega_spec=new_spec
        )
        result["_ui_hints"] = _build_ui_hints(new_spec)
        return result

    en = _escape_vega_str(node_name)
    edge_expr = f"datum.source === '{en}' || datum.target === '{en}' ? 0.75 : 0.08"
    node_expr = f"datum.name === '{en}' ? 1.0 : 0.2"

    edge_mark = _find_mark(new_spec, "edgeMark")
    if edge_mark:
        update = edge_mark.setdefault("encode", {}).setdefault("update", {})
        update["fillOpacity"] = {"signal": edge_expr}

    node_mark = _find_mark(new_spec, "nodeRect")
    if node_mark:
        update = node_mark.setdefault("encode", {}).setdefault("update", {})
        update["fillOpacity"] = {"signal": node_expr}

    result = _make_success(
        "trace_node",
        f"Traced all connections of node: {node_name}",
        vega_spec=new_spec
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


def color_flows(vega_spec: Dict, nodes: List[str], color: str = "#e74c3c") -> Dict[str, Any]:
    """
    给与指定节点相连的流着色。

    Args:
        vega_spec: Vega 规范
        nodes:     节点名称列表
        color:     着色颜色（默认红色 #e74c3c）
    """
    links_data, _ = _get_raw_links(vega_spec)
    if links_data is None:
        return _make_error("Cannot find rawLinks data source")

    nodes_set = set(nodes or [])
    if not nodes_set:
        return _make_error("nodes list is empty")

    colored_edges = [
        (l.get("source"), l.get("target"))
        for l in links_data
        if l.get("source") in nodes_set or l.get("target") in nodes_set
    ]
    if not colored_edges:
        return _make_error(f"No flows connected to nodes: {sorted(nodes_set)}")

    new_spec = copy.deepcopy(vega_spec)

    parts = [
        f"(datum.source === '{_escape_vega_str(s)}' && datum.target === '{_escape_vega_str(t)}')"
        for s, t in colored_edges
    ]
    is_colored = " || ".join(parts)

    edge_mark = _find_mark(new_spec, "edgeMark")
    if edge_mark is None:
        return _make_error("Cannot find edgeMark in Vega spec")

    update_enc = edge_mark.get("encode", {}).get("update", {})
    original_fill = update_enc.get("fill", {})

    if isinstance(original_fill, dict) and "scale" in original_fill and "field" in original_fill:
        scale_name = original_fill["scale"]
        field_name = original_fill["field"]
        fallback = f"scale('{scale_name}', datum.{field_name})"
    elif isinstance(original_fill, dict) and "signal" in original_fill:
        fallback = f"({original_fill['signal']})"
    elif isinstance(original_fill, dict) and "value" in original_fill:
        fallback = f"'{original_fill['value']}'"
    else:
        fallback = "scale('color', datum.source)"

    fill_signal = f"({is_colored}) ? '{color}' : {fallback}"

    update = edge_mark.setdefault("encode", {}).setdefault("update", {})
    update["fill"] = {"signal": fill_signal}

    result = _make_success(
        "color_flows",
        f"Colored {len(colored_edges)} flows connected to nodes: {sorted(nodes_set)}",
        vega_spec=new_spec,
        colored_count=len(colored_edges)
    )
    result["_ui_hints"] = _build_ui_hints(new_spec)
    return result


# ═══════════════════════════════════════════════════════════
#  纯分析类工具
# ═══════════════════════════════════════════════════════════

def calculate_conversion_rate(
    vega_spec: Dict,
    node_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    计算转化率：分析每个节点的入流、出流和转化率。

    Args:
        vega_spec: Vega 规范
        node_name: 指定节点名称（可选）。不指定则返回所有节点的转化率。
    """
    links, _ = _get_raw_links(vega_spec)
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    node_flows = _compute_node_flows(links)
    ui_hints = _build_ui_hints(vega_spec)

    conversions = []
    for name in sorted(node_flows.keys()):
        info = node_flows[name]
        inflow = info["inflow"]
        outflow = info["outflow"]

        if inflow == 0 and outflow > 0:
            node_type = "source"
            rate = "source"
        elif outflow == 0 and inflow > 0:
            node_type = "sink"
            rate = 0.0
        else:
            node_type = "intermediate"
            rate = round(outflow / inflow, 4) if inflow > 0 else 0.0

        conversion: Dict[str, Any] = {
            "node": name,
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
            "rate": rate,
            "type": node_type
        }

        if node_type == "intermediate" and inflow > 0:
            loss = inflow - outflow
            conversion["loss"] = round(loss, 2)
            conversion["loss_rate"] = round(loss / inflow, 4)

        conversions.append(conversion)

    if node_name:
        target = [c for c in conversions if c["node"] == node_name]
        if not target:
            return _make_error(f'Node "{node_name}" not found')

        upstream = [
            {"from": l["source"], "value": l["value"]}
            for l in links if l.get("target") == node_name
        ]
        downstream = [
            {"to": l["target"], "value": l["value"]}
            for l in links if l.get("source") == node_name
        ]

        result = _make_success(
            "calculate_conversion_rate",
            f"Conversion analysis for {node_name}",
            node=node_name,
            conversion=target[0],
            upstream=upstream,
            downstream=downstream
        )
        result["_ui_hints"] = ui_hints
        return result

    sources = [c for c in conversions if c["type"] == "source"]
    sinks = [c for c in conversions if c["type"] == "sink"]
    intermediates = [c for c in conversions if c["type"] == "intermediate"]

    high_loss = sorted(
        [c for c in intermediates if c.get("loss_rate", 0) > 0],
        key=lambda x: x["loss_rate"],
        reverse=True
    )[:5]

    result = _make_success(
        "calculate_conversion_rate",
        f"Calculated conversion rates for {len(node_flows)} nodes",
        summary={
            "total_nodes": len(node_flows),
            "source_nodes": len(sources),
            "sink_nodes": len(sinks),
            "intermediate_nodes": len(intermediates)
        },
        conversions=conversions,
        high_loss_nodes=high_loss
    )
    result["_ui_hints"] = ui_hints
    return result


def find_bottleneck(vega_spec: Dict, top_n: int = 3) -> Dict[str, Any]:
    """
    识别流失最严重的节点。

    Args:
        vega_spec: Vega 规范
        top_n:     返回流失最严重的前 N 个节点
    """
    links, _ = _get_raw_links(vega_spec)
    if links is None:
        return _make_error("Cannot find rawLinks data source")

    node_flows = _compute_node_flows(links)

    bottlenecks = []
    for name, info in node_flows.items():
        inflow = info["inflow"]
        outflow = info["outflow"]
        if inflow > 0 and outflow > 0 and inflow > outflow:
            loss = inflow - outflow
            bottlenecks.append({
                "node": name,
                "inflow": round(inflow, 2),
                "outflow": round(outflow, 2),
                "loss": round(loss, 2),
                "loss_rate": round(loss / inflow, 4)
            })

    bottlenecks.sort(key=lambda x: x["loss_rate"], reverse=True)
    top = bottlenecks[:top_n]

    if not top:
        result = _make_success(
            "find_bottleneck",
            "No bottlenecks found (no intermediate nodes with loss)",
            bottlenecks=[],
            total_bottleneck_nodes=0
        )
        result["_ui_hints"] = _build_ui_hints(vega_spec)
        return result

    result = _make_success(
        "find_bottleneck",
        f"Found top {len(top)} bottleneck nodes with highest loss rates",
        bottlenecks=top,
        total_bottleneck_nodes=len(bottlenecks)
    )
    result["_ui_hints"] = _build_ui_hints(vega_spec)
    return result


# ═══════════════════════════════════════════════════════════
#  模块导出
# ═══════════════════════════════════════════════════════════

__all__ = [
    "get_node_options",
    "filter_flow",
    "highlight_path",
    "calculate_conversion_rate",
    "trace_node",
    "collapse_nodes",
    "expand_node",
    "auto_collapse_by_rank",
    "color_flows",
    "find_bottleneck",
    "reorder_nodes_in_layer",
]