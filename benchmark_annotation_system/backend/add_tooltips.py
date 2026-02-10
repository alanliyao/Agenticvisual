#!/usr/bin/env python3
"""
为 specs 目录下所有没有 tooltip 的 Vega-Lite spec 添加 tooltip
"""
import json
import copy
from pathlib import Path

SPECS_DIR = Path(__file__).parent / "specs"


def is_vega_full(spec: dict) -> bool:
    """判断是否为完整 Vega 规格（非 Vega-Lite，如 Sankey）"""
    schema = spec.get("$schema", "")
    if "vega/v" in schema and "vega-lite" not in schema:
        return True
    if isinstance(spec.get("data"), list):
        names = [d.get("name") for d in spec["data"] if isinstance(d, dict)]
        if "nodes" in names or "links" in names:
            return True
    return False


def has_tooltip(spec: dict) -> bool:
    """检查 spec 是否已有 tooltip"""
    enc = spec.get("encoding", {})
    if enc.get("tooltip"):
        return True
    for layer in spec.get("layer", []):
        if isinstance(layer, dict) and layer.get("encoding", {}).get("tooltip"):
            return True
    return False


def get_encoding_fields(enc: dict) -> list:
    """从 encoding 提取用于 tooltip 的字段列表 [(field, type), ...]"""
    if not enc:
        return []
    result = []
    for ch in ("x", "y", "color", "size", "opacity", "strokeDash", "detail"):
        e = enc.get(ch)
        if isinstance(e, dict) and e.get("field"):
            f = e["field"]
            t = e.get("type", "nominal")
            if (f, t) not in [(r[0], r[1]) for r in result]:
                result.append((f, t))
    # 堆叠条形图可能有 xOffset
    xoff = enc.get("xOffset")
    if isinstance(xoff, dict) and xoff.get("field"):
        f = xoff["field"]
        if (f, "nominal") not in [(r[0], r[1]) for r in result]:
            result.append((f, "nominal"))
    return result


def build_tooltip_array(fields: list) -> list:
    return [{"field": f, "type": t} for f, t in fields]


def add_tooltip_to_encoding(enc: dict, fields: list) -> None:
    if not enc or not fields:
        return
    tooltip = build_tooltip_array(fields)
    if tooltip:
        enc["tooltip"] = tooltip


def add_tooltip_to_spec(spec: dict, sample_row: dict = None) -> bool:
    """
    为 Vega-Lite spec 添加 tooltip，如有必要。
    返回是否进行了修改。
    """
    if is_vega_full(spec):
        return False  # 跳过 Vega 格式（Sankey 等）

    modified = False

    # 1. 顶层 encoding
    enc = spec.get("encoding", {})
    if enc and not enc.get("tooltip"):
        fields = get_encoding_fields(enc)
        if fields:
            add_tooltip_to_encoding(enc, fields)
            modified = True

    # 2. layer 结构
    layers = spec.get("layer", [])
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        le = layer.get("encoding", {})
        if le and not le.get("tooltip"):
            # 只对包含 x 或 y 的可视化层添加（排除 rule/tick/text 等轴层）
            has_xy = "x" in le or "y" in le
            mark = layer.get("mark")
            mark_type = mark.get("type", mark) if isinstance(mark, dict) else mark
            if has_xy and mark_type in ("line", "point", "circle", "bar", "rect", "area"):
                fields = get_encoding_fields(le)
                # 平行坐标长格式：添加 dimension, normalized_value 及 color/detail 字段
                if "dimension" in [f[0] for f in fields] or (sample_row and "dimension" in sample_row):
                    dims = ["dimension", "normalized_value"]
                    for k, v in (sample_row or {}).items():
                        if k not in ("dimension", "normalized_value", "_index") and not k.startswith("_"):
                            if isinstance(v, str) or (v is not None and not isinstance(v, (int, float))):
                                dims.append(k)
                            elif isinstance(v, (int, float)):
                                dims.append(k)
                    # 去重并保留顺序
                    seen = set()
                    extra = []
                    for f in dims:
                        if f not in seen:
                            seen.add(f)
                            t = "nominal" if sample_row and isinstance(sample_row.get(f), str) else "quantitative"
                            extra.append((f, t))
                    fields = extra if extra else fields
                if fields:
                    add_tooltip_to_encoding(le, fields)
                    modified = True
        # 递归处理嵌套 layer
        if "layer" in layer:
            if add_tooltip_to_spec(layer, sample_row):
                modified = True

    return modified


def get_sample_row(spec: dict) -> dict:
    """从 spec 获取一行样例数据"""
    data = spec.get("data")
    if isinstance(data, dict) and data.get("values"):
        vals = data["values"]
        if vals and isinstance(vals[0], dict):
            return vals[0]
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("values"):
                vals = d["values"]
                if vals and isinstance(vals[0], dict):
                    return vals[0]
    return {}


def process_file(fp: Path) -> bool:
    """处理单个文件，返回是否修改"""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [SKIP] {fp.name}: {e}")
        return False

    if "spec" in data:
        spec = data["spec"]
        root = data
        key = "spec"
    else:
        spec = data
        root = data
        key = None

    if has_tooltip(spec):
        return False

    sample = get_sample_row(spec)
    if add_tooltip_to_spec(spec, sample):
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(root, f, ensure_ascii=False, indent=2)
        return True
    return False


def main():
    count = 0
    for fp in sorted(SPECS_DIR.glob("*.json")):
        if fp.name.startswith("."):
            continue
        if process_file(fp):
            count += 1
            print(f"  + {fp.name}")
    print(f"\nDone: added tooltip to {count} files")


if __name__ == "__main__":
    main()
