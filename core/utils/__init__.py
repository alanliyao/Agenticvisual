"""工具函数模块"""
from .logger import setup_logger, app_logger, error_logger
from .image_utils import encode_image_to_base64, decode_base64_to_image, create_data_url
from .json_utils import safe_json_loads, safe_json_dumps, extract_json_from_text
from typing import Dict, List, Any


def get_spec_data_values(spec: Dict) -> List[Dict[str, Any]]:
    """
    获取 Vega 或 Vega-Lite 规范中的数据值，兼容两种格式。
    
    - Vega-Lite: data 是对象 {"values": [...]}
    - Vega: data 是数组 [{"name": "xxx", "values": [...]}, ...]
    
    Returns:
        数据值列表（对于 Vega 格式返回第一个数据源的值）
    """
    data = spec.get("data", {})
    if isinstance(data, list):
        # Vega 格式：返回第一个有 values 的数据源
        for d in data:
            if isinstance(d, dict) and d.get("values"):
                return d.get("values", [])
        return []
    else:
        # Vega-Lite 格式
        return data.get("values", []) if isinstance(data, dict) else []


def get_spec_data_count(spec: Dict) -> int:
    """获取 Vega/Vega-Lite 规范中数据点的总数"""
    data = spec.get("data", {})
    if isinstance(data, list):
        # Vega 格式：统计所有数据源的值
        return sum(len(d.get("values", [])) for d in data if isinstance(d, dict))
    else:
        # Vega-Lite 格式
        return len(data.get("values", [])) if isinstance(data, dict) else 0


def is_vega_full_spec(spec: Dict) -> bool:
    """检测是否为完整 Vega 规范（而非 Vega-Lite）"""
    schema = spec.get("$schema", "")
    if "vega-lite" in schema.lower():
        return False
    if "/vega/" in schema.lower() or "vega/v" in schema.lower():
        return True
    # 如果有 signals 或 scales 顶层字段，通常是 Vega
    if "signals" in spec or ("scales" in spec and "encoding" not in spec):
        return True
    # 如果 data 是列表，通常是 Vega
    if isinstance(spec.get("data"), list):
        return True
    return False


__all__ = [
    'setup_logger', 'app_logger', 'error_logger',
    'encode_image_to_base64', 'decode_base64_to_image', 'create_data_url',
    'safe_json_loads', 'safe_json_dumps', 'extract_json_from_text',
    'get_spec_data_values', 'get_spec_data_count', 'is_vega_full_spec',
]
