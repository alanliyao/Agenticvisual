"""
Benchmark Annotation System - Backend
实现散点图区域重采样和Sankey图自动折叠机制
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import copy
import traceback
import sys
import random
import re

# ============================================================
# 项目路径配置
# ============================================================
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 常量配置
# ============================================================
MAX_SCATTER_POINTS = 500      # 散点图每个视图最大显示点数
MAX_PARALLEL_LINES = 100      # 平行坐标图最多显示线条数
SANKEY_TOP_N_PER_LAYER = 5    # Sankey图每层最多显示节点数

# ============================================================
# FastAPI应用
# ============================================================
app = FastAPI(title="Benchmark Annotation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 全局状态
# ============================================================
_specs: List[Dict] = []                           # 所有加载的spec
_original_data_store: Dict[int, List[Dict]] = {}  # spec_index -> 原始完整数据
_spec_metadata: Dict[int, Dict] = {}              # spec_index -> {x_field, y_field, ...}
_tool_executor = None


# ============================================================
# 散点图数据管理器 - 区域重采样
# ============================================================
class ScatterDataManager:
    """
    管理大型散点图数据集的区域重采样
    
    核心功能：
    1. init_sample(): 初始随机采样到MAX_SCATTER_POINTS
    2. load_region(): 从原始数据筛选区域内的点，再采样到MAX_SCATTER_POINTS
    """
    
    def __init__(self, full_data: List[Dict], x_field: str, y_field: str, max_points: int = 500):
        self.full_data = full_data
        self.x_field = x_field
        self.y_field = y_field
        self.max_points = max_points
        self.total_points = len(full_data)
        
        # 计算数据范围
        x_values = [d.get(x_field) for d in full_data if d.get(x_field) is not None]
        y_values = [d.get(y_field) for d in full_data if d.get(y_field) is not None]
        
        if x_values and y_values:
            self.x_min, self.x_max = min(x_values), max(x_values)
            self.y_min, self.y_max = min(y_values), max(y_values)
        else:
            self.x_min = self.x_max = self.y_min = self.y_max = 0
    
    def init_sample(self) -> Tuple[List[Dict], Dict]:
        """
        初始随机采样
        返回: (采样数据, 采样信息)
        """
        if len(self.full_data) <= self.max_points:
            return self.full_data, {
                "sampled": False,
                "displayed": len(self.full_data),
                "total": len(self.full_data),
                "message": f"显示全部 {len(self.full_data)} 个数据点"
            }
        
        sampled = random.sample(self.full_data, self.max_points)
        return sampled, {
            "sampled": True,
            "displayed": self.max_points,
            "total": self.total_points,
            "message": f"从 {self.total_points} 个点中随机采样 {self.max_points} 个显示",
            "hint": "使用 zoom 工具缩放到感兴趣的区域可以看到更多数据点"
        }
    
    def load_region(self, x_range: List[float], y_range: List[float]) -> Tuple[List[Dict], Dict]:
        """
        加载指定区域内的数据点
        如果区域内点数超过max_points，则随机采样
        
        参数:
            x_range: [x_min, x_max]
            y_range: [y_min, y_max]
        返回: (区域数据, 采样信息)
        """
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        
        # 筛选区域内的点
        region_data = []
        for d in self.full_data:
            x_val = d.get(self.x_field)
            y_val = d.get(self.y_field)
            if x_val is not None and y_val is not None:
                if x_min <= x_val <= x_max and y_min <= y_val <= y_max:
                    region_data.append(d)
        
        region_total = len(region_data)
        
        # 如果区域内点数超过限制，进行采样
        if region_total > self.max_points:
            sampled = random.sample(region_data, self.max_points)
            return sampled, {
                "sampled": True,
                "displayed": self.max_points,
                "region_total": region_total,
                "total": self.total_points,
                "message": f"区域内有 {region_total} 个点，采样显示 {self.max_points} 个",
                "hint": "继续缩放可以看到更多区域内的数据点"
            }
        
        return region_data, {
            "sampled": False,
            "displayed": region_total,
            "region_total": region_total,
            "total": self.total_points,
            "message": f"区域内共 {region_total} 个点，全部显示"
        }


# ============================================================
# Sankey图工具导入（统一使用 main 的 tools）
# ============================================================
try:
    from tools import sankey_tools
except ImportError:
    sankey_tools = None


# ============================================================
# 图表类型检测
# ============================================================
def detect_chart_type(spec: Dict) -> str:
    """从spec检测图表类型"""
    # 简化策略：优先按标题关键词识别（你要求的主路径）
    metadata = spec.get('_metadata', {}) or {}

    def _title_text() -> str:
        # title can be str or {"text": "..."}
        t = spec.get('title', '')
        if isinstance(t, dict):
            t = t.get('text', '') or ''
        if not t and isinstance(metadata.get('title'), str):
            t = metadata.get('title') or ''
        if not t and isinstance(spec.get('description'), str):
            t = spec.get('description') or ''
        return str(t)

    title_lower = _title_text().lower()
    title_keywords = [
        ('sankey', 'sankey'),
        ('parallel', 'parallel'),
        ('heatmap', 'heatmap'),
        ('heat map', 'heatmap'),
        ('scatter', 'scatter'),
        ('bar chart', 'bar'),
        ('bar', 'bar'),
        ('line chart', 'line'),
        ('line', 'line'),
    ]
    for keyword, chart_type in title_keywords:
        if keyword in title_lower:
            return chart_type
    
    # 2. 检查Sankey（Vega格式）
    if isinstance(spec.get('data'), list):
        data_names = [d.get('name') for d in spec['data'] if isinstance(d, dict)]
        if 'nodes' in data_names or 'links' in data_names:
            return 'sankey'
    
    # 3. 获取基本信息
    transforms = spec.get('transform', [])
    encoding = spec.get('encoding', {})
    x_enc = encoding.get('x', {})
    y_enc = encoding.get('y', {})
    
    x_type = x_enc.get('type', '')
    x_field = x_enc.get('field', '')
    
    # 4. 检查是否有fold transform
    has_fold = any(isinstance(t, dict) and t.get('fold') for t in transforms)
    
    # 5. 检查是否是平行坐标图的特征
    if has_fold:
        # 平行坐标图的关键特征：
        # - X轴是nominal/ordinal类型，且field通常是"key"（fold产生的）
        # - 有joinaggregate transform（用于归一化）
        # - 有detail encoding（用于连接同一数据点的不同维度）
        
        has_joinaggregate = any(
            isinstance(t, dict) and t.get('joinaggregate') 
            for t in transforms
        )
        has_detail = 'detail' in encoding
        x_is_nominal = x_type in ['nominal', 'ordinal']
        x_is_key_field = x_field in ['key', 'variable', 'dimension']  # fold常用的字段名
        
        # 如果X轴是时间类型 → 折线图
        if x_type == 'temporal':
            return 'line'
        
        # 如果有归一化或detail encoding，且X是nominal → 平行坐标图
        if x_is_nominal and (has_joinaggregate or has_detail or x_is_key_field):
            return 'parallel'
        
        # 如果X是nominal但没有其他平行坐标图特征 → 可能还是折线图（多系列）
        # 需要进一步检查：看layer结构
    
    # 6. 检查repeat（平行坐标图的另一种实现方式）
    if spec.get('repeat'):
        return 'parallel'
    
    # 7. 检查layer结构
    layers = spec.get('layer', [])
    if layers:
        # 平行坐标图通常有rule mark作为坐标轴
        has_rule_mark = any(
            (layer.get('mark') == 'rule' or 
             (isinstance(layer.get('mark'), dict) and layer.get('mark', {}).get('type') == 'rule'))
            for layer in layers
        )
        
        # 检查layer中line的X轴类型
        for layer in layers:
            layer_mark = layer.get('mark', {})
            mark_type = layer_mark if isinstance(layer_mark, str) else layer_mark.get('type', '')
            
            if mark_type == 'line':
                layer_x = layer.get('encoding', {}).get('x', {})
                layer_x_type = layer_x.get('type', '')
                layer_x_field = (layer_x.get('field') or '').lower()
                x_looks_temporal = (
                    layer_x_type == 'temporal' or
                    layer_x_field in ('year', 'date', 'month', 'time') or
                    'year' in layer_x_field or 'date' in layer_x_field
                )
                # 如果line的X轴是时间 → 折线图
                if x_looks_temporal:
                    return 'line'
                # 如果line的X轴是nominal且有rule → 平行坐标图
                if layer_x_type in ['nominal', 'ordinal'] and has_rule_mark:
                    return 'parallel'
    
    # 8. 根据顶层X轴类型判断（针对非fold的情况）
    if x_type == 'temporal':
        return 'line'
    
    # 9. 检查mark类型
    mark = spec.get('mark', {})
    mark_type = mark if isinstance(mark, str) else mark.get('type', '')
    
    mark_mapping = {
        'bar': 'bar',
        'line': 'line',
        'point': 'scatter',
        'circle': 'scatter',
        'rect': 'heatmap',
        'area': 'line'
    }
    
    if mark_type in mark_mapping:
        return mark_mapping[mark_type]
    
    return 'scatter'


def is_vega_full_spec(spec: Dict) -> bool:
    """判断是否为完整Vega规格（非Vega-Lite）"""
    schema = spec.get('$schema', '')
    if 'vega/v' in schema and 'vega-lite' not in schema:
        return True
    # 无schema但有典型Vega结构
    if not schema and isinstance(spec.get('data'), list):
        if any(d.get('name') for d in spec['data'] if isinstance(d, dict)):
            return True
    return False


def get_encoding_fields(spec: Dict) -> Tuple[str, str]:
    """从spec提取x/y字段名"""
    encoding = spec.get('encoding', {})
    x_field = encoding.get('x', {}).get('field', 'x')
    y_field = encoding.get('y', {}).get('field', 'y')
    return x_field, y_field


def _sample_parallel_data(spec: Dict, max_lines: int = None) -> bool:
    """
    对平行坐标图 data.values 按线条 ID 采样，超过 max_lines 时保留 max_lines 条线。
    直接修改 spec['data']['values']。返回是否进行了采样。
    """
    if max_lines is None:
        max_lines = MAX_PARALLEL_LINES
    if not isinstance(spec.get('data'), dict):
        return False
    vals = spec['data'].get('values', [])
    if not vals:
        return False
    line_id_field = '_index'
    if line_id_field not in vals[0]:
        for k in ('index', 'id', 'line_id'):
            if k in vals[0]:
                line_id_field = k
                break
    line_ids = list({r.get(line_id_field) for r in vals if r.get(line_id_field) is not None})
    if len(line_ids) <= max_lines:
        return False
    keep = set(random.sample(line_ids, max_lines))
    spec['data']['values'] = [r for r in vals if r.get(line_id_field) in keep]
    return True


# ============================================================
# 加载Specs
# ============================================================
def load_specs():
    """从specs目录加载所有规格文件"""
    global _specs, _original_data_store, _spec_metadata
    
    _specs = []
    _original_data_store = {}
    _spec_metadata = {}
    
    specs_dir = BACKEND_DIR / "specs"
    if not specs_dir.exists():
        specs_dir.mkdir(parents=True)
        print(f"Created specs directory: {specs_dir}")
        return
    

    json_files = sorted(
    specs_dir.glob("*.json"), 
    key=lambda p: int(p.name.split('_')[0])
)
    
    for idx, fp in enumerate(json_files):
        # 跳过.DS_Store等隐藏文件
        if fp.name.startswith('.'):
            continue
            
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取spec和meta
            if 'spec' in data:
                spec = copy.deepcopy(data['spec'])
                meta = data.get('meta', {})
            else:
                spec = copy.deepcopy(data)
                meta = {}
            
            # 检测图表类型
            chart_type = meta.get('chart_type') or detect_chart_type(spec)
            is_full_vega = is_vega_full_spec(spec)
            
            # 提取原始数据
            original_data = []
            if isinstance(spec.get('data'), dict) and 'values' in spec['data']:
                original_data = spec['data']['values']
            elif isinstance(spec.get('data'), list):
                for d in spec['data']:
                    if isinstance(d, dict) and d.get('values'):
                        original_data = d['values']
                        break
            
            total_points = len(original_data)
            sampling_info = None
            collapse_info = None
            
            # ========== 散点图重采样 ==========
            if chart_type == 'scatter' and total_points > MAX_SCATTER_POINTS:
                # 存储原始数据
                _original_data_store[idx] = original_data
                
                # 获取x/y字段
                x_field, y_field = get_encoding_fields(spec)
                _spec_metadata[idx] = {'x_field': x_field, 'y_field': y_field}
                
                # 创建数据管理器并采样
                dm = ScatterDataManager(original_data, x_field, y_field, MAX_SCATTER_POINTS)
                sampled_data, sampling_info = dm.init_sample()
                
                # 更新spec中的数据
                spec['data']['values'] = sampled_data
                
                print(f"  [Scatter] {fp.name}: 采样 {len(sampled_data)}/{total_points} 点")
            
            # ========== 平行坐标图线条数限制 ==========
            elif chart_type == 'parallel':
                n_before = len(spec.get('data', {}).get('values', [])) if isinstance(spec.get('data'), dict) else 0
                if _sample_parallel_data(spec, MAX_PARALLEL_LINES):
                    n_after = len(spec['data']['values'])
                    print(f"  [Parallel] {fp.name}: 采样 {n_after}/{n_before} 行 ({MAX_PARALLEL_LINES} 条线)")
            
            # ========== Sankey图自动折叠（使用 sankey_tools，与 main 一致）==========
            elif chart_type == 'sankey' and isinstance(spec.get('data'), list):
                if sankey_tools:
                    # 使用 sankey_tools.auto_collapse_by_rank（和 main 的 session_manager 一致）
                    collapse_result = sankey_tools.auto_collapse_by_rank(spec, top_n=SANKEY_TOP_N_PER_LAYER)
                    if collapse_result.get('success'):
                        spec = collapse_result['vega_spec']
                        collapse_info = collapse_result.get('collapsed_groups', {})
                        total_collapsed = sum(len(nodes) for nodes in collapse_info.values())
                        # 调试：检查 _sankey_state 是否在返回的 spec 中
                        has_state = '_sankey_state' in spec
                        state_keys = list(spec.get('_sankey_state', {}).keys()) if has_state else []
                        print(f"  [Sankey] {fp.name}: 每层保留 {SANKEY_TOP_N_PER_LAYER} 个节点，折叠了 {total_collapsed} 个节点")
                        print(f"  [DEBUG] load_specs({fp.name}): spec has _sankey_state={has_state}, keys={state_keys}")
                    else:
                        print(f"  [Sankey] {fp.name}: 自动折叠失败: {collapse_result.get('error')}")
                        collapse_info = {}
                else:
                    print(f"  [Sankey] {fp.name}: sankey_tools 未加载，跳过自动折叠")
                    collapse_info = {}
            
            # 构建meta信息
            displayed_points = len(spec.get('data', {}).get('values', [])) if isinstance(spec.get('data'), dict) else total_points
            
            meta.update({
                'chart_type': chart_type,
                'title': meta.get('title', spec.get('title', fp.stem)),
                'filename': fp.name,
                'index': idx,  # 添加 index 用于 exportTask
                'data_points': displayed_points,
                'total_points': total_points,
                'is_sampled': chart_type == 'scatter' and total_points > MAX_SCATTER_POINTS,
                'is_vega_full': is_full_vega,
                'sampling_info': sampling_info,
                'collapse_info': collapse_info
            })
            
            _specs.append({
                'spec': spec,
                'meta': meta,
                'file': str(fp)
            })
            
            print(f"Loaded: {fp.name} ({chart_type}, {displayed_points}/{total_points} points)")
            
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            traceback.print_exc()
    
    print(f"\nTotal loaded: {len(_specs)} specs")


# ============================================================
# Tool Executor
# ============================================================
def load_tool_executor():
    """尝试加载项目的tool executor"""
    global _tool_executor
    
    try:
        from tools.tool_executor import get_tool_executor
        _tool_executor = get_tool_executor()
        print("✓ Tool executor loaded")
        return True
    except ImportError as e:
        print(f"⚠ Tool executor not available: {e}")
        return False


# ============================================================
# API Models
# ============================================================
class ToolRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any]
    vega_spec: Dict[str, Any]
    spec_index: Optional[int] = None  # 用于标识当前spec以获取原始数据


class FinishRequest(BaseModel):
    question: str
    answer: Optional[Any] = None  # str or structured {type, value, ...}
    answer_type: str = "categorical"  # categorical | numeric | boolean | open_ended | region
    answer_config: Dict[str, Any] = {}  # type-specific config: alternatives, tolerance, etc.
    iterations: List[Dict[str, Any]]
    final_spec: Dict[str, Any]
    original_spec: Dict[str, Any]
    chart_type: str
    task_type: str  # clear_single | clear_multi | vague_single | vague_multi (required, no default)
    meta: Dict[str, Any] = {}


# ============================================================
# 启动事件
# ============================================================
@app.on_event("startup")
async def startup():
    load_specs()
    load_tool_executor()


# ============================================================
# API Endpoints
# ============================================================
@app.get("/api/specs")
async def get_specs():
    """获取所有specs列表"""
    return {
        "specs": [
            {
                "index": i,
                "chart_type": s["meta"].get("chart_type", "unknown"),
                "title": s["meta"].get("title", f"Spec {i}"),
                "filename": s["meta"].get("filename", ""),
                "data_points": s["meta"].get("data_points", 0),
                "total_points": s["meta"].get("total_points", 0),
                "is_sampled": s["meta"].get("is_sampled", False),
                "is_vega_full": s["meta"].get("is_vega_full", False)
            }
            for i, s in enumerate(_specs)
        ]
    }


@app.get("/api/spec/{index}")
async def get_spec(index: int):
    """获取指定spec"""
    if index < 0 or index >= len(_specs):
        raise HTTPException(status_code=404, detail="Spec not found")
    
    spec_data = _specs[index]
    spec = spec_data["spec"]
    # 调试：检查返回的 spec 是否包含 _sankey_state
    if isinstance(spec.get('data'), list) and any(d.get('name') in ['nodes', 'links'] for d in spec.get('data', [])):
        has_state = '_sankey_state' in spec
        state_keys = list(spec.get('_sankey_state', {}).keys()) if has_state else []
        print(f"[DEBUG] get_spec({index}): spec has _sankey_state={has_state}, keys={state_keys}")
    
    # 使用 JSONResponse 确保所有键（包括 _sankey_state）都被保留
    response_data = {
        "spec": spec,
        "meta": spec_data["meta"],
        "index": index
    }
    # 使用 json.dumps 确保 _sankey_state 被序列化
    return JSONResponse(content=response_data)


@app.post("/api/execute_tool")
async def execute_tool_api(request: ToolRequest):
    """执行工具并处理重采样/展开逻辑"""
    tool_name = request.tool_name
    params = request.params
    vega_spec = request.vega_spec
    spec_index = request.spec_index
    
    # 调试：检查 _sankey_state 是否在请求中
    if tool_name in ['expand_node', 'collapse_nodes']:
        has_state = '_sankey_state' in vega_spec
        state_keys = list(vega_spec.get('_sankey_state', {}).keys()) if has_state else []
        print(f"[DEBUG] {tool_name}: vega_spec has _sankey_state={has_state}, keys={state_keys}")
    
    # 清理内部字段
    internal_fields = {'clicked', 'clickedDatum', 'brush', 'brush_x', 'brush_y', 'spec_index', 'original_spec', 'spec_history'}
    clean_params = {k: v for k, v in params.items() if k not in internal_fields}
    
    # 准备执行参数
    exec_params = copy.deepcopy(clean_params)
    exec_params['vega_spec'] = copy.deepcopy(vega_spec)
    
    # ========== 特殊工具处理 ==========
    if tool_name == 'reset_view':
        # Read from vega_spec metadata instead of params
        original = vega_spec.get('_original_spec')
        if not original:
            # Fallback to params for backward compatibility
            original = params.get('original_spec', vega_spec)
        return {
            "success": True,
            "new_spec": original,
            "result": {"reset": True}
        }
    
    if tool_name == 'undo_view':
        # Read from vega_spec metadata instead of params
        history = vega_spec.get('_spec_history')
        if not history:
            # Fallback to params for backward compatibility
            history = params.get('spec_history', [])
        if history and isinstance(history, list) and len(history) > 0:
            return {
                "success": True,
                "new_spec": history[-1],
                "result": {"undone": True}
            }
        return {"success": False, "result": {"error": "No history to undo"}}
    
    # ========== 使用Tool Executor执行（包括 expand_node，统一使用 sankey_tools）==========
    if _tool_executor:
        result = _tool_executor.execute(tool_name, exec_params, validate=True)
        
        if result.get('success', False):
            new_spec = result.get('vega_spec', vega_spec)
            # 调试：检查返回的 new_spec 是否包含 _sankey_state
            if tool_name in ['expand_node', 'collapse_nodes']:
                has_state = '_sankey_state' in new_spec
                state_keys = list(new_spec.get('_sankey_state', {}).keys()) if has_state else []
                print(f"[DEBUG] {tool_name} result: new_spec has _sankey_state={has_state}, keys={state_keys}")
            sampling_info = None
            
            # ========== 区域缩放后的重采样 ==========
            if tool_name in ['zoom_dense_area', 'select_region', 'brush_region', 'zoom_to_region']:
                if spec_index is not None and spec_index in _original_data_store:
                    # 获取区域范围
                    x_range = clean_params.get('x_range')
                    y_range = clean_params.get('y_range')
                    
                    # 也检查brush参数
                    if not x_range and 'brush' in params:
                        brush = params['brush']
                        if isinstance(brush, dict):
                            x_range = brush.get('x')
                            y_range = brush.get('y')
                    
                    # 从scale domain获取范围
                    if not x_range:
                        encoding = new_spec.get('encoding', {})
                        x_scale = encoding.get('x', {}).get('scale', {})
                        y_scale = encoding.get('y', {}).get('scale', {})
                        x_range = x_scale.get('domain')
                        y_range = y_scale.get('domain')
                    
                    if x_range and y_range:
                        original_data = _original_data_store[spec_index]
                        metadata = _spec_metadata.get(spec_index, {})
                        x_field = metadata.get('x_field', 'x')
                        y_field = metadata.get('y_field', 'y')
                        
                        # 重新采样
                        dm = ScatterDataManager(original_data, x_field, y_field, MAX_SCATTER_POINTS)
                        region_data, sampling_info = dm.load_region(x_range, y_range)
                        
                        # 更新spec中的数据
                        if isinstance(new_spec.get('data'), dict):
                            new_spec['data']['values'] = region_data
            
            # 使用 JSONResponse 确保所有键（包括 _sankey_state）都被保留
            response_data = {
                "success": True,
                "new_spec": new_spec,
                "result": {k: v for k, v in result.items() if k not in ['success', 'vega_spec']},
                "sampling_info": sampling_info
            }
            # 确保 _sankey_state 在 new_spec 中被保留
            if tool_name in ['expand_node', 'collapse_nodes'] and '_sankey_state' in new_spec:
                print(f"[DEBUG] execute_tool_api return: new_spec has _sankey_state with keys: {list(new_spec['_sankey_state'].keys())}")
            return JSONResponse(content=response_data)
        else:
            return {
                "success": False,
                "result": {
                    "error": result.get('error', 'Unknown error'),
                    "details": result.get('details', [])
                }
            }
    
    # 无tool executor
    return {
        "success": False,
        "result": {"error": "Tool executor not available"}
    }


def _infer_param_type(value: Any) -> str:
    """推断参数的评估类型"""
    if isinstance(value, list):
        return "list"
    elif isinstance(value, bool):
        return "categorical"  # boolean 作为 categorical 处理
    elif isinstance(value, (int, float)):
        return "numeric"
    else:
        return "categorical"


def _build_param_eval(params: Dict[str, Any], tool_name: str = None) -> Dict[str, Any]:
    """
    将工具参数转换为评估器兼容的 param_eval 格式。
    
    输入: {"order": "descending"} 或 {"categories": ["a", "b"]}
    输出: {"type": "categorical", "param": "order", "target": "descending"}
          或 {"type": "list", "param": "categories", "target": ["a", "b"]}
    
    如果有多个参数，只取第一个（通常是最重要的）。
    
    reset_view 和 undo_view 不需要参数评估，直接返回空字典。
    """
    # reset_view 和 undo_view 不需要参数评估
    if tool_name in ['reset_view', 'undo_view']:
        return {}

    if not params:
        return {}

    # 通用保存所有参数
    return {
        "type": "object",
        "param": "all_params",
        "target": params
    }


def _build_structured_answer(answer: Any, answer_type: str, answer_config: Dict) -> Dict[str, Any]:
    """
    构建结构化的 answer 对象。
    
    输入: answer="1 Series", answer_type="categorical", answer_config={"alternatives": ["1series"]}
    输出: {"type": "categorical", "value": "1 Series", "alternatives": ["1series"]}
    """
    result = {
        "type": answer_type,
        "value": answer
    }
    
    # 合并 answer_config 中的额外配置
    if answer_type == "categorical":
        result["alternatives"] = answer_config.get("alternatives", [])
    elif answer_type == "numeric":
        result["tolerance"] = answer_config.get("tolerance", 0.05)
    elif answer_type == "region":
        result["metric"] = answer_config.get("metric", "iou")
        result["threshold"] = answer_config.get("threshold", 0.5)
    # open_ended 和 boolean 不需要额外配置
    
    return result


@app.post("/api/finish")
async def finish_question(request: FinishRequest):
    """完成标注并生成benchmark"""
    
    # 提取工具序列
    tools = [it.get('tool_name') for it in request.iterations if it.get('tool_name')]

    # 收集 key_insights（汇总多轮）和 reasoning（每轮一个 {iteration, tool, reasoning}）
    key_insights = []
    reasoning = []
    for i, it in enumerate(request.iterations):
        key_insights.extend(it.get('key_insights', []))
        if it.get('reasoning'):
            reasoning.append({
                'iteration': i + 1,
                'tool': it.get('tool_name'),
                'reasoning': it.get('reasoning')
            })

    state_eval = extract_state_eval(request.final_spec, request.chart_type)

    # tool_eval: 统一为 { tools: [ { tool, param_eval }, ... ] }
    # param_eval 转换为评估器兼容格式
    tool_eval = {
        "tools": [
            {
                "tool": it["tool_name"], 
                "param_eval": _build_param_eval(it.get("parameters", {}), it.get("tool_name"))
            }
            for it in request.iterations if it.get("tool_name")
        ]
    }

    # 构建结构化 answer
    structured_answer = _build_structured_answer(
        request.answer, 
        request.answer_type, 
        request.answer_config
    )

    task_type_abbrev = {'clear_single': 'cs', 'clear_multi': 'cm', 'vague_single': 'vs', 'vague_multi': 'vm'}.get(request.task_type, 'qx')
    qid = f"{task_type_abbrev}_{len(tools):02d}"

    benchmark = {
        'task_type': request.task_type,
        'chart_type': request.chart_type,
        'meta': request.meta,
        'questions': [{
            'qid': qid,
            'question': request.question,
            'ground_truth': {
                'task_type': request.task_type,
                'answer': structured_answer,
                'key_insights': key_insights,
                'reasoning': reasoning,
                'tool_eval': tool_eval,
                'state_eval': state_eval
            }
        }],
        'iterations': request.iterations
    }

    return {"success": True, "benchmark": benchmark}


def extract_state_eval(spec: Dict, chart_type: str) -> Dict:
    """从spec提取状态评估信息"""
    state = {}
    encoding = spec.get('encoding', {})
    transform = spec.get('transform', [])
    
    # 可见域
    x_domain = encoding.get('x', {}).get('scale', {}).get('domain')
    y_domain = encoding.get('y', {}).get('scale', {}).get('domain')
    if x_domain or y_domain:
        state['visible_domain'] = {'x': x_domain, 'y': y_domain}
    
    # 编码字段
    state['encoding'] = {
        'x': encoding.get('x', {}).get('field'),
        'y': encoding.get('y', {}).get('field'),
        'color': encoding.get('color', {}).get('field')
    }
    
    # 过滤器
    filters = [t for t in transform if isinstance(t, dict) and 'filter' in t]
    if filters:
        state['data_filtered'] = filters
    
    # 图层
    layers = spec.get('layer', [])
    if layers:
        state['layers'] = {
            'count': len(layers),
            'types': [
                (m if isinstance(m, str) else (m or {}).get('type'))
                for m in (l.get('mark') for l in layers)
            ]
        }
    
    # Sankey特定状态
    if chart_type == 'sankey' and isinstance(spec.get('data'), list):
        for d in spec['data']:
            if d.get('name') == 'nodes':
                nodes = d.get('values', [])
                state['visible_nodes'] = [n.get('name', n.get('id')) for n in nodes]
                state['collapsed_nodes'] = [
                    n.get('name') for n in nodes 
                    if n.get('_collapsed') or 'Others' in n.get('name', '')
                ]
    
    return state


@app.post("/api/reload")
async def reload_specs():
    """重新加载specs"""
    load_specs()
    return {"success": True, "count": len(_specs)}


@app.get("/api/tools")
async def get_tools():
    """获取可用工具列表"""
    try:
        from tools.tool_registry import tool_registry
        return {"tools": tool_registry.list_all_tools()}
    except ImportError:
        return {"tools": [], "error": "Tool registry not available"}


@app.get("/api/tool/{tool_name}")
async def get_tool_info(tool_name: str):
    """获取工具详情"""
    try:
        from tools.tool_registry import tool_registry
        tool_info = tool_registry.get_tool(tool_name)
        if tool_info:
            return {
                "name": tool_name,
                "description": tool_info.get('description', ''),
                "category": tool_info.get('category', ''),
                "params": {
                    k: {pk: pv for pk, pv in v.items() if pk != 'type' or pv != 'dict'}
                    for k, v in tool_info.get('params', {}).items()
                    if k != 'vega_spec'
                }
            }
        return {"error": f"Tool '{tool_name}' not found"}
    except ImportError:
        return {"error": "Tool registry not available"}


# ============================================================
# 静态文件服务
# ============================================================
frontend_path = BACKEND_DIR.parent / "frontend"

if frontend_path.exists():
    css_path = frontend_path / "css"
    js_path = frontend_path / "js"
    
    if css_path.exists():
        app.mount("/css", StaticFiles(directory=str(css_path)), name="css")
    if js_path.exists():
        app.mount("/js", StaticFiles(directory=str(js_path)), name="js")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))
else:
    print(f"Warning: Frontend path not found: {frontend_path}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 