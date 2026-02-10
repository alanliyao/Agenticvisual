"""
会话管理器
负责会话状态维护、意图识别、模式分发
"""

import copy
import json
from pathlib import Path
from typing import Dict, Optional
import uuid
import time

from config.chart_types import ChartType, get_candidate_chart_types
from config.intent_types import IntentType
from core.data_manager import LargeDatasetManager
from core.vlm_service import get_vlm_service
from core.vega_service import get_vega_service
from core.modes import ChitchatMode, GoalOrientedMode, AutonomousExplorationMode
from prompts import get_prompt_manager
from core.utils import app_logger, get_spec_data_count, get_spec_data_values, is_vega_full_spec
from tools import sankey_tools


class SessionManager:
    """session manager"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> session_data
        self.vlm = get_vlm_service()
        self.vega = get_vega_service()
        self.prompt_mgr = get_prompt_manager()
        
        # initialize each mode
        self.chitchat_mode = ChitchatMode()
        self.goal_mode = GoalOrientedMode()
        self.explore_mode = AutonomousExplorationMode()
        
        app_logger.info("Session Manager initialized")
    
    def create_session(self, vega_spec: Dict) -> str:
        """
        create a new session
        
        Args:
            vega_spec: Vega-Lite JSON specification
        
        Returns:
            session_id: the id of the session
        """
        session_id = str(uuid.uuid4())
        working_spec = copy.deepcopy(vega_spec)

        # if there is a large dataset configuration, initialize the manager and do the first sampling
        original_count = get_spec_data_count(working_spec)
        data_manager = self._maybe_init_data_manager(working_spec)
        if data_manager:
            initial_values = data_manager.init_sample()
            working_spec.setdefault("data", {})["values"] = initial_values
            app_logger.info(
                f"large dataset detected: {original_count} points -> "
                f"sampled to {len(initial_values)} points (view_limit={data_manager.view_limit})"
            )
        else:
            app_logger.info(f"dataset size: {original_count} points (no sampling needed)")
        
        # Sankey diagram auto collapse: if it is a Vega format and the number of nodes is too many
        working_spec = self._maybe_auto_collapse_sankey(working_spec)
        
        # render the initial view
        render_result = self.vega.render(working_spec)
        
        if not render_result.get("success"):
            app_logger.error("Failed to render initial view")
            return None
        
        # identify the chart type
        chart_type = self._identify_chart_type(
            working_spec,
            render_result["image_base64"]
        )
        
        # create session data
        self.sessions[session_id] = {
            "session_id": session_id,
            "vega_spec": working_spec,
            "original_spec": working_spec,  # 保存原始规范
            "current_image": render_result["image_base64"],
            "chart_type": chart_type,
            "conversation_history": [],
            "created_at": time.time(),
            "last_activity": time.time(),
            "data_manager": data_manager,
            "base_dir": str(Path(__file__).resolve().parent.parent),
            "spec_history": []
        }
        
        app_logger.info(f"Session created: {session_id}, chart_type: {chart_type}")
        return session_id

    def _maybe_init_data_manager(self, vega_spec: Dict) -> Optional[LargeDatasetManager]:
        """when the data amount is greater than the view limit or provided full_data_path, initialize the data manager."""
        # Vega format (such as Sankey diagram) does not apply data sampling
        if is_vega_full_spec(vega_spec):
            return None
        
        meta = vega_spec.get("_metadata") or {}
        view_limit = meta.get("view_limit", 500)
        full_data_path = meta.get("full_data_path")

        encoding = vega_spec.get("encoding", {})
        x_field = encoding.get("x", {}).get("field")
        y_field = encoding.get("y", {}).get("field")

        full_values = get_spec_data_values(vega_spec) or []

        # if provided full_data_path, load it first
        if full_data_path:
            try:
                base_dir = Path(__file__).resolve().parent.parent
                data_path = (base_dir / full_data_path).resolve()
                if data_path.exists():
                    loaded = json.loads(data_path.read_text(encoding="utf-8")).get("values", [])
                    if loaded:
                        full_values = loaded
                else:
                    app_logger.warning(f"full_data_path not found: {data_path}")
            except Exception as exc:  # noqa: BLE001
                app_logger.error(f"failed to load full_data_path {full_data_path}: {exc}")

        if not full_values:
            return None

        if len(full_values) <= int(view_limit or 500):
            # the data amount is not greater than the limit, do not enable the manager
            return None

        return LargeDatasetManager(
            full_values=full_values,
            x_field=x_field,
            y_field=y_field,
            view_limit=view_limit,
        )
    
    def _maybe_auto_collapse_sankey(self, vega_spec: Dict, nodes_per_layer: int = 5) -> Dict:
        """
        Sankey diagram auto collapse: if the number of nodes in each layer is greater than the threshold, auto collapse
        
        this is the implementation of the "physical interaction necessity" of the Sankey diagram:
        - similar to the automatic sampling of the scatter plot
        - initially only display the top N nodes in each layer
        - the user must expand the node to see the collapsed nodes
        
        Args:
            vega_spec: Vega specification
            nodes_per_layer: the number of nodes to keep in each layer (default 5)
        
        Returns:
            the vega_spec that may have been collapsed
        """
        # only process the Vega format (Sankey diagram)
        if not is_vega_full_spec(vega_spec):
            return vega_spec
        
        # check if there are nodes and links data (Sankey diagram feature)
        data = vega_spec.get("data", [])
        if not isinstance(data, list):
            return vega_spec
        
        nodes_data = None
        for d in data:
            if isinstance(d, dict) and d.get("name") == "nodes":
                nodes_data = d.get("values", [])
                break
        
        if not nodes_data:
            return vega_spec
        
        # count the number of nodes by layer
        depth_counts = {}
        for node in nodes_data:
            depth = node.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # check if collapse is needed (any layer exceeds the threshold)
        needs_collapse = any(count > nodes_per_layer for count in depth_counts.values())
        
        if not needs_collapse:
            app_logger.info(f"sankey: {len(nodes_data)} nodes, no auto-collapse needed")
            return vega_spec
        
        # call auto collapse
        result = sankey_tools.auto_collapse_by_rank(vega_spec, top_n=nodes_per_layer)
        
        if result.get("success"):
            collapsed_info = result.get("collapsed_groups", {})
            total_collapsed = sum(len(nodes) for nodes in collapsed_info.values())
            app_logger.info(
                f"large sankey auto-collapsed: {len(nodes_data)} nodes -> "
                f"kept top {nodes_per_layer} per layer, {total_collapsed} nodes collapsed into {len(collapsed_info)} groups"
            )
            return result["vega_spec"]
        else:
            app_logger.warning(f"Sankey auto-collapse failed: {result.get('error')}")
            return vega_spec
    
    def _identify_chart_type(self, vega_spec: Dict, image_base64: str) -> ChartType:
        """identify the chart type"""
        # 首先从Vega规范推测
        candidates = get_candidate_chart_types(vega_spec)
        
        if len(candidates) == 1 and candidates[0] != ChartType.UNKNOWN:
            return candidates[0]
        
        # if cannot determine, use VLM visual recognition
        prompt = """请识别这个图表的类型。返回JSON格式：
{
    "chart_type": "bar_chart|line_chart|scatter_plot|parallel_coordinates|heatmap|sankey_diagram",
    "confidence": 0.0-1.0,
    "reasoning": "判断理由"
}"""
        
        response = self.vlm.call_with_image(prompt, image_base64, expect_json=True)
        
        if response.get("success"):
            parsed = response.get("parsed_json", {})
            chart_type_str = parsed.get("chart_type", "unknown")
            
            # convert to ChartType enum
            for ct in ChartType:
                if ct.value == chart_type_str:
                    return ct
        
        return ChartType.UNKNOWN
    
    def process_query(self, session_id: str, user_query: str, benchmark_mode: bool = False) -> Dict:
        """
        process the user query
        
        Args:
            session_id: the id of the session
            user_query: the user query text
            benchmark_mode: whether in benchmark evaluation mode
        
        Returns:
            the processing result
        """
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        session["last_activity"] = time.time()
        
        # 1. intent recognition
        intent = self._recognize_intent(
            user_query,
            session["current_image"],
            session["chart_type"]
        )
        
        # 2. dispatch to different modes based on the intent
        if intent == IntentType.CHITCHAT:
            result = self.chitchat_mode.execute(user_query, session["current_image"], session)
        elif intent == IntentType.EXPLICIT_ANALYSIS:
            result = self.goal_mode.execute(
                user_query,
                session["vega_spec"],
                session["current_image"],
                session["chart_type"],
                session,
                benchmark_mode=benchmark_mode
            )
            # update the session state
            if result.get("success"):
                session["vega_spec"] = result.get("final_spec", session["vega_spec"])
                session["current_image"] = result.get("final_image", session["current_image"])
        else:  # VAGUE_EXPLORATION
            result = self.explore_mode.execute(
                user_query,
                session["vega_spec"],
                session["current_image"],
                session["chart_type"],
                session
            )
        
        # 3. update the conversation history
        session["conversation_history"].append({
            "query": user_query,
            "intent": intent.value if isinstance(intent, IntentType) else str(intent),
            "result": result,
            "timestamp": time.time()
        })
        
        return result

    def load_region(self, session_id: str, region: Dict, current_spec: Dict) -> Dict:
        """load incremental data based on the region, return the new vega_spec."""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}

        session = self.sessions[session_id]
        data_manager: Optional[LargeDatasetManager] = session.get("data_manager")
        if not data_manager:
            return {"success": False, "error": "No data manager for session"}

        new_values = data_manager.load_region(region)
        new_spec = copy.deepcopy(current_spec)
        new_spec.setdefault("data", {})["values"] = new_values

        return {"success": True, "vega_spec": new_spec}
    
    def _recognize_intent(self, user_query: str, image_base64: str, 
                         chart_type: ChartType) -> IntentType:
        """identify the user intent"""
        # 快速判断：基于关键词识别明显的意图
        query_lower = user_query.lower().strip()
        
        # greetings keywords
        greetings = [
            '你好', '您好', 'hi', 'hello', 'hey', '嗨', 'hola',
            '早上好', '中午好', '下午好', '晚上好', '早安', '晚安'
        ]
        
        # polite words keywords
        polite_words = [
            '谢谢', '多谢', 'thanks', 'thank you', 'thx',
            '再见', '拜拜', 'bye', 'goodbye', 'see you'
        ]
        
        # system query keywords
        system_queries = [
            '你是谁', '你叫什么', '你能做什么', '你会什么',
            '怎么用', '怎么使用', '如何使用', '使用方法',
            'what can you do', 'how to use', 'who are you'
        ]
        
        # explicit action keywords (表示 EXPLICIT_ANALYSIS)
        explicit_actions = [
            '筛选', '过滤', 'filter', 'select',
            '放大', '缩小', 'zoom', 'scale',
            '高亮', '突出', 'highlight', 'emphasize',
            '排序', 'sort', 'order',
            '显示', '隐藏', 'show', 'hide',
            '对比', '比较', 'compare', 'contrast',
            '选择', '选中', 'choose', 'pick',
            '调整', '修改', 'adjust', 'modify',
            '聚焦', 'focus',
            '只看', '只显示', 'only show',
            '去掉', '删除', 'remove', 'delete',
            '添加', '增加', 'add',
            '改成', '换成', 'change to',
            '设置为', 'set to'
        ]
        
        # check if it is a pure greeting or polite words (length less than 10 characters)
        if len(query_lower) < 10:
            for greeting in greetings:
                if greeting in query_lower:
                    app_logger.info(f"Quick intent recognition: CHITCHAT (greeting: {greeting})")
                    return IntentType.CHITCHAT
            
            for polite in polite_words:
                if polite in query_lower:
                    app_logger.info(f"Quick intent recognition: CHITCHAT (polite: {polite})")
                    return IntentType.CHITCHAT
        
        # check if it is a system query (even if it is a long sentence)
        for sys_query in system_queries:
            if sys_query in query_lower:
                app_logger.info(f"Quick intent recognition: CHITCHAT (system query: {sys_query})")
                return IntentType.CHITCHAT
        
        # check if it is an explicit action keyword
        for action in explicit_actions:
            if action in query_lower:
                app_logger.info(f"Quick intent recognition: EXPLICIT_ANALYSIS (action: {action})")
                return IntentType.EXPLICIT_ANALYSIS
        
        # use VLM to identify the intent
        intent_prompt = self.prompt_mgr.get_intent_recognition_prompt(
            user_query=user_query,
            chart_type=chart_type
        )
        
        response = self.vlm.call_with_image(
            intent_prompt, image_base64,
            expect_json=True
        )
        
        if response.get("success"):
            parsed = response.get("parsed_json", {})
            intent_str = parsed.get("intent_type", "unknown")
            
            app_logger.info(f"VLM intent recognition: {intent_str}")
            
            for it in IntentType:
                if it.value == intent_str:
                    return it
        
        return IntentType.UNKNOWN
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """get the session data"""
        return self.sessions.get(session_id)
    
    def reset_view(self, session_id: str) -> Dict:
        """reset the view to the original state"""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        session = self.sessions[session_id]
        session["vega_spec"] = session["original_spec"]
        
        render_result = self.vega.render(session["vega_spec"])
        if render_result.get("success"):
            session["current_image"] = render_result["image_base64"]
        
        return {"success": True, "message": "View reset to original state"}


_session_manager = None

def get_session_manager() -> SessionManager:
    """get the session manager singleton"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
