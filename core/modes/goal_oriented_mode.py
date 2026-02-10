"""目标导向模式"""
from typing import Dict, List
import time
import copy
from core.vlm_service import get_vlm_service
from core.vega_service import get_vega_service
from tools import get_tool_executor
from prompts import get_prompt_manager
from config.settings import Settings
from core.utils import app_logger, get_spec_data_count


class GoalOrientedMode:
    """目标导向模式"""
    
    def __init__(self):
        self.vlm = get_vlm_service()
        self.vega = get_vega_service()
        self.tool_executor = get_tool_executor()
        self.prompt_mgr = get_prompt_manager()
    
    def execute(self, user_query: str, vega_spec: Dict, 
                image_base64: str, chart_type, context: Dict = None, 
                benchmark_mode: bool = False) -> Dict:
        """执行目标导向分析（按DashScope标准多轮对话格式）"""
        if benchmark_mode:
            app_logger.info("🎯 Benchmark mode enabled: ANSWER field will be required in final iteration")
        system_prompt = self.prompt_mgr.assemble_system_prompt(
            chart_type=chart_type,
            mode="goal_oriented",
            include_tools=True,
            benchmark_mode=benchmark_mode
        )
        
        # 从context读取messages历史（如果有）
        messages = context.get('goal_oriented_messages', []) if context else []
        iterations = context.get('goal_oriented_iterations', []) if context else []
        
        # 如果是新会话，初始化第一条user消息
        if len(messages) == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"text": f"请分析这个视图，用户的分析目标是：{user_query}"},
                    {"image": f"data:image/png;base64,{image_base64}"}
                ]
            })
        
        # 保存原始 vega_spec，用于 reset_view 工具
        original_vega_spec = copy.deepcopy(vega_spec)
        
        current_spec = vega_spec
        current_image = image_base64
        
        for iteration in range(Settings.MAX_GOAL_ORIENTED_ITERATIONS):
            # 📊 日志：打印messages结构
            app_logger.info(f"iteration {iteration+1} - messages count: {len(messages)}")
            for idx, msg in enumerate(messages):
                role = msg['role']
                content_items = len(msg.get('content', []))
                has_image = any('image' in c for c in msg.get('content', []))
                app_logger.info(f"  消息{idx}: role={role}, items={content_items}, 含图片={has_image}")
            
            # VLM调用
            response = self.vlm.call(messages, system_prompt, expect_json=True)
            #如果调用失败
            if not response.get("success"):
                app_logger.error(f"iteration {iteration+1} VLM failed: {response.get('error', 'Unknown')}")
                
                # 记录失败的迭代
                iterations.append({
                    "iteration": iteration + 1,
                    "success": False,
                    "error": response.get('error', 'Unknown'),
                    "timestamp": time.time()
                })
                break
            
            # 关键：直接追加VLM返回的assistant消息（按DashScope标准）
            decision = response.get("parsed_json", {})
            assistant_message = {
                "role": "assistant",
                "content": [{"text": response.get("content", "")}]  # VLM原始输出文本
            }
            messages.append(assistant_message)
            
            # 📊 日志
            tool_info = decision.get('tool_call', {}).get('tool', 'None') if decision.get('tool_call') else 'None'
            achieved = decision.get('goal_achieved', False)
            app_logger.info(f"iteration {iteration+1} VLM decision: tool={tool_info}, goal_achieved={achieved}")
            
            # 记录迭代
            iteration_record = {
                "iteration": iteration + 1,
                "success": True,
                "timestamp": time.time(),
                "decision": decision,
                "vlm_raw_output": response.get("content", ""),  # 保存VLM原始输出
                "images": [current_image],
                "analysis_summary": {
                    "key_insights": decision.get("key_insights", []),
                    "reasoning": decision.get("reasoning", "")
                }
            }
            
            # 检查是否达成目标
            if decision.get("goal_achieved", False):
                iterations.append(iteration_record)
                app_logger.info(f"Goal achieved at iteration {iteration + 1}")
                break
            
            # 执行工具
            if decision.get("tool_call"):
                tool_call = decision["tool_call"]
                tool_name = tool_call["tool"]
                tool_params = tool_call.get("params", {})
                tool_params['vega_spec'] = current_spec
                # 只有需要context的工具才传递
                if tool_name in ('reset_view', 'undo_view'):
                    tool_params['context'] = context
                
                tool_result = self.tool_executor.execute(tool_name, tool_params)
                
                # 保存工具执行记录（排除vega_spec避免序列化问题和数据冗余）
                iteration_record["tool_execution"] = {
                    "tool_name": tool_name,
                    "tool_params": {k: v for k, v in tool_params.items() if k not in ('vega_spec', 'context')},
                    "tool_result": {k: v for k, v in tool_result.items() if k != 'vega_spec'}
                }
                
                if tool_result.get("success") and "vega_spec" in tool_result:
                    # 情况1：工具成功且返回新的vega_spec（修改型工具）
                    # 先将旧 spec 入栈（排除 reset/undo）
                    if tool_name not in ['reset_view', 'undo_view']:
                        if context is not None:
                            history = context.setdefault("spec_history", [])
                            history.append(copy.deepcopy(current_spec))

                    current_spec = tool_result["vega_spec"]

                    # 若会话存在大数据管理器，按区域补点
                    current_spec = self._apply_data_manager(current_spec, context)
                    render_result = self.vega.render(current_spec)
                    
                    if render_result.get("success"):
                        current_image = render_result["image_base64"]
                        iteration_record["images"].append(current_image)
                        
                        # 追加user消息：工具成功反馈
                        success_msg = tool_result.get("message", "操作完成")
                        messages.append({
                            "role": "user",
                            "content": [
                                {"text": f"✅ 工具 {tool_name} 执行成功。\n\n结果：{success_msg}\n\n这是更新后的视图："},
                                {"image": f"data:image/png;base64,{current_image}"}
                            ]
                        })
                        
                        app_logger.info(f"Re-rendered chart after {tool_name}: {success_msg}")
                    else:
                        # 渲染失败
                        render_error = render_result.get('error', 'Render failed')
                        app_logger.error(f"Failed to render after {tool_name}: {render_error}")
                        iteration_record["success"] = False
                        
                        messages.append({
                            "role": "user",
                            "content": [
                                {"text": f"❌ 工具 {tool_name} 执行后渲染失败：{render_error}\n\n当前视图（未变化）："},
                                {"image": f"data:image/png;base64,{current_image}"}
                            ]
                        })
                
                elif tool_result.get("success"):
                    # 情况2：工具成功但没有返回vega_spec（分析型工具，如calculate_correlation）
                    analysis_msg = tool_result.get("message", str(tool_result))
                    messages.append({
                        "role": "user",
                        "content": [
                            {"text": f"✅ 工具 {tool_name} 执行成功。\n\n分析结果：{analysis_msg}\n\n视图未变化，当前视图："},
                            {"image": f"data:image/png;base64,{current_image}"}
                        ]
                    })
                    
                    app_logger.info(f"Tool {tool_name} completed (analysis only): {analysis_msg}")
                
                else:
                    # 情况3：工具执行失败
                    error_msg = tool_result.get("error", "Unknown error")
                    messages.append({
                        "role": "user",
                        "content": [
                            {"text": f"❌ 工具 {tool_name} 执行失败。\n\n错误原因：{error_msg}\n\n请选择其他可用工具，或如果目标已达成，设置 goal_achieved: true。\n\n当前视图（未变化）："},
                            {"image": f"data:image/png;base64,{current_image}"}
                        ]
                    })
                    
                    iteration_record["success"] = False
                    app_logger.warning(f"Tool {tool_name} failed: {error_msg}")
            
            iterations.append(iteration_record)
        
        # 保存messages和iterations到context（用于下次调用）
        if context is not None:
            context['goal_oriented_messages'] = messages
            context['goal_oriented_iterations'] = iterations
        
        return {
            "success": True,
            "mode": "goal_oriented",
            "iterations": iterations,
            "final_spec": current_spec,
            "final_image": current_image
        }

    def _extract_region(self, spec: Dict) -> Dict:
        """从 spec 中推测缩放区域（基于 encoding.scale.domain）。"""
        region = {}
        encoding = spec.get("encoding", {}) if isinstance(spec, dict) else {}
        x_enc = encoding.get("x", {}) if isinstance(encoding, dict) else {}
        y_enc = encoding.get("y", {}) if isinstance(encoding, dict) else {}

        def _parse_domain(dom):
            if isinstance(dom, list) and len(dom) == 2:
                try:
                    return float(dom[0]), float(dom[1])
                except Exception:  # noqa: BLE001
                    return None, None
            return None, None

        x_min, x_max = _parse_domain(x_enc.get("scale", {}).get("domain") if isinstance(x_enc.get("scale"), dict) else None)
        y_min, y_max = _parse_domain(y_enc.get("scale", {}).get("domain") if isinstance(y_enc.get("scale"), dict) else None)

        if x_min is not None or x_max is not None:
            region["x_min"] = x_min
            region["x_max"] = x_max
        if y_min is not None or y_max is not None:
            region["y_min"] = y_min
            region["y_max"] = y_max

        region["x_field"] = x_enc.get("field")
        region["y_field"] = y_enc.get("field")

        return region if any(v is not None for v in region.values()) else {}

    def _apply_data_manager(self, spec: Dict, context: Dict = None) -> Dict:
        """如果会话有 data_manager，则按区域补点后返回新的 spec。"""
        if not context:
            return spec

        data_manager = context.get("data_manager")
        session_id = context.get("session_id")
        if not data_manager or not session_id:
            return spec

        region = self._extract_region(spec)
        if not region:
            return spec

        try:
            current_count = get_spec_data_count(spec)
            new_values = data_manager.load_region(region)
            new_spec = copy.deepcopy(spec)
            new_spec.setdefault("data", {})["values"] = new_values
            app_logger.info(
                f"🔍 Region data loaded: {current_count} -> {len(new_values)} points "
                f"(region: x=[{region.get('x_min')}, {region.get('x_max')}], "
                f"y=[{region.get('y_min')}, {region.get('y_max')}])"
            )
            return new_spec
        except Exception as exc:  # noqa: BLE001
            app_logger.error(f"apply_data_manager failed: {exc}")
            return spec
