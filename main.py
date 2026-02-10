"""
主程序入口
提供简单的命令行交互接口
"""

import base64
import json
import sys
from pathlib import Path
from datetime import datetime

from config import validate_config
from core import get_session_manager, get_vega_service
from core.utils import app_logger


def save_exploration_result(result: dict, session_id: str):
    """保存探索结果到文件"""
    try:
        # 创建results目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        mode = result.get("mode", "autonomous_exploration")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exploration_{timestamp}_{mode}.json"
        filepath = results_dir / filename
        records = result.get("explorations") if mode == "autonomous_exploration" else result.get("iterations", [])
        image_dir = results_dir / f"images_{timestamp}_{mode}"
        image_dir.mkdir(exist_ok=True)
        
        # 处理每轮图像
        for exp in records:
            view_files = []
            for idx, image_b64 in enumerate(exp.get("images", []), start=1):
                try:
                    image_bytes = base64.b64decode(image_b64.split(",")[-1])
                    iter_num = exp.get("iteration", 0)
                    view_filename = image_dir / f"exploration_{session_id[:8]}_iter{iter_num}_view{idx}.png"
                    with open(view_filename, "wb") as img_f:
                        img_f.write(image_bytes)
                    view_files.append(str(view_filename))
                except Exception as exc:
                    app_logger.error(f"failed to save view: {exc}", exc_info=True)
            if view_files:
                exp["view_files"] = view_files
            # 移除images字段（base64字符串），只保留view_files（文件路径）
            if "images" in exp:
                del exp["images"]
        
        # 准备保存的数据
        save_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "mode": result.get("mode"),
            "total_iterations": result.get("total_iterations"),
            "explorations": records,
            "final_report": result.get("final_report")
        }
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"system>  exploration results saved to: {filepath}\n")
        
        # 生成人类可读的文本报告
        txt_filepath = filepath.with_suffix('.txt')
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            for exp in records:
                iter_num = exp.get('iteration', 0)
                f.write(f"=== iteration {iter_num} ===\n\n")
                
                analysis = exp.get('analysis_summary', {})
                
                # 关键洞察
                insights = analysis.get('key_insights', [])
                if insights:
                    f.write("key insights:\n")
                    for idx, insight in enumerate(insights, 1):
                        f.write(f"{idx}. {insight}\n")
                    f.write("\n")
                
                # 推理过程
                reasoning = analysis.get('reasoning', '')
                if reasoning:
                    f.write("reasoning process:\n")
                    # 兼容列表和字符串两种格式
                    if isinstance(reasoning, list):
                        reasoning_text = "\n".join(reasoning)
                    else:
                        reasoning_text = reasoning
                    f.write(f"{reasoning_text}\n")
                    f.write("\n")
                
                # 使用的工具
                tool_exec = exp.get('tool_execution')
                if tool_exec:
                    tool_name = tool_exec.get('tool_name', '')
                    if tool_name:
                        f.write(f"使用工具：{tool_name}\n\n")
                
                f.write("\n")
        
        print(f"system>  text report saved to: {txt_filepath}\n")
    except Exception as e:
        app_logger.error(f"failed to save exploration result: {e}", exc_info=True)
        print(f"system> failed to save exploration result: {e}\n")


def main():
    """主函数"""
    print("=" * 60)
    print(" Visual Analysis System")
    print("=" * 60)
    
    # 验证配置
    errors = validate_config()
    if errors:
        print("\n config errors:")
        for error in errors:
            print(f"  - {error}")
        print("\n please set DASHSCOPE_API_KEY in environment variables or .env file")
        return
    
    print("\n config validation passed")
    
    # 初始化会话管理器
    session_mgr = get_session_manager()
    print(" system initialized\n")
    
    
    # 示例：加载Vega-Lite规范
    print("Please provide the Vega-Lite specification file path (or enter 'demo' to use the example):")
    spec_path = input("> ").strip()
    
    if spec_path.lower() == 'demo':
        # 使用示例规范
        vega_spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "category", "type": "nominal"},
                "y": {"field": "value", "type": "quantitative"}
            },
            "data": {
                "values": [
                    {"category": "A", "value": 28},
                    {"category": "B", "value": 55},
                    {"category": "C", "value": 43}
                ]
            }
        }
    else:
        try:
            with open(spec_path, 'r') as f:
                vega_spec = json.load(f)
        except Exception as e:
            print(f" failed to load file: {e}")
            return
    
    # 创建会话
    print("\n creating session...")
    session_id = session_mgr.create_session(vega_spec)
    
    if not session_id:
        print(" session creation failed")
        return
    
    print(f" session created successfully: {session_id}\n")
    
    
    # 交互循环
    print("start conversation (input 'exit' to exit, 'reset' to reset the view, 'save' to save the result): \n")
    
    last_result = None  # 保存最后一次结果
    
    while True:
        user_query = input("user> ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() == 'exit':
            print("\n goodbye!")
            break
        
        if user_query.lower() == 'reset':
            result = session_mgr.reset_view(session_id)
            print(f"system> {result.get('message', 'reset completed')}\n")
            continue
        
        if user_query.lower() == 'save':
            if last_result and last_result.get("mode") in ("autonomous_exploration", "goal_oriented"):
                save_exploration_result(last_result, session_id)
            else:
                print("system> no exploration results to save\n")
            continue
        
        # 处理查询
        print("\n processing...")
        result = session_mgr.process_query(session_id, user_query)
        last_result = result  # 保存结果
        
        if result.get("success"):
            mode = result.get("mode", "unknown")
            print(f"\n[{mode.upper()} mode]")
            
            if mode == "chitchat":
                print(f"system> {result.get('response', '')}\n")
            elif mode == "goal_oriented":
                iterations = result.get("iterations", [])
                print(f"executed {len(iterations)} iterations")
                for it in iterations:
                    print(f"  - iteration {it['iteration']}:")
                    decision = it.get("decision", {})
                    insights = decision.get("key_insights", [])
                    if insights:
                        print(f"      insights:")
                        for insight in insights:
                            print(f"       - {insight}")
                    reasoning = decision.get("reasoning")
                    if reasoning:
                        print(f"      thinking: {reasoning}")
                    view_files = it.get("view_files", [])
                    if view_files:
                        print(f"      view files ({len(view_files)}):")
                        for path in view_files:
                            print(f"       • {path}")
                print()
            elif mode == "autonomous_exploration":
                explorations = result.get("explorations", [])
                report = result.get("final_report", {})
                
                print(f"executed {len(explorations)} explorations\n")
                
                # 显示每轮探索的详细信息
                for exp in explorations:
                    iter_num = exp.get("iteration", 0)
                    success = exp.get("success", False)
                    
                    print(f"【iteration {iter_num}】")
                    
                    if not success:
                        print(f"    failed: {exp.get('error', 'Unknown error')}")
                        print()
                        continue
                    
                    # 显示分析摘要
                    analysis = exp.get("analysis_summary", {})
                    
                    # 关键洞察
                    insights = analysis.get("key_insights", [])
                    if insights:
                        print(f"    insights:")
                        for idx, insight in enumerate(insights[:3], 1):  # 最多显示3个
                            print(f"     {idx}. {insight}")
                    
                    # 建议
                    reasoning = analysis.get("reasoning", "")
                    if reasoning:
                        print(f"    reasoning:")
                        # 兼容列表和字符串两种格式
                        if isinstance(reasoning, list):
                            for idx, reason in enumerate(reasoning[:2], 1):  # 最多显示2个
                                print(f"     {idx}. {reason}")
                        else:
                            print(f"     {reasoning}")
                    
                    # 工具使用
                    tool_exec = exp.get("tool_execution")
                    if tool_exec:
                        tool_name = tool_exec.get("tool_name", "unknown tool")
                        tool_success = tool_exec.get("tool_result", {}).get("success", False)
                        status = "success" if tool_success else "failed"
                        print(f"   tool: {tool_name} {status}")
                        
                        tool_result = tool_exec.get("tool_result", {})
                        if tool_result.get("message"):
                            print(f"     {tool_result['message']}")
                        if tool_result.get("error"):
                            print(f"     error: {tool_result['error']}")
                        details = tool_result.get("details")
                        if details:
                            print("     details:")
                            for detail in details:
                                print(f"       • {detail}")
                    
                    # 耗时
                    duration = exp.get("duration", 0)
                    print(f"  duration: {duration:.2f} seconds")
                    print()
                
                # 显示最终报告
                print(f"【{mode} summary】")
                print(f"  total iterations: {report.get('total_iterations', 0)}")
                print(f"  successful iterations: {report.get('successful_iterations', 0)}")
                print(f"  {report.get('summary', 'exploration completed')}")
                
                
                # 汇总所有洞察
                all_insights = report.get('all_insights', [])
                if all_insights:
                    print(f"\n    all insights ({len(all_insights)} insights):")
                    for idx, insight in enumerate(all_insights[:5], 1):  # 最多显示5条
                        print(f"     {idx}. {insight}")
                    if len(all_insights) > 5:
                        print(f"     ... there are {len(all_insights) - 5} more insights")
                
                # 工具使用统计
                tools_used = report.get('tools_used', [])
                if tools_used:
                    print(f"\n    tool usage statistics:")
                    for tool_info in tools_used:
                        status = "success" if tool_info.get("success") else "failed"
                        print(f"     {status} iteration {tool_info['iteration']}: {tool_info['tool']}")
                
                print()
        else:
            print(f"\n error: {result.get('error', 'Unknown error')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n interrupted exit")
        sys.exit(0)
    except Exception as e:
        app_logger.error(f"program exception: {e}", exc_info=True)
        print(f"\n program exception: {e}")
        sys.exit(1)
