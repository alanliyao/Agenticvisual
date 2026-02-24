"""
GPT-5 Benchmark 测试脚本
连接 MCP 服务器，通过 MCP 协议进行工具调用
"""

import json
import os
import sys
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI

# MCP 客户端导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))

from core.vega_service import get_vega_service


# =============================================================================
# 配置
# =============================================================================

GPT_CONFIG = {
    'api_key_env': 'OPENAI_API_KEY',
    'base_url': 'https://api.oaipro.com/v1',
    'model': 'gpt-5',
    'max_iterations': 8,
    'temperature': 0,
    'timeout': 180,
    'save_images': True,
    'max_tokens' : 2000,
}

MCP_SERVER_PATH = Path(__file__).parent / 'chart_tools_mcp_server.py'


# =============================================================================
# MCP 客户端辅助函数
# =============================================================================

def _fix_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    修复 JSON Schema 中常见的不完整定义，以通过 OpenAI/Qwen/Gemini 等校验。
    - array: 必须包含 items
    - object: 建议添加 additionalProperties
    - 去除不支持字段：$ref / nullable
    """
    if not isinstance(schema, dict):
        return schema
    
    # 移除不支持字段
    schema.pop("$ref", None)
    schema.pop("nullable", None)
    
    schema_type = schema.get("type")
    
    # 处理 object
    if schema_type == "object":
        props = schema.get("properties", {})
        for prop_name, prop_def in props.items():
            props[prop_name] = _fix_schema_types_with_name(prop_def, prop_name)
        schema["properties"] = props
        
        # 允许动态键，避免严格校验报错
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = True
    
    # 处理 array
    if schema_type == "array":
        if "items" not in schema:
            schema["items"] = {"type": "string"}
        else:
            schema["items"] = _fix_schema_types(schema["items"])
    
    return schema


def _fix_schema_types_with_name(prop_def: Any, prop_name: str) -> Any:
    """
    根据字段名对 array 的 items 做合理默认推断。
    """
    if not isinstance(prop_def, dict):
        return prop_def
    
    prop_type = prop_def.get("type")
    
    # 数组项推断
    if prop_type == "array" and "items" not in prop_def:
        name_lower = prop_name.lower()
        if any(k in name_lower for k in ["range", "position", "coord", "point", "area", "bbox"]):
            prop_def["items"] = {"type": "number"}
        elif any(k in name_lower for k in ["id", "name", "label", "category", "field"]):
            prop_def["items"] = {"type": "string"}
        else:
            prop_def["items"] = {"type": "string"}
    
    # 递归修复子 schema
    return _fix_schema_types(prop_def)


def convert_mcp_tools_to_openai_format(mcp_tools) -> List[Dict[str, Any]]:
    """
    将 MCP 工具定义转换为 OpenAI Function Calling 格式，并做 Schema 标准化。
    """
    openai_tools = []
    
    for tool in mcp_tools:
        parameters = tool.inputSchema if tool.inputSchema else {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # 从参数中移除 vega_spec（模型不需要知道这个参数），并修复 Schema
        params = _fix_schema_types(parameters)
        if "properties" in params and "vega_spec" in params["properties"]:
            del params["properties"]["vega_spec"]
        if "required" in params and "vega_spec" in params["required"]:
            params["required"].remove("vega_spec")
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": params
            }
        }
        
        openai_tools.append(openai_tool)
    
    return openai_tools


# =============================================================================
# 系统提示词
# =============================================================================

def get_system_prompt(chart_type: str) -> str:
    """Generate system prompt for the given chart type."""
    
    return f"""You are a professional data visualization analysis assistant. Your task is to analyze chart data based on user questions and discover valuable insights.

Current chart type: **{chart_type}**

## Analysis Strategy
1. Carefully read the user question and understand the task
2. Use the provided tools for analysis (if a specific tool is mentioned in the question, you must use that tool)
3. Answer the user question based on tool results

## Tool Selection Guidelines
- **scatter_plot**: select_region, calculate_correlation, identify_clusters, zoom_dense_area, brush_region, change_encoding,filter_categorical,show_regression,
- **bar_chart**: sort_bars, highlight_top_n,filter_categories,expand_stack,toggle_stack_mode,add_bars,remove_bars,add_bar_items,remove_bar_items,change_encoding
- **line_chart**: zoom_time_range, detect_anomalies,highlight_trend,bold_lines,filter_lines,show_moving_average,focus_lines,drilldown_line_time,reset_line_drilldown,resample_time,reset_resample,change_encoding
- **heatmap**: adjust_color_scale,filter_cells,highlight_region,cluster_rows_cols,select_submatrix,find_extremes,threshold_mask,drilldown_time,reset_drilldown,add_marginal_bars,transpose,change_encoding
- **parallel_coordinates**: filter_dimension, highlight_cluster, reorder_dimensions,filter_by_category,highlight_category,hide_dimensions,reset_hidden_dimensions
- **sankey_diagram**: trace_node, highlight_path, filter_flow,calculate_conversion_rate,collapse_nodes,expand_node,auto_collapse_by_rank,color_flows,find_bottleneck,reorder_nodes_in_layer

## Answer Format Specification
Numeric Questions
When to use: Questions asking "how many", "what is the value", "count", "coefficient", "percentage"
Format: Single number only
Example: "How many cars are there in the dataset?" -> "100"

Categorical Questions
When to use: Questions asking "which", "what category", "what type", "which country/region"
Format: Single word/phrase only
Example: "Which country has the highest horsepower?" -> "United States"


Boolean Questions
When to use: Questions asking "is there", "does it", "are they", "yes/no question"
Format: `Yes` or `No` only
Example: "Are there any cars with horsepower greater than 200?" -> "Yes"

Open-ended Questions
When to use: Questions asking about vague exploration of the data
Format: freely answer the question with sentences
Example: "Reveal subtle differences in temperature patterns across cities and months
" -> "Denver’s June temperature (around 22°C) is now visibly higher than its January temperature (around 8°C).\nMiami’s temperatures are consistently high across all months, with its lowest monthly temperature still being warmer than the highest temperatures in Denver or Seattle."

## Output Requirements
- After completing tool calls, provide a clear answer
- If the question requires a specific tool, ensure that tool is called
- Answers should be direct and concise"""




def get_analysis_prompt() -> str:
    """Generate analysis phase prompt."""
    return """Based on the current view and tool results, please return your analysis in JSON format:

```json
{
  "key_insights": ["Insight 1", "Insight 2"],
  "reasoning": "Your reasoning process",
  "answer": "Direct answer to the user question",
  "exploration_complete": true
}
```

Field descriptions:
- **key_insights**: Discovered insights
- **reasoning**: Reasoning process
- **answer**: Direct answer to the user question (required)
- **exploration_complete**: Whether exploration is complete (usually true)

Please ensure you return valid JSON."""

# =============================================================================
# 辅助函数
# =============================================================================

def load_vega_spec(vega_spec_path: str) -> dict:
    """加载 Vega 规范文件"""
    if not os.path.isabs(vega_spec_path):
        script_dir = Path(__file__).parent
        vega_spec_path = script_dir / vega_spec_path
    
    with open(vega_spec_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_openai_client() -> OpenAI:
    """创建 OpenAI 客户端"""
    api_key = os.getenv(GPT_CONFIG['api_key_env'])
    if not api_key:
        raise ValueError(f"请设置环境变量: {GPT_CONFIG['api_key_env']}")
    
    return OpenAI(
        api_key=api_key,
        base_url=GPT_CONFIG['base_url'],
        timeout=GPT_CONFIG['timeout']
    )


def format_user_message_with_image(text: str, image_base64: str) -> dict:
    """格式化带图像的用户消息"""
    return {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': text},
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{image_base64}',
                    'detail': 'high'
                }
            }
        ]
    }


def save_image(image_base64: str, output_path: Path) -> None:
    """保存 base64 图像到文件"""
    image_data = base64.b64decode(image_base64)
    with open(output_path, 'wb') as f:
        f.write(image_data)


def parse_json_from_response(content: str) -> dict:
    """从 GPT 响应中解析 JSON"""
    if not content:
        return {}
    
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 ```json ... ``` 块
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # 尝试提取 ``` ... ``` 块
    if "```" in content:
        try:
            json_str = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    return {}


# =============================================================================
# 主要执行逻辑（异步）
# =============================================================================

async def run_benchmark_async(task_path: str) -> dict:
    """运行 benchmark 测试"""
    
    # 1. 加载任务
    with open(task_path, 'r', encoding='utf-8') as f:
        task = json.load(f)
    
    task_id = task['task_id']
    vega_spec_path = task['task']['initial_visualization']['vega_spec_path']
    vega_spec = load_vega_spec(vega_spec_path)
    query = task['task']['query']
    chart_type = task['metadata'].get('chart_type', 'scatter_plot')
    
    print(f" 任务: {task_id}")
    print(f" 图表类型: {chart_type}")
    print(f" 查询: {query}\n")
    
    # 2. 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = GPT_CONFIG['model'].replace('.', '_').replace('/', '_')
    output_base_dir = Path('benchmark/results/gpt5_mcp')
    run_dir = output_base_dir / f"{task_id}_{model_name}_{timestamp}"
    images_dir = run_dir / 'images'
    
    if GPT_CONFIG['save_images']:
        images_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 初始化 OpenAI 客户端和 Vega 服务
    openai_client = get_openai_client()
    vega_service = get_vega_service()
    
    # 4. 渲染初始图像
    render_result = vega_service.render(vega_spec)
    if not render_result['success']:
        print(f" 渲染失败: {render_result.get('error')}")
        return None
    
    current_image = render_result['image_base64']
    current_spec = vega_spec
    
    if GPT_CONFIG['save_images']:
        save_image(current_image, images_dir / 'iteration_0_initial.png')
    
    # 5. 连接 MCP 服务器
    print(" 连接 MCP 服务器...")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            # 初始化 MCP 会话
            await mcp_session.initialize()
            print(" MCP 服务器连接成功")
            
            # 6. 从 MCP 获取工具列表
            mcp_tools_response = await mcp_session.list_tools()
            mcp_tools = mcp_tools_response.tools
            print(f"🔧 从 MCP 获取到 {len(mcp_tools)} 个工具")
            
            # 转换为 OpenAI 格式
            openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)
            
            # 7. 初始化对话
            system_prompt = get_system_prompt(chart_type)
            messages = [
                {'role': 'system', 'content': system_prompt},
                format_user_message_with_image(f"请探索这个图表并发现洞察。\n\n用户查询：{query}", current_image)
            ]
            
            explorations = []
            all_tools_called = []
            
            print(" 开始探索分析...")
            print("=" * 70)
            
            # 8. 多轮对话循环（两阶段 MCP）
            for i in range(GPT_CONFIG['max_iterations']):
                print(f"\n{'='*20} 第 {i+1} 轮 {'='*20}")
                
                # ========== 阶段1：工具调用（Function Calling） ==========
                print("\n 阶段1：工具调用...")
                
                response1 = openai_client.chat.completions.create(
                    model=GPT_CONFIG['model'],
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=GPT_CONFIG['temperature'],
                )
                
                message1 = response1.choices[0].message
                
                # 将助手消息添加到历史
                assistant_msg = {'role': 'assistant', 'content': message1.content}
                if message1.tool_calls:
                    assistant_msg['tool_calls'] = [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        }
                        for tc in message1.tool_calls
                    ]
                messages.append(assistant_msg)
                
                # 构建 exploration 记录
                exploration = {
                    'iteration': i + 1,
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_summary': {
                        'key_insights': [],
                        'reasoning': ''
                    },
                    'tool_execution': None
                }
                
                # 处理工具调用（通过 MCP）
                if message1.tool_calls:
                    for tool_call in message1.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"\n 通过 MCP 调用工具: {tool_name}")
                        print(f"   参数: {json.dumps(tool_args, ensure_ascii=False)}")
                        
                        # ========== 真正的 MCP 工具调用 ==========
                        # 添加 vega_spec 参数
                        mcp_args = {**tool_args, 'vega_spec': current_spec}
                        
                        mcp_result = await mcp_session.call_tool(
                            name=tool_name,
                            arguments=mcp_args
                        )
                        
                        # 调试：打印 MCP 返回的原始内容，便于确认类型/结构
                        print(f"   ↩️ MCP 原始返回: {mcp_result.content}")
                        
                        # 解析 MCP 返回结果
                        tool_result = {}
                        if mcp_result.content:
                            for content_item in mcp_result.content:
                                if content_item.type == 'text':
                                    try:
                                        tool_result = json.loads(content_item.text)
                                    except json.JSONDecodeError:
                                        tool_result = {'success': False, 'message': content_item.text}
                        
                        all_tools_called.append(tool_name)
                        
                        exploration['tool_execution'] = {
                            'tool_name': tool_name,
                            'parameters': tool_args,
                            'result': {
                                'success': tool_result.get('success', False),
                                'message': tool_result.get('message', '')
                            }
                        }
                        
                        # 构建工具结果消息
                        tool_response_content = json.dumps({
                            'success': tool_result.get('success', False),
                            'message': tool_result.get('message', ''),
                            'data': tool_result.get('cluster_statistics') or tool_result.get('correlation') or tool_result.get('summary') or {}
                        }, ensure_ascii=False)
                        
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': tool_response_content
                        })
                        
                        # 更新视图
                        if tool_result.get('success') and 'vega_spec' in tool_result:
                            current_spec = tool_result['vega_spec']
                            render_result = vega_service.render(current_spec)
                            
                            if render_result.get('success'):
                                current_image = render_result['image_base64']
                                print(f"    MCP 调用成功，视图已更新")
                                
                                if GPT_CONFIG['save_images']:
                                    save_image(current_image, images_dir / f'iteration_{i+1}_{tool_name}.png')
                            else:
                                print(f"    渲染失败")
                        elif tool_result.get('success'):
                            print(f"    MCP 调用成功（分析工具，无视图更新）")
                        else:
                            print(f"    MCP 调用失败: {tool_result.get('error', '未知错误')}")
                            exploration['success'] = False
                else:
                    print("   （无工具调用）")
                
                # ========== 阶段2：分析总结（JSON 输出） ==========
                print("\n 阶段2：分析总结...")
                
                messages.append(format_user_message_with_image(
                    get_analysis_prompt(),
                    current_image
                ))
                
                response2 = openai_client.chat.completions.create(
                    model=GPT_CONFIG['model'],
                    messages=messages,
                    temperature=GPT_CONFIG['temperature'],
                    response_format={"type": "json_object"}
                )
                
                message2 = response2.choices[0].message
                content2 = message2.content or ""
                
                print(f"\n GPT 分析输出:")
                print("-" * 50)
                print(content2[:500] + "..." if len(content2) > 500 else content2)
                print("-" * 50)
                
                # 解析 JSON
                parsed = parse_json_from_response(content2)
                
                if parsed:
                    key_insights = parsed.get('key_insights', [])
                    reasoning = parsed.get('reasoning', '')
                    exploration_complete = parsed.get('exploration_complete', False)
                    
                    exploration['analysis_summary']['key_insights'] = key_insights
                    exploration['analysis_summary']['reasoning'] = reasoning
                    
                    print(f"\n 解析结果:")
                    print(f"  - key_insights: {len(key_insights)} 条")
                    print(f"  - exploration_complete: {exploration_complete}")
                else:
                    print(" 无法解析 JSON")
                    exploration_complete = False
                
                messages.append({'role': 'assistant', 'content': content2})
                explorations.append(exploration)
                
                if parsed and exploration_complete:
                    print(f"\n 探索完成，共 {i + 1} 轮")
                    break
    
    # 9. 汇总所有洞察
    all_insights = []
    for exp in explorations:
        all_insights.extend(exp.get('analysis_summary', {}).get('key_insights', []))
    
    # 10. 构建最终结果
    result = {
        'task_id': task_id,
        'model': GPT_CONFIG['model'],
        'chart_type': chart_type,
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'mode': 'gpt5_real_mcp_benchmark',
        'total_iterations': len(explorations),
        'explorations': explorations,
        'summary': {
            'all_insights': all_insights,
            'tools_called': all_tools_called
        }
    }
    
    # 11. 保存结果
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n 结果已保存: {result_path}")
    
    return result


def run_benchmark(task_path: str) -> dict:
    """运行 benchmark 测试（同步包装器）"""
    return asyncio.run(run_benchmark_async(task_path))


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT Benchmark 测试')
    parser.add_argument('task_path', help='Benchmark 任务 JSON 文件路径')
    parser.add_argument('--model', choices=['gpt-5', 'gpt-5', 'gpt-4o-mini'], default='gpt-5')
    parser.add_argument('--max-iterations', type=int, default=8)
    parser.add_argument('--base-url', default='https://api.oaipro.com/v1')
    parser.add_argument('--no-save-images', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=2000)
    
    args = parser.parse_args()
    
    GPT_CONFIG['model'] = args.model
    GPT_CONFIG['max_iterations'] = args.max_iterations
    GPT_CONFIG['base_url'] = args.base_url
    GPT_CONFIG['save_images'] = not args.no_save_images
    
    print("=" * 70)
    print(" GPT Benchmark 测试")
    print("=" * 70)
    print(f" 模型: {GPT_CONFIG['model']}")
    print(f" 任务: {args.task_path}")
    print(f" 最大轮数: {GPT_CONFIG['max_iterations']}")
    print(f" MCP 服务器: {MCP_SERVER_PATH}")
    print("=" * 70)
    
    result = run_benchmark(args.task_path)
    
    if result:
        print("\n" + "=" * 70)
        print(" 测试完成！")
        print("=" * 70)
        print(f" 总轮数: {result['total_iterations']}")
        print(f" 工具调用: {result['summary']['tools_called']}")
        print(f" 洞察数量: {len(result['summary']['all_insights'])}")
        
        if result['summary']['all_insights']:
            print(f"\n 发现的洞察:")
            for idx, insight in enumerate(result['summary']['all_insights'][:5], 1):
                print(f"   {idx}. {insight}")
            if len(result['summary']['all_insights']) > 5:
                print(f"   ... 还有 {len(result['summary']['all_insights']) - 5} 条")
        
        print("=" * 70)
    else:
        print("\n 测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()

