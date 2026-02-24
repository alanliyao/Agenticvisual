"""
Claude Benchmark Script (New Format Support)

Calls Claude model via OAI Pro's OpenAI-compatible API and connects to MCP server for tool execution.

Supports new multi-question task format:
{
  "chart_id": "cars_scatter_001",
  "vega_spec_path": "benchmark/data/cars_performance_efficiency.json",
  "questions": [{"qid": "cs_01", "question": "...", "ground_truth": {...}}]
}
"""

import copy
import json
import os
import sys
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI

# MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))

from core.vega_service import get_vega_service


# =============================================================================
# Configuration (Using OAI Pro's OpenAI-compatible API for Claude)
# =============================================================================

CLAUDE_CONFIG = {
    'api_key_env': 'OPENROUTER_API_KEY',
    'base_url': 'https://api.oaipro.com/v1',
    'model': 'anthropic/claude-3-5-sonnet-20241022',
    'max_iterations': 8,
    'temperature': 0,
    'timeout': 180,
    'save_images': True,
}

MCP_SERVER_PATH = Path(__file__).parent / 'chart_tools_mcp_server.py'


# =============================================================================
# MCP Client Helper Functions
# =============================================================================

def _fix_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix common incomplete definitions in JSON Schema to pass OpenAI/Qwen/Gemini validation.
    """
    if not isinstance(schema, dict):
        return schema
    
    schema.pop("$ref", None)
    schema.pop("nullable", None)
    
    schema_type = schema.get("type")
    
    if schema_type == "object":
        props = schema.get("properties", {})
        for prop_name, prop_def in props.items():
            props[prop_name] = _fix_schema_types_with_name(prop_def, prop_name)
        schema["properties"] = props
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = True
    
    if schema_type == "array":
        if "items" not in schema:
            schema["items"] = {"type": "string"}
        else:
            schema["items"] = _fix_schema_types(schema["items"])
    
    return schema


def _fix_schema_types_with_name(prop_def: Any, prop_name: str) -> Any:
    """Infer reasonable default items type for arrays based on property name."""
    if not isinstance(prop_def, dict):
        return prop_def
    
    prop_type = prop_def.get("type")
    
    if prop_type == "array" and "items" not in prop_def:
        name_lower = prop_name.lower()
        if any(k in name_lower for k in ["range", "position", "coord", "point", "area", "bbox"]):
            prop_def["items"] = {"type": "number"}
        elif any(k in name_lower for k in ["id", "name", "label", "category", "field"]):
            prop_def["items"] = {"type": "string"}
        else:
            prop_def["items"] = {"type": "string"}
    
    return _fix_schema_types(prop_def)


def convert_mcp_tools_to_openai_format(mcp_tools) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions to OpenAI Function Calling format."""
    openai_tools = []
    
    for tool in mcp_tools:
        parameters = tool.inputSchema if tool.inputSchema else {
            "type": "object",
            "properties": {},
            "required": []
        }
        
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
# System Prompts
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
# Helper Functions
# =============================================================================

def load_vega_spec(vega_spec_path: str) -> dict:
    """Load Vega specification file."""
    if not os.path.isabs(vega_spec_path):
        script_dir = Path(__file__).parent
        vega_spec_path = script_dir / vega_spec_path
    
    with open(vega_spec_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_openai_client() -> OpenAI:
    """Create OpenAI client."""
    api_key = os.getenv(CLAUDE_CONFIG['api_key_env'])
    if not api_key:
        raise ValueError(f"Please set environment variable: {CLAUDE_CONFIG['api_key_env']}")
    
    return OpenAI(
        api_key=api_key,
        base_url=CLAUDE_CONFIG['base_url'],
        timeout=CLAUDE_CONFIG['timeout']
    )


def format_user_message_with_image(text: str, image_base64: str) -> dict:
    """Format user message with image."""
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


def save_image(image_base64: str, output_path: Path) -> str:
    """Save base64 image to file and return the path."""
    image_data = base64.b64decode(image_base64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    return str(output_path)


def strip_data_values(spec: dict) -> dict:
    """Remove data.values from spec to reduce file size."""
    if not spec:
        return spec
    result = spec.copy()
    if "data" in result and isinstance(result["data"], dict):
        if "values" in result["data"]:
            count = len(result["data"]["values"]) if isinstance(result["data"]["values"], list) else "?"
            result["data"] = {"_values_omitted": f"{count} items"}
    return result


def parse_json_from_response(content: str) -> dict:
    """Parse JSON from response content."""
    if not content:
        return {}
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    if "```" in content:
        try:
            json_str = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    return {}


# =============================================================================
# Core: Single Question MCP Dialog Logic
# =============================================================================

async def run_single_question_with_mcp(
    mcp_session,
    openai_client: OpenAI,
    openai_tools: List[Dict],
    question: str,
    vega_spec: Dict,
    chart_type: str,
    vega_service,
    images_dir: Optional[Path],
    qid: str
) -> Dict:
    """
    Run MCP multi-turn dialog for a single question.
    
    Returns:
        {
            "qid": str,
            "question": str,
            "success": bool,
            "answer": str,
            "tool_calls": List[Dict],
            "final_spec": Dict,
            "explorations": List[Dict],
            "final_view_path": str
        }
    """
    # Render initial image
    render_result = vega_service.render(vega_spec)
    if not render_result.get('success'):
        return {
            "qid": qid,
            "question": question,
            "success": False,
            "error": f"Render failed: {render_result.get('error')}",
            "answer": "",
            "tool_calls": [],
            "final_spec": strip_data_values(vega_spec),
            "explorations": []
        }
    
    current_image = render_result['image_base64']
    current_spec = copy.deepcopy(vega_spec)
    
    # Initialize conversation
    system_prompt = get_system_prompt(chart_type)
    messages = [
        {'role': 'system', 'content': system_prompt},
        format_user_message_with_image(f"Please answer the following question:\n\n{question}", current_image)
    ]
    
    explorations = []
    all_tool_calls = []
    final_answer = ""
    
    # Multi-turn dialog loop
    for i in range(CLAUDE_CONFIG['max_iterations']):
        print(f"    [{qid}] Round {i+1}...")
        
        # Phase 1: Tool calling
        response1 = openai_client.chat.completions.create(
            model=CLAUDE_CONFIG['model'],
            messages=messages,
            tools=openai_tools,
            tool_choice={"type": "required"},
            temperature=CLAUDE_CONFIG['temperature'],
        )
        
        message1 = response1.choices[0].message
        content1 = (message1.content or "").rstrip()
        
        assistant_msg = {'role': 'assistant', 'content': content1}
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
        
        exploration = {
            'iteration': i + 1,
            'success': True,
            'tool_execution': None,
            'analysis_summary': {'key_insights': [], 'reasoning': ''}
        }
        
        # Process tool calls
        if message1.tool_calls:
            for tool_call in message1.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"      Tool: {tool_name}")
                
                # MCP tool call
                mcp_args = {**tool_args, 'vega_spec': current_spec}
                mcp_result = await mcp_session.call_tool(
                    name=tool_name,
                    arguments=mcp_args
                )
                
                # Parse result
                tool_result = {}
                if mcp_result.content:
                    for content_item in mcp_result.content:
                        if content_item.type == 'text':
                            try:
                                tool_result = json.loads(content_item.text)
                            except json.JSONDecodeError:
                                tool_result = {'success': False, 'message': content_item.text}
                
                all_tool_calls.append({
                    'name': tool_name,
                    'params': tool_args,
                    'result': {
                        'success': tool_result.get('success', False),
                        'message': tool_result.get('message', '')
                    }
                })
                
                exploration['tool_execution'] = {
                    'tool_name': tool_name,
                    'parameters': tool_args,
                    'result': {
                        'success': tool_result.get('success', False),
                        'message': tool_result.get('message', '')
                    }
                }
                
                # Tool result message
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
                
                # Update view
                if tool_result.get('success') and 'vega_spec' in tool_result:
                    current_spec = tool_result['vega_spec']
                    render_result = vega_service.render(current_spec)
                    if render_result.get('success'):
                        current_image = render_result['image_base64']
                        print(f"      View updated")
        
        # Phase 2: Analysis summary
        messages.append(format_user_message_with_image(get_analysis_prompt(), current_image))
        
        response2 = openai_client.chat.completions.create(
            model=CLAUDE_CONFIG['model'],
            messages=messages,
            temperature=CLAUDE_CONFIG['temperature'],
            response_format={"type": "json_object"}
        )
        
        message2 = response2.choices[0].message
        content2 = (message2.content or "").strip()
        
        parsed = parse_json_from_response(content2)
        
        if parsed:
            exploration['analysis_summary']['key_insights'] = parsed.get('key_insights', [])
            exploration['analysis_summary']['reasoning'] = parsed.get('reasoning', '')
            final_answer = parsed.get('answer', '')
            exploration_complete = parsed.get('exploration_complete', False)
        else:
            exploration_complete = False
        
        messages.append({'role': 'assistant', 'content': content2})
        explorations.append(exploration)
        
        if exploration_complete:
            print(f"    [{qid}] Done, {i + 1} rounds")
            break
    
    # If no explicit answer, merge all insights
    if not final_answer:
        all_insights = []
        for exp in explorations:
            all_insights.extend(exp.get('analysis_summary', {}).get('key_insights', []))
        final_answer = "\n".join(all_insights) if all_insights else ""
    
    # Save final view
    final_view_path = ""
    if images_dir and CLAUDE_CONFIG['save_images']:
        view_path = images_dir / f"{qid}_final.png"
        final_view_path = save_image(current_image, view_path)
    
    return {
        "qid": qid,
        "question": question,
        "success": True,
        "answer": final_answer,
        "tool_calls": all_tool_calls,
        "final_spec": strip_data_values(current_spec),
        "explorations": explorations,
        "final_view_path": final_view_path,
        "reasoning": explorations[-1].get('analysis_summary', {}).get('reasoning', '') if explorations else ""
    }


# =============================================================================
# Main Execution Logic
# =============================================================================

async def run_benchmark_async(task_path: str) -> dict:
    """Run benchmark test (new format multi-question task)."""
    
    # 1. Load task
    with open(task_path, 'r', encoding='utf-8') as f:
        task = json.load(f)
    
    # Parse new format task
    chart_id = task.get('chart_id', task.get('task_id', 'unknown'))
    task_type = task.get('task_type', 'clear_single')
    vega_spec_path = task.get('vega_spec_path')
    questions = task.get('questions', [])
    
    if not vega_spec_path:
        print(f"Error: Task file missing vega_spec_path field")
        return None
    
    if not questions:
        print(f"Error: Task file missing questions field")
        return None
    
    vega_spec = load_vega_spec(vega_spec_path)
    chart_type = task.get('metadata', {}).get('chart_type', 'scatter_plot')
    
    print(f"Task: {chart_id}")
    print(f"Type: {task_type}")
    print(f"Questions: {len(questions)}")
    print()
    
    # 2. Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = CLAUDE_CONFIG['model'].replace('.', '_').replace('/', '_')
    output_base_dir = Path('benchmark/results/claude_mcp')
    run_dir = output_base_dir / f"{chart_id}_{model_name}_{timestamp}"
    images_dir = run_dir / 'images'
    
    if CLAUDE_CONFIG['save_images']:
        images_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Initialize clients and services
    openai_client = get_openai_client()
    vega_service = get_vega_service()
    
    # 4. Connect to MCP server
    print("Connecting to MCP server...")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)]
    )
    
    results = []
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            print("MCP server connected")
            
            # Get tool list
            mcp_tools_response = await mcp_session.list_tools()
            mcp_tools = mcp_tools_response.tools
            openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)
            print(f"Retrieved {len(openai_tools)} tools")
            print()
            
            # 5. Run MCP dialog for each question
            print("=" * 60)
            for idx, q in enumerate(questions):
                qid = q.get('qid', f'q_{idx}')
                question_text = q.get('question', '')
                
                print(f"\n[{idx+1}/{len(questions)}] {qid}: {question_text[:50]}...")
                
                # Reset vega_spec for each question
                result = await run_single_question_with_mcp(
                    mcp_session=mcp_session,
                    openai_client=openai_client,
                    openai_tools=openai_tools,
                    question=question_text,
                    vega_spec=copy.deepcopy(vega_spec),  # Reset
                    chart_type=chart_type,
                    vega_service=vega_service,
                    images_dir=images_dir,
                    qid=qid
                )
                
                results.append(result)
                
                if result.get('success'):
                    answer_preview = result.get('answer', '')[:80]
                    tools_used = [tc['name'] for tc in result.get('tool_calls', [])]
                    print(f"    Answer: {answer_preview}...")
                    print(f"    Tools: {tools_used}")
                else:
                    print(f"    Failed: {result.get('error', 'Unknown')}")
            
            print("\n" + "=" * 60)
    
    # 6. Build final result (evaluation framework compatible format)
    final_result = {
        "task_id": chart_id,
        "task_type": task_type,
        "model": CLAUDE_CONFIG['model'],
        "model_name": f"Claude ({CLAUDE_CONFIG['model']})",
        "timestamp": datetime.now().isoformat(),
        "questions_count": len(questions),
        "success_count": sum(1 for r in results if r.get('success')),
        "results": results
    }
    
    # 7. Save results
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved: {result_path}")
    print(f"Success: {final_result['success_count']}/{len(questions)}")
    
    return final_result


def run_benchmark(task_path: str) -> dict:
    """Run benchmark test (sync wrapper)."""
    return asyncio.run(run_benchmark_async(task_path))


# =============================================================================
# Command Line Entry
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Benchmark Test (New Format)')
    parser.add_argument('task_path', help='Path to benchmark task JSON file')
    parser.add_argument('--model', default='anthropic/claude-3-opus-20240229',
                        help='Claude model name')
    parser.add_argument('--max-iterations', type=int, default=8)
    parser.add_argument('--base-url', default='https://openrouter.ai/api/v1',
                        help='OpenAI-compatible endpoint')
    parser.add_argument('--no-save-images', action='store_true')
    
    args = parser.parse_args()
    
    CLAUDE_CONFIG['model'] = args.model
    CLAUDE_CONFIG['max_iterations'] = args.max_iterations
    CLAUDE_CONFIG['base_url'] = args.base_url
    CLAUDE_CONFIG['save_images'] = not args.no_save_images
    
    print("=" * 60)
    print("Claude Benchmark Test (New Format)")
    print("=" * 60)
    print(f"Model: {CLAUDE_CONFIG['model']}")
    print(f"Task: {args.task_path}")
    print(f"Max iterations: {CLAUDE_CONFIG['max_iterations']}")
    print(f"MCP server: {MCP_SERVER_PATH}")
    print("=" * 60)
    
    result = run_benchmark(args.task_path)
    
    if result:
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
    else:
        print("\nTest failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
