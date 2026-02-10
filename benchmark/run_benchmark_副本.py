#!/usr/bin/env python3
"""
unified Benchmark running script

supports new format tasks (multiple questions), can call any model (including your own system).

usage:
    # run single model
    python -m benchmark.run_benchmark \
      --task benchmark/tasks/objective/clear+single/scatter_cars_001.json \
      --model system

    # run multiple models
    python -m benchmark.run_benchmark \
      --task benchmark/tasks/objective/clear+single/scatter_cars_001.json \
      --models claude,gpt,system

    # run and evaluate
    python -m benchmark.run_benchmark \
      --task benchmark/tasks/objective/clear+single/scatter_cars_001.json \
      --model system \
      --eval
"""

import json
import os
import sys
import copy
import base64
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from openai import OpenAI

# MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import get_model_config, get_api_key, list_available_models, ModelConfig
from core.vega_service import get_vega_service

# MCP server path
MCP_SERVER_PATH = Path(__file__).parent.parent / 'chart_tools_mcp_server.py'


# ============================================================
# Evaluation Format Instruction
# ============================================================

EVALUATION_FORMAT = """
=== OUTPUT FORMAT INSTRUCTIONS ===

For EACH iteration of your analysis, output your reasoning:

REASONING: <Explain your analysis approach, observations from the current view, 
and preparations for the next step in several sentences>

After completing ALL iterations, provide your final summary:

KEY_INSIGHTS:
- <Insight from iteration 1 with specific data>
- <Insight from iteration 2 with specific data>
- ...
- <Final insight from the last iteration>

ANSWER: 
- For Objective Questions: <ONLY one word or short phrase, NO explanation>
  Examples: "Yes", "No", "Japan", "0.85", "3 clusters"
- For Subjective Questions: <Comprehensive answer with clarity and logic>

IMPORTANT: For objective questions, the ANSWER field must contain ONLY the direct answer.
Put any explanation in REASONING or KEY_INSIGHTS, NOT in ANSWER.
"""


# ============================================================
# Utility Functions
# ============================================================

def strip_data_values(spec: Dict) -> Dict:
    """
    Remove data.values from spec to reduce file size while keeping structure for evaluation.
    StateEvaluator only needs encoding, params, transform - not the raw data.
    """
    if not spec:
        return spec
    result = spec.copy()
    if "data" in result and isinstance(result["data"], dict):
        if "values" in result["data"]:
            count = len(result["data"]["values"]) if isinstance(result["data"]["values"], list) else "?"
            result["data"] = {"_values_omitted": f"{count} items"}
    return result


def load_task(task_path: str) -> Dict:
    """load task config"""
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vega_spec(chart_id: str, task_dir: Path) -> Dict:
    """load Vega spec based on chart_id"""
    
    # chart_id to file name mapping
    ID_MAP = {
        "cars_scatter_001": "cars_performance_efficiency",
        "cars_multivariate_002": "cars_performance_efficiency",
        "cars_multiregion_003": "cars_performance_efficiency",
        "scatter_clustering_001": "cars_performance_efficiency",
        # more mappings can be added here
    }
    
    # try mapping
    file_name = ID_MAP.get(chart_id, chart_id)
    
    # try multiple possible paths
    base_paths = [
        task_dir.parent.parent.parent / "data",
        task_dir.parent.parent.parent.parent / "data",
        Path("benchmark/data"),
        Path("data"),
    ]
    
    # if chart_id contains path information
    if "/" in chart_id:
        full_path = Path(chart_id)
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
    
    # try various file names and path combinations
    tried_paths = []
    for base in base_paths:
        for name in [file_name, chart_id]:
            path = base / f"{name}.json"
            tried_paths.append(path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    
    raise FileNotFoundError(f"Vega spec not found for chart_id: {chart_id}. Tried: {tried_paths[:4]}")


def save_view(vega_spec: Dict, output_path: Path) -> str:
    """save view as image"""
    vega_service = get_vega_service()
    result = vega_service.render(vega_spec)
    
    if result.get("success"):
        image_data = result["image_base64"]
        # remove data URL prefix
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        
        return str(output_path)
    
    return ""


def create_client(config: ModelConfig) -> OpenAI:
    """create OpenAI client"""
    api_key = get_api_key(config)
    
    # for local system, use dummy key
    if api_key is None:
        api_key = "dummy-key"
    
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout
    )


def encode_image(image_base64: str) -> str:
    """ensure image is in correct data URL format"""
    if image_base64.startswith("data:"):
        return image_base64
    return f"data:image/png;base64,{image_base64}"


# ============================================================
# MCP Helper Functionse
# ============================================================

def _fix_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common incomplete definitions in JSON Schema for OpenAI validation."""
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


def get_system_prompt(chart_type: str) -> str:
    """Generate system prompt for the given chart type."""
    return f"""You are a professional data visualization analysis assistant. Your task is to analyze chart data based on user questions and discover valuable insights.

Current chart type: **{chart_type}**

## Analysis Strategy
1. Carefully read the user question and understand the task
2. You MUST use the provided tools for analysis - do NOT imagine or conceptually simulate tool operations
3. If a specific tool is mentioned in the question, prioritize using that tool
4. Answer the user question based on tool results

## Tool Selection Guidelines
- **scatter_plot**: select_region, calculate_correlation, identify_clusters, zoom_dense_area, brush_region, change_encoding,filter_categorical,show_regression,
- **bar_chart**: sort_bars, highlight_top_n,filter_categories,expand_stack,toggle_stack_mode,add_bars,remove_bars,add_bar_items,remove_bar_items
- **line_chart**: zoom_time_range, detect_anomalies,highlight_trend,bold_lines,filter_lines,show_moving_average,focus_lines,drilldown_line_time,reset_line_drilldown,resample_time,reset_resample
- **heatmap**: adjust_color_scale,filter_cells,highlight_region,cluster_rows_cols,select_submatrix,find_extremes,threshold_mask,drilldown_time,reset_drilldown,add_marginal_bars,transpose
- **parallel_coordinates**: filter_dimension, highlight_cluster, reorder_dimensions,filter_by_category,highlight_category,hide_dimensions,reset_hidden_dimensions
- **sankey_diagram**: trace_node, highlight_path, filter_flow,calculate_conversion_rate,collapse_nodes,expand_node,auto_collapse_by_rank,color_flows,find_bottleneck,reorder_nodes_in_layer

## Stop criteria
- When the user question is answered, stop the analysis.
- when all the required tools are called, stop the analysis.
- when repeating the same tool call for too many times stop the analysis.
- when gaining the same insights and reasoning as the above iteration, stop the analysis.

## Answer Format Specification
There are two types of questions: objective and subjective.

subjective questions are questions that cannot be answered with a single number, word, or phrase.
You should answer subjective questions with a comprehensive answer with clarity and logic, providing key insights and reasoning.

Objective questions are questions that can be answered with a single number, word, or phrase.
- **Objective Questions**
There are several types of objective questions: numeric, categorical, region/range, boolean, and year.

- **Numeric Questions**
When to use: Questions asking "how many", "what is the value", "count", "coefficient", "percentage"
Format: Single number only

- **Categorical Questions**
When to use: Questions asking "which", "what category", "what type", "which country/region"
Format: Single word/phrase only

- **Region/Range Questions**
When to use: Questions asking "what range", "between what values", "interval"
Format: Format: `[min, max]` or `min-max`
Example: "What is the range of horsepower?" -> "[100, 200]"

- **Boolean Questions**
When to use: Questions asking "is there", "does it", "are they", "yes/no question"
Format: `Yes` or `No` only
Example: "Are there any cars with horsepower greater than 200?" -> "Yes"

- **Year Questions**
When to use: Questions asking "what year", "which year"
Format: Full year: `2023`
Example: "What year is the data from?" -> "2023"

- ## Output Requirements
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


def parse_json_from_response(content: str) -> dict:
    """Parse JSON from response content."""
    if not content:
        return {}
    
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


def save_image_from_base64(image_base64: str, output_path: Path) -> str:
    """Save base64 image to file and return the path."""
    image_data = base64.b64decode(image_base64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    return str(output_path)


# ============================================================
# System API Calling (No External Tools)
# ============================================================

def _extract_message_dict(message):
    """Extract dict from message object using multiple methods"""
    if isinstance(message, dict):
        return message
    if hasattr(message, 'model_dump'):
        return message.model_dump()
    if hasattr(message, '__dict__'):
        return message.__dict__
    # Try attribute access
    result = {}
    for key in ['final_spec', 'tool_calls_history', 'reasoning', 'iterations', 'mode']:
        if hasattr(message, key):
            result[key] = getattr(message, key, None)
    return result


def run_system_api_question(
    client: OpenAI,
    config: ModelConfig,
    question: Dict,
    vega_spec: Dict,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Call the system API which has internal tool logic.
    No MCP tools needed - just send messages directly.
    """
    question_text = question.get("question", "")
    qid = question.get("qid", "unknown")
    
    # Prepare messages for system API
    # System API expects vega_spec in system message
    messages = [
        {"role": "system", "content": json.dumps(vega_spec)},
        {"role": "user", "content": f"{question_text}\n\n{EVALUATION_FORMAT}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        message = response.choices[0].message
        answer = message.content or ""
        
        # Extract fields with defaults
        msg_dict = _extract_message_dict(message)
        final_spec = msg_dict.get('final_spec', vega_spec)
        tool_calls_history = msg_dict.get('tool_calls_history', [])
        reasoning = msg_dict.get('reasoning', "")
        iterations = msg_dict.get('iterations', 1)
        mode = msg_dict.get('mode', "")
        
        # Convert tool calls
        tool_calls = [
            {
                "name": tc.get("name", ""),
                "params": tc.get("params", {}),
                "result": tc.get("result", {})
            }
            for tc in tool_calls_history if isinstance(tc, dict)
        ]
        
        # Build explorations if needed
        explorations = []
        if mode == "goal_oriented" and (tool_calls or reasoning):
            exploration = {
                "iteration": iterations,
                "success": True,
                "tool_execution": {
                    "name": tool_calls[0].get("name", ""),
                    "params": tool_calls[0].get("params", {}),
                    "result": tool_calls[0].get("result", {})
                } if tool_calls else None,
                "analysis_summary": {
                    "key_insights": [],
                    "reasoning": reasoning
                }
            }
            explorations.append(exploration)
        
        return {
            "qid": qid,
            "question": question_text,
            "success": True,
            "answer": answer,
            "model": config.model,
            "tool_calls": tool_calls,
            "final_spec": strip_data_values(final_spec),
            "reasoning": reasoning,
            "iterations": iterations,
            "explorations": explorations
        }
        
    except Exception as e:
        print(f"      System API error: {e}")
        return {
            "qid": qid,
            "question": question_text,
            "success": False,
            "error": str(e),
            "answer": "",
            "tool_calls": [],
            "final_spec": strip_data_values(vega_spec),
            "reasoning": "",
            "iterations": 0,
            "explorations": []
        }


# ============================================================
# Multi-turn MCP Tool Calling
# ============================================================

async def run_multi_turn_with_mcp(
    mcp_session,
    client: OpenAI,
    config: ModelConfig,
    openai_tools: List[Dict],
    question: Dict,
    vega_spec: Dict,
    chart_type: str,
    output_dir: Optional[Path] = None,
    max_iterations: int = 3
) -> Dict:
    """
    Multi-turn MCP tool calling for a single question.
    Returns evaluation-compatible format.
    """
    question_text = question.get("question", "")
    qid = question.get("qid", "unknown")
    
    # Render initial image
    vega_service = get_vega_service()
    render_result = vega_service.render(vega_spec)
    
    if not render_result.get('success'):
        return {
            "qid": qid,
            "question": question_text,
            "success": False,
            "error": f"Render failed: {render_result.get('error')}",
            "answer": "",
            "tool_calls": [],
            "final_spec": strip_data_values(vega_spec),
            "reasoning": "",
            "iterations": 0,
            "explorations": []
        }
    
    current_image = render_result['image_base64']
    current_spec = copy.deepcopy(vega_spec)
    
    # Initialize conversation
    system_prompt = get_system_prompt(chart_type)
    # Include EVALUATION_FORMAT for structured output
    formatted_question = f"Please answer the following question:\n\n{question_text}\n\n{EVALUATION_FORMAT}"
    messages = [
        {'role': 'system', 'content': system_prompt},
        format_user_message_with_image(formatted_question, current_image)
    ]
    
    explorations = []
    all_tool_calls = []
    final_answer = ""
    
    # Multi-turn dialog loop
    for i in range(max_iterations):
        print(f"    [{qid}] Round {i+1}...")
        
        # Phase 1: Tool calling
        # Choose tool_choice format based on model config
        if config.tool_choice_format == "dict":
            tool_choice_value = {"type": "auto"}
        else:
            tool_choice_value = "auto"
        
        try:
            response1 = client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=openai_tools,
                tool_choice=tool_choice_value,
                temperature=config.temperature,
            )
        except Exception as e:
            print(f"      API error: {e}")
            break
        
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
        else:
            # No tool calls, model gave direct answer
            print(f"      No tool call")
        
        # Phase 2: Analysis summary
        messages.append(format_user_message_with_image(get_analysis_prompt(), current_image))
        
        try:
            response2 = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            # Some models don't support json_object, try without it
            try:
                response2 = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                )
            except Exception as e2:
                print(f"      Analysis error: {e2}")
                break
        
        message2 = response2.choices[0].message
        print(f"      Message2: {message2}")
        content2 = (message2.content or "").strip()
        print(f"      Content2: {content2}")
        
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
    if output_dir:
        view_path = output_dir / "images" / f"{qid}_final.png"
        final_view_path = save_image_from_base64(current_image, view_path)
    
    # Get reasoning from last exploration
    reasoning = ""
    if explorations:
        reasoning = explorations[-1].get('analysis_summary', {}).get('reasoning', '')
    
    return {
        "qid": qid,
        "question": question_text,
        "success": True,
        "answer": final_answer,
        "model": config.model,
        "tool_calls": all_tool_calls,
        "final_spec": strip_data_values(current_spec),
        "reasoning": reasoning,
        "iterations": len(explorations),
        "explorations": explorations,
        "final_view_path": final_view_path
    }



# 核心逻辑


def run_single_question(
    client: OpenAI,
    config: ModelConfig,
    question: Dict,
    vega_spec: Dict,
    save_views: bool = True,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    run single question
    
    for external model: send question + image, let the model call tools
    for internal model: call API directly, internal tool processing
    """
    question_text = question.get("question", "")
    qid = question.get("qid", "unknown")
    
    # 渲染初始图片
    vega_service = get_vega_service()
    render_result = vega_service.render(vega_spec)
    
    if not render_result.get("success"):
        return {
            "qid": qid,
            "success": False,
            "error": "Failed to render initial view"
        }
    
    image_base64 = render_result["image_base64"]
    
    # Append format instruction to question
    formatted_question = f"{question_text}\n\n{EVALUATION_FORMAT}"
    
    # Build messages
    if config.name == "My System":
        # Internal system: use special format
        messages = [
            {"role": "system", "content": json.dumps(vega_spec)},
            {"role": "user", "content": formatted_question}
        ]
    else:
        # External model: send image + question with format instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image(image_base64)}
                    },
                    {
                        "type": "text",
                        "text": f"Based on this visualization, please answer: {formatted_question}"
                    }
                ]
            }
        ]
    
    try:
        # call API
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        message = response.choices[0].message
        
        # extract result
        result = {
            "qid": qid,
            "question": question_text,
            "success": True,
            "answer": message.content or "",
            "model": config.model,
        }
        
        # if internal model, extract extra fields
        if hasattr(message, "final_spec"):
            result["final_spec"] = strip_data_values(message.final_spec)
            result["tool_calls"] = getattr(message, "tool_calls_history", [])
            result["reasoning"] = getattr(message, "reasoning", "")
            result["iterations"] = getattr(message, "iterations", 0)
        else:
            # for external model, try to extract from message
            try:
                msg_dict = message.model_dump() if hasattr(message, "model_dump") else {}
                result["final_spec"] = strip_data_values(msg_dict.get("final_spec", vega_spec))
                result["tool_calls"] = msg_dict.get("tool_calls_history", [])
                result["reasoning"] = msg_dict.get("reasoning", "")
                result["iterations"] = msg_dict.get("iterations", 1)
            except:
                result["final_spec"] = strip_data_values(vega_spec)
                result["tool_calls"] = []
                result["reasoning"] = ""
                result["iterations"] = 1
        
        # save final view
        if save_views and output_dir:
            final_spec = result.get("final_spec", vega_spec)
            view_path = output_dir / "images" / f"{qid}_final.png"
            result["final_view_path"] = save_view(final_spec, view_path)
        
        return result
        
    except Exception as e:
        return {
            "qid": qid,
            "question": question_text,
            "success": False,
            "error": str(e)
        }


async def run_task_async(
    task_path: str,
    model_name: str,
    save_views: bool = True,
    output_dir: Optional[str] = None
) -> Dict:
    """
    run full task (all questions) - use MCP multi-turn iterations
    """
    # load task
    task = load_task(task_path)
    task_id = task.get("task_id", task.get("chart_id", Path(task_path).stem))
    task_type = task.get("task_type", "clear_single")
    questions = task.get("questions", [])
    chart_type = task.get("metadata", {}).get("chart_type", "scatter_plot")
    
    if not questions:
        return {"success": False, "error": "No questions found in task"}
    
    # get model config
    config = get_model_config(model_name)
    client = create_client(config)
    
    # load Vega spec
    chart_id = task.get("chart_id", task_id)
    vega_spec = load_vega_spec(chart_id, Path(task_path).parent)
    
    # output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("benchmark/results/new") / f"{task_id}_{model_name}_{timestamp}"
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning task: {task_id}")
    print(f"Model: {config.name} ({model_name})")
    print(f"Questions: {len(questions)}")
    print(f"Max iterations: {config.max_iterations}")
    print("-" * 50)
    
    results = []
    
    # Check if model uses external MCP tools
    if not config.supports_external_tools:
        # System API path - no MCP needed
        print("Using system API (internal tool logic)...")
        print("-" * 50)
        
        for i, question in enumerate(questions):
            qid = question.get("qid", f"q_{i}")
            print(f"\n[{i+1}/{len(questions)}] {qid}: {question.get('question', '')[:50]}...")
            
            result = run_system_api_question(
                client=client,
                config=config,
                question=question,
                vega_spec=copy.deepcopy(vega_spec),
                output_dir=out_path if save_views else None
            )
            results.append(result)
            
            if result.get("success"):
                answer_preview = result.get("answer", "")[:80]
                print(f"    Answer: {answer_preview}...")
            else:
                print(f"    Failed: {result.get('error', 'Unknown')}")
    else:
        # MCP multi-turn path
        print("Connecting to MCP server...")
        server_params = StdioServerParameters(
            command="python",
            args=[str(MCP_SERVER_PATH)]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                print("MCP server connected")
                
                # Get tool list
                mcp_tools_response = await mcp_session.list_tools()
                mcp_tools = mcp_tools_response.tools
                openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)
                print(f"Retrieved {len(openai_tools)} tools")
                print("-" * 50)
                
                # Run each question with multi-turn MCP
                for i, question in enumerate(questions):
                    qid = question.get("qid", f"q_{i}")
                    print(f"\n[{i+1}/{len(questions)}] {qid}: {question.get('question', '')[:50]}...")
                    
                    result = await run_multi_turn_with_mcp(
                        mcp_session=mcp_session,
                        client=client,
                        config=config,
                        openai_tools=openai_tools,
                        question=question,
                        vega_spec=copy.deepcopy(vega_spec),  # Reset for each question
                        chart_type=chart_type,
                        output_dir=out_path if save_views else None,
                        max_iterations=config.max_iterations
                    )
                    results.append(result)
                    
                    if result.get("success"):
                        answer_preview = result.get("answer", "")[:80]
                        tools_used = [tc['name'] for tc in result.get('tool_calls', [])]
                        print(f"    Answer: {answer_preview}...")
                        print(f"    Tools: {tools_used}")
                    else:
                        print(f"    Failed: {result.get('error', 'Unknown')}")
    
    # summarize results
    final_result = {
        "task_id": task_id,
        "task_type": task_type,
        "model": model_name,
        "model_name": config.name,
        "timestamp": datetime.now().isoformat(),
        "questions_count": len(questions),
        "success_count": sum(1 for r in results if r.get("success")),
        "results": results
    }
    
    # save results
    # ensure output directory exists (prevent relative path issues)
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("-" * 50)
    print(f"Results saved to: {result_path}")
    print(f"Success: {final_result['success_count']}/{len(questions)}")
    
    return final_result


def run_task(
    task_path: str,
    model_name: str,
    save_views: bool = True,
    output_dir: Optional[str] = None
) -> Dict:
    """
    run full task (sync wrapper)
    """
    return asyncio.run(run_task_async(task_path, model_name, save_views, output_dir))


# ============================================================
# command line entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner")
    parser.add_argument("--task", help="Path to task JSON file")
    parser.add_argument("--model", help="Model to use (see --list-models)")
    parser.add_argument("--models", help="Comma-separated list of models")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--no-save-views", action="store_true", help="Don't save view images")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after benchmark")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # list models
    if args.list_models:
        print("Available models:")
        for name, display in list_available_models().items():
            print(f"  {name}: {display}")
        return
    
    # validate task path
    if not args.task:
        print("Error: --task is required when running benchmarks")
        parser.print_help()
        return
    
    # determine model to run
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.model:
        models = [args.model]
    else:
        print("Error: --model or --models required")
        parser.print_help()
        return
    
    # run
    all_results = []
    for model in models:
        try:
            result = run_task(
                args.task,
                model,
                save_views=not args.no_save_views,
                output_dir=args.output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error running model {model}: {e}")
            import traceback
            traceback.print_exc()
    
    # evaluate
    if args.eval and all_results:
        print("\n" + "=" * 50)
        print("Running evaluation...")
        from benchmark.evaluators import UnifiedEvaluator
        
        evaluator = UnifiedEvaluator()
        task_config = load_task(args.task)
        
        for result in all_results:
            if result.get("results"):
                print(f"\nModel: {result['model_name']}")
                for i, q_result in enumerate(result["results"]):
                    if q_result.get("success"):
                        # 构建评估输入
                        agent_result = {
                            "answer": q_result.get("answer"),
                            "tool_calls": q_result.get("tool_calls", []),
                            "final_spec": q_result.get("final_spec", {}),
                            "explorations": [{"tool_execution": tc} for tc in q_result.get("tool_calls", [])]
                        }
                        
                        eval_result = evaluator.evaluate_task(task_config, agent_result, i)
                        print(f"  {q_result['qid']}: answer={eval_result.answer_score:.2f}, "
                              f"tool={eval_result.tool_score:.2f}, "
                              f"state={eval_result.state_score:.2f}, "
                              f"total={eval_result.total_score:.2f}")


if __name__ == "__main__":
    main()
