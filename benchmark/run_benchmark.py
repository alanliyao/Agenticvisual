#!/usr/bin/env python3
"""
Unified Benchmark running script (Simplified Version)

Supports new format tasks (multiple questions), can call any model.

Changes:
1. key_insights changed to string array format: ["insight1", "insight2"]
2. Enhanced prompts for accurate question type detection (objective vs subjective)
3. Simplified run logic:
   - Subjective: return KEY_INSIGHTS + REASONING
   - Objective: return ANSWER (single word/number)
"""

import json
import os
import sys
import copy
import base64
import asyncio
import argparse
import re
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
# Evaluation Format Instruction (Enhanced for Question Type Detection)
# ============================================================

EVALUATION_FORMAT = """
=== OUTPUT FORMAT INSTRUCTIONS ===

IMPORTANT: First determine the question type, then format your response accordingly.

**Question Type Detection:**
- OBJECTIVE (QA): Questions asking for specific facts, numbers, categories, yes/no answers
  Examples: "What is the correlation?", "Which country has the highest?", "Is X greater than Y?"
- SUBJECTIVE (Open-ended): Questions asking for analysis, explanation, interpretation, patterns
  Examples: "What insights can you draw?", "Describe the distribution", "What patterns do you observe?"

**For EACH iteration of analysis, output:**

REASONING: <Explain your analysis approach, observations from the current view, and preparations for the next step>

**After completing ALL iterations, provide your final output:**

For OBJECTIVE Questions:
ANSWER: <One word, number, or short phrase ONLY>

For SUBJECTIVE Questions:
KEY_INSIGHTS:
- <Insight 1 with specific data>
- <Insight 2 with specific data>
- <Additional insights as needed>

ANSWER: <Comprehensive answer summarizing your analysis>
"""


# ============================================================
# Utility Functions (Same as before)
# ============================================================

def strip_data_values(spec: Dict) -> Dict:
    """Remove data.values from spec to reduce file size."""
    if not spec:
        return spec
    result = spec.copy()
    if "data" in result and isinstance(result["data"], dict):
        if "values" in result["data"]:
            count = len(result["data"]["values"]) if isinstance(result["data"]["values"], list) else "?"
            result["data"] = {"_values_omitted": f"{count} items"}
    return result


def load_task(task_path: str) -> Dict:
    """Load task config."""
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vega_spec(chart_id: str, task_dir: Path, vega_spec_path: Optional[str] = None) -> Dict:
    """Load Vega spec. Prefer vega_spec_path (relative to project root) when provided."""
    project_root = Path(__file__).resolve().parent.parent
    if vega_spec_path:
        path = project_root / vega_spec_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"vega_spec_path not found: {vega_spec_path} (resolved: {path})")
    
    ID_MAP = {
        "cars_scatter_001": "cars_performance_efficiency",
        "cars_multivariate_002": "cars_performance_efficiency",
        "cars_multiregion_003": "cars_performance_efficiency",
        "scatter_clustering_001": "cars_performance_efficiency",
    }
    file_name = ID_MAP.get(chart_id, chart_id)
    base_paths = [
        task_dir.parent.parent.parent / "data",
        task_dir.parent.parent.parent.parent / "data",
        project_root / "benchmark" / "data",
        project_root / "data",
        project_root / "benchmark_annotation_system" / "backend" / "specs",
    ]
    if "/" in chart_id:
        full_path = Path(chart_id)
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
    tried_paths = []
    for base in base_paths:
        for name in [file_name, chart_id]:
            p = f"{name}.json" if not str(name).endswith(".json") else name
            path = base / p
            tried_paths.append(path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
    raise FileNotFoundError(f"Vega spec not found for chart_id: {chart_id}. Tried: {tried_paths[:6]}")


def save_view(vega_spec: Dict, output_path: Path) -> str:
    """Save view as image."""
    vega_service = get_vega_service()
    result = vega_service.render(vega_spec)
    
    if result.get("success"):
        image_data = result["image_base64"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        
        return str(output_path)
    return ""


def create_client(config: ModelConfig) -> OpenAI:
    """Create OpenAI client."""
    api_key = get_api_key(config)
    if api_key is None:
        api_key = "dummy-key"
    
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout
    )


def encode_image(image_base64: str) -> str:
    """Ensure image is in correct data URL format."""
    if image_base64.startswith("data:"):
        return image_base64
    return f"data:image/png;base64,{image_base64}"


# ============================================================
# MCP Helper Functions
# ============================================================

def _fix_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common incomplete definitions in JSON Schema."""
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
    return f"""You are a professional data visualization analysis assistant.

Current chart type: **{chart_type}**

## Analysis Strategy
1. Read the user question carefully and determine if it's OBJECTIVE (specific answer) or SUBJECTIVE (analysis/interpretation)
2. Use the provided tools for analysis - do NOT imagine or simulate tool operations
3. If a specific tool is mentioned in the question, prioritize using that tool
4. Answer based on actual tool results and observations

## Tool Selection Guidelines
- **scatter_plot**: select_region, calculate_correlation, identify_clusters, zoom_dense_area, brush_region, change_encoding, filter_categorical, show_regression
- **bar_chart**: sort_bars, highlight_top_n, filter_categories, expand_stack, toggle_stack_mode, change_encoding
- **line_chart**: zoom_time_range, detect_anomalies, highlight_trend, bold_lines, filter_lines, change_encoding
- **heatmap**: adjust_color_scale, filter_cells, highlight_region, cluster_rows_cols, change_encoding
- **parallel_coordinates**: filter_dimension, highlight_cluster, reorder_dimensions
- **sankey_diagram**: trace_node, highlight_path, filter_flow

## Stop Criteria
- When the question is answered with sufficient evidence
- When all required tools have been called
- When repeating the same tool call multiple times
- When gaining no new insights from additional iterations

## Output Requirements
- For OBJECTIVE questions: provide a clear, concise answer (word/number/phrase)
- For SUBJECTIVE questions: provide key insights and comprehensive analysis"""


def get_analysis_prompt(is_final: bool = False) -> str:
    """Generate analysis phase prompt - every round returns key_insights."""
    if is_final:
        return """Based on all your analysis, provide your FINAL response in this EXACT JSON format:

```json
{
  "question_type": "objective" or "subjective",
  "reasoning": "Your step-by-step reasoning for THIS round",
  "key_insights": ["New insight from this round", "Another insight discovered"],
  "answer": "YOUR FINAL ANSWER - REQUIRED",
  "exploration_complete": true
}
```

CRITICAL RULES:
1. "question_type": 
   - "objective": questions asking for specific values (numbers, names, yes/no, categories)
   - "subjective": questions asking for analysis, explanation, or interpretation

2. "key_insights": MUST be an array of STRINGS. Include insights discovered in THIS round.
   - For BOTH objective and subjective questions, provide insights
   - Example: ["4-cylinder cars dominate the low-HP region", "The correlation is -0.78"]

3. "answer" - REQUIRED, NEVER LEAVE EMPTY:
   - OBJECTIVE: Direct answer ONLY (e.g., "4", "-0.78", "Japan", "Yes")
   - SUBJECTIVE: Comprehensive summary paragraph

Return VALID JSON only."""
    else:
        return """Analyze the current view and tool results. Return JSON with insights from THIS round:

```json
{
  "question_type": "objective" or "subjective",
  "reasoning": "What you observed and analyzed in this round",
  "key_insights": ["Insight 1 from this round", "Insight 2 from this round"],
  "answer": "Your current best answer (update as you learn more)",
  "exploration_complete": false,
  "next_action": "What tool to use next and why"
}
```

IMPORTANT:
- "key_insights": Include what you learned THIS round (even for objective questions)
- "answer": Always provide your current best answer
- Each round should add new insights based on new observations"""


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
    
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return {}


def extract_answer_from_text(content: str) -> Dict[str, Any]:
    """
    Extract answer from plain text response when JSON parsing fails.
    Looks for ANSWER:, KEY_INSIGHTS:, REASONING: markers.
    """
    result = {
        "question_type": "subjective",
        "answer": "",
        "key_insights": [],
        "reasoning": ""
    }
    
    # Extract ANSWER
    answer_match = re.search(r'ANSWER:\s*(.+?)(?=\n(?:KEY_INSIGHTS|REASONING)|$)', content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    # Extract KEY_INSIGHTS (bullet points)
    insights_match = re.search(r'KEY_INSIGHTS:\s*(.+?)(?=\n(?:ANSWER|REASONING)|$)', content, re.DOTALL | re.IGNORECASE)
    if insights_match:
        insights_text = insights_match.group(1)
        # Parse bullet points
        insights = re.findall(r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)', insights_text, re.DOTALL)
        result["key_insights"] = [ins.strip() for ins in insights if ins.strip()]
    
    # Extract REASONING
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n(?:ANSWER|KEY_INSIGHTS)|$)', content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    # Determine question type based on answer format
    if result["answer"]:
        # Short answers (single word/number) indicate objective
        answer_words = result["answer"].split()
        if len(answer_words) <= 3 and not result["key_insights"]:
            result["question_type"] = "objective"
    
    return result


def save_image_from_base64(image_base64: str, output_path: Path) -> str:
    """Save base64 image to file."""
    image_data = base64.b64decode(image_base64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    return str(output_path)


# ============================================================
# System API Calling (No External Tools)
# ============================================================

def _extract_message_dict(message):
    """Extract dict from message object."""
    if isinstance(message, dict):
        return message
    if hasattr(message, 'model_dump'):
        return message.model_dump()
    if hasattr(message, '__dict__'):
        return message.__dict__
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
    """
    question_text = question.get("question", "")
    qid = question.get("qid", "unknown")
    
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
        
        msg_dict = _extract_message_dict(message)
        final_spec = msg_dict.get('final_spec', vega_spec)
        tool_calls_history = msg_dict.get('tool_calls_history', [])
        reasoning = msg_dict.get('reasoning', "")
        iterations = msg_dict.get('iterations', 1)
        
        tool_calls = [
            {
                "tool_name": tc.get("tool_name", ""),
                "parameters": tc.get("parameters", {}),
                "result": tc.get("result", {})
            }
            for tc in tool_calls_history if isinstance(tc, dict)
        ]
        
        # Parse structured output from answer
        parsed = parse_json_from_response(answer)
        if not parsed:
            parsed = extract_answer_from_text(answer)
        
        key_insights = parsed.get("key_insights", [])
        question_type = parsed.get("question_type", "subjective").lower()  # Normalize to lowercase
        final_answer = parsed.get("answer") or parsed.get("final_answer") or answer
        parsed_reasoning = parsed.get("reasoning", reasoning)
        
        # Build reasoning_rounds for evaluator compatibility
        reasoning_rounds = []
        if parsed_reasoning:
            reasoning_rounds.append({
                "iteration": 1,
                "reasoning": parsed_reasoning
            })
        
        gt = question.get("ground_truth") or {}
        return {
            "qid": qid,
            "question": question_text,
            "success": True,
            "answer": str(final_answer).strip() if final_answer else "",
            "question_type": question_type,
            "task_type": gt.get("task_type"),
            "key_insights": key_insights,
            "reasoning_rounds": reasoning_rounds,
            "reasoning": parsed_reasoning,
            "model": config.model,
            "tool_calls": tool_calls,
            "final_spec": strip_data_values(final_spec),
            "iterations": iterations
        }
        
    except Exception as e:
        print(f"      System API error: {e}")
        gt = question.get("ground_truth") or {}
        return {
            "qid": qid,
            "question": question_text,
            "success": False,
            "error": str(e),
            "answer": "",
            "question_type": "unknown",
            "task_type": gt.get("task_type"),
            "key_insights": [],
            "reasoning_rounds": [],
            "reasoning": "",
            "tool_calls": [],
            "final_spec": strip_data_values(vega_spec),
            "iterations": 0
        }


# ============================================================
# Multi-turn MCP Tool Calling (Simplified)
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
    max_iterations: int = 8
) -> Dict:
    """
    Multi-turn MCP tool calling for a single question.
    Simplified output format.
    """
    question_text = question.get("question", "")
    qid = question.get("qid", "unknown")
    
    # Render initial image
    vega_service = get_vega_service()
    render_result = vega_service.render(vega_spec)
    
    if not render_result.get('success'):
        gt = question.get("ground_truth") or {}
        return {
            "qid": qid,
            "question": question_text,
            "success": False,
            "error": f"Render failed: {render_result.get('error')}",
            "answer": "",
            "question_type": "unknown",
            "task_type": gt.get("task_type"),
            "key_insights": [],
            "reasoning_rounds": [],
            "reasoning": "",
            "tool_calls": [],
            "final_spec": strip_data_values(vega_spec),
            "iterations": 0
        }
    
    current_image = render_result['image_base64']
    current_spec = copy.deepcopy(vega_spec)
    
    # Initialize conversation
    system_prompt = get_system_prompt(chart_type)
    formatted_question = f"Please answer the following question:\n\n{question_text}\n\n{EVALUATION_FORMAT}"
    messages = [
        {'role': 'system', 'content': system_prompt},
        format_user_message_with_image(formatted_question, current_image)
    ]
    
    all_tool_calls = []
    all_reasoning_rounds = []  # List[Dict] 格式，兼容评估器
    all_insights = []
    final_answer = ""
    question_type = "subjective"
    
    # Multi-turn dialog loop
    for i in range(max_iterations):
        print(f"    [{qid}] Round {i+1}...")
        
        # Phase 1: Tool calling
        tool_choice_value = {"type": "auto"} if config.tool_choice_format == "dict" else "auto"
        
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
                    'tool_name': tool_name,
                    'parameters': tool_args,
                    'result': {
                        'success': tool_result.get('success', False),
                        'message': tool_result.get('message', '')
                    }
                })
                
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
            print(f"      No tool call")
        
        # Phase 2: Analysis - use final prompt on last iteration
        is_final = (i == max_iterations - 1) or (not message1.tool_calls)
        analysis_prompt = get_analysis_prompt(is_final=is_final)
        messages.append(format_user_message_with_image(analysis_prompt, current_image))
        
        try:
            response2 = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                response_format={"type": "json_object"}
            )
        except Exception:
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
        content2 = (message2.content or "").strip()
        
        parsed = parse_json_from_response(content2)
        if not parsed:
            parsed = extract_answer_from_text(content2)
        
        # Collect insights (as strings)
        round_insights = parsed.get('key_insights', [])
        all_insights.extend(round_insights)
        
        # Collect reasoning as List[Dict] for evaluator compatibility
        round_reasoning = parsed.get('reasoning', '')
        if round_reasoning:
            all_reasoning_rounds.append({
                "iteration": i + 1,
                "reasoning": round_reasoning
            })
        
        # Update question type and answer
        if parsed.get('question_type'):
            question_type = parsed.get('question_type').lower()  # Normalize to lowercase
        
        # Extract answer - try multiple fields
        extracted_answer = parsed.get('answer') or parsed.get('final_answer') or ''
        if extracted_answer:
            final_answer = str(extracted_answer).strip()
        
        messages.append({'role': 'assistant', 'content': content2})
        
        exploration_complete = parsed.get('exploration_complete', False)
        if exploration_complete or is_final:
            print(f"    [{qid}] Done, {i + 1} rounds")
            break
    
    # Deduplicate insights
    unique_insights = list(dict.fromkeys(all_insights))
    
    # If no explicit answer, try fallback extraction
    if not final_answer:
        # For objective questions, try to extract from insights or content
        if question_type.lower() == "objective" and unique_insights:
            # Try to extract a short answer from first insight
            first_insight = unique_insights[0]
            # If insight looks like an answer (short), use it
            if len(first_insight.split()) <= 5:
                final_answer = first_insight
        elif question_type.lower() == "subjective" and unique_insights:
            final_answer = " ".join(unique_insights)
    
    # Save final view
    final_view_path = ""
    if output_dir:
        view_path = output_dir / "images" / f"{qid}_final.png"
        final_view_path = save_image_from_base64(current_image, view_path)
    
    gt = question.get("ground_truth") or {}
    return {
        "qid": qid,
        "question": question_text,
        "success": True,
        "answer": final_answer,
        "question_type": question_type,
        "task_type": gt.get("task_type"),
        "key_insights": unique_insights,
        "reasoning_rounds": all_reasoning_rounds,
        "reasoning": "\n".join([r["reasoning"] for r in all_reasoning_rounds]),
        "model": config.model,
        "tool_calls": all_tool_calls,
        "final_spec": strip_data_values(current_spec),
        "iterations": len(all_reasoning_rounds) or 1,
        "final_view_path": final_view_path
    }


# ============================================================
# Main Task Runner
# ============================================================

async def run_task_async(
    task_path: str,
    model_name: str,
    save_views: bool = True,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run full task (all questions).
    """
    # Load task
    task = load_task(task_path)
    task_id = task.get("task_id", task.get("chart_id", Path(task_path).stem))
    questions = task.get("questions", [])
    chart_type = task.get("metadata", {}).get("chart_type", "scatter_plot")
    
    if not questions:
        return {"success": False, "error": "No questions found in task"}
    
    # Get model config
    config = get_model_config(model_name)
    client = create_client(config)
    
    # Load Vega spec (prefer vega_spec_path when present)
    task_dir = Path(task_path).parent
    chart_id = task.get("chart_id", task_id)
    vega_spec_path = task.get("vega_spec_path")
    vega_spec = load_vega_spec(chart_id, task_dir, vega_spec_path=vega_spec_path)
    
    # Output directory
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
        # System API path
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
                print(f"    Type: {result.get('question_type', 'unknown')}")
                print(f"    Answer: {result.get('answer', '')[:80]}...")
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
                
                mcp_tools_response = await mcp_session.list_tools()
                mcp_tools = mcp_tools_response.tools
                openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)
                print(f"Retrieved {len(openai_tools)} tools")
                print("-" * 50)
                
                # Run each question
                for i, question in enumerate(questions):
                    qid = question.get("qid", f"q_{i}")
                    print(f"\n[{i+1}/{len(questions)}] {qid}: {question.get('question', '')[:50]}...")
                    
                    result = await run_multi_turn_with_mcp(
                        mcp_session=mcp_session,
                        client=client,
                        config=config,
                        openai_tools=openai_tools,
                        question=question,
                        vega_spec=copy.deepcopy(vega_spec),
                        chart_type=chart_type,
                        output_dir=out_path if save_views else None,
                        max_iterations=config.max_iterations
                    )
                    results.append(result)
                    
                    if result.get("success"):
                        print(f"    Type: {result.get('question_type', 'unknown')}")
                        print(f"    Answer: {result.get('answer', '')[:80]}...")
                        tools_used = [tc['tool_name'] for tc in result.get('tool_calls', [])]
                        print(f"    Tools: {tools_used}")
                    else:
                        print(f"    Failed: {result.get('error', 'Unknown')}")
    
    # Summarize results (task_type is per-question in ground_truth; no top-level task_type)
    final_result = {
        "task_id": task_id,
        "model": model_name,
        "model_name": config.name,
        "timestamp": datetime.now().isoformat(),
        "questions_count": len(questions),
        "success_count": sum(1 for r in results if r.get("success")),
        "results": results
    }
    
    # Save results
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    final_result["_output_dir"] = str(out_path)
    
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
    """Run full task (sync wrapper)."""
    return asyncio.run(run_task_async(task_path, model_name, save_views, output_dir))


# ============================================================
# Command Line Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Runner (Simplified)")
    parser.add_argument("--task", help="Path to task JSON file")
    parser.add_argument("--model", help="Model to use (see --list-models)")
    parser.add_argument("--models", help="Comma-separated list of models")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--no-save-views", action="store_true", help="Don't save view images")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after benchmark")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, display in list_available_models().items():
            print(f"  {name}: {display}")
        return
    
    if not args.task:
        print("Error: --task is required when running benchmarks")
        parser.print_help()
        return
    
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.model:
        models = [args.model]
    else:
        print("Error: --model or --models required")
        parser.print_help()
        return
    
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
    
    # Evaluation (run when --eval, always save results to output dir)
    if args.eval and all_results:
        print("\n" + "=" * 50)
        print("Running evaluation...")
        from benchmark.evaluators import UnifiedEvaluator
        from benchmark.run_evaluation import result_to_dict
        
        evaluator = UnifiedEvaluator()
        task_config = load_task(args.task)
        
        for result in all_results:
            if result.get("results"):
                print(f"\nModel: {result['model_name']}")
                eval_results_list = []
                for i, q_result in enumerate(result["results"]):
                    if q_result.get("success"):
                        # Build evaluation input
                        agent_result = {
                            "answer": q_result.get("answer"),
                            "question_type": q_result.get("question_type", "subjective"),
                            "key_insights": q_result.get("key_insights", []),  # List[str]
                            "reasoning_rounds": q_result.get("reasoning_rounds", []),  # List[Dict]
                            "reasoning": q_result.get("reasoning", ""),  # String
                            "tool_calls": q_result.get("tool_calls", []),
                            "final_spec": q_result.get("final_spec", {})
                        }
                        
                        eval_result = evaluator.evaluate_task(task_config, agent_result, i)
                        qid = q_result.get("qid", f"q_{i}")
                        line = f"  {qid}: answer={eval_result.answer_score:.2f}, " \
                               f"tool={eval_result.tool_score:.2f}, " \
                               f"reasoning={eval_result.reasoning_score:.2f}, " \
                               f"state={eval_result.state_score:.2f}, " \
                               f"total={eval_result.total_score:.2f}"
                        jr = eval_result.agent_judge_result
                        if eval_result.agent_judge_triggered and jr:
                            line += f"\n    [Judge] {jr.get('verdict','')} " \
                                    f"adjusted={jr.get('adjusted_score', jr.get('final_score', 0)):.2f}"
                            if jr.get("reasoning"):
                                line += f"\n    Reasoning: {jr['reasoning'][:150]}{'...' if len(jr.get('reasoning',''))>150 else ''}"
                        print(line)
                        d = result_to_dict(eval_result)
                        d["qid"] = qid
                        d["question_idx"] = i
                        eval_results_list.append(d)
                    else:
                        eval_results_list.append({"qid": q_result.get("qid", f"q_{i}"), "success": False})
                
                # Save eval results to output dir (alongside result.json)
                output_dir = result.get("_output_dir")
                if output_dir and eval_results_list:
                    eval_path = Path(output_dir) / "eval_result.json"
                    with open(eval_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "model": result.get("model_name", result.get("model", "")),
                            "task_id": result.get("task_id", ""),
                            "timestamp": datetime.now().isoformat(),
                            "results": eval_results_list
                        }, f, indent=2, ensure_ascii=False)
                    print(f"  Evaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()