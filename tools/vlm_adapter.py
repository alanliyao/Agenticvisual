"""
通用VLM工具适配器
支持将工具转换为标准的function calling格式，使任何支持function calling的VLM都能使用

修复内容：
1. 为 array 类型添加 items 定义（OpenAI API 要求）
2. 为 object 类型添加 additionalProperties（避免验证错误）
3. 在转换时自动过滤 vega_spec 参数
4. 改进参数描述
"""

from typing import Dict, List, Any, Optional
from .tool_registry import tool_registry
from config.chart_types import ChartType


class VLMToolAdapter:
    """VLM工具适配器，支持多种格式"""
    
    def __init__(self):
        self.registry = tool_registry
    
    def to_openai_format(self, chart_type: Optional[ChartType] = None) -> List[Dict[str, Any]]:
        """
        转换为OpenAI function calling格式
        
        Args:
            chart_type: 图表类型，如果指定则只返回该类型的工具
            
        Returns:
            OpenAI格式的工具列表
        """
        tools = []
        
        # 获取工具列表
        if chart_type:
            tool_names = self.registry.list_tools_for_chart(chart_type)
        else:
            tool_names = self.registry.list_all_tools()
        
        for tool_name in tool_names:
            tool_info = self.registry.get_tool(tool_name)
            if not tool_info:
                continue
            
            # 转换参数为 JSON Schema 格式
            params_schema = self._convert_params_to_json_schema(tool_info['params'])
            
            # 从 properties 中移除 vega_spec（模型不需要知道这个参数）
            if 'properties' in params_schema and 'vega_spec' in params_schema['properties']:
                del params_schema['properties']['vega_spec']
            
            # 从 required 中移除 vega_spec
            if 'required' in params_schema and 'vega_spec' in params_schema['required']:
                params_schema['required'].remove('vega_spec')
            
            # 🔧 修复：确保所有 array 和 object 类型的 schema 都完整
            self._fix_schema_types(params_schema)
            
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info['description'],
                    "parameters": params_schema
                }
            }
            tools.append(openai_tool)
        
        return tools
    
    def _fix_schema_types(self, schema: Dict[str, Any]) -> None:
        """
        修复 JSON Schema 中的类型定义问题
        - 为 array 类型添加 items
        - 为 object 类型添加 additionalProperties
        
        Args:
            schema: JSON Schema 字典（会被原地修改）
        """
        if 'properties' not in schema:
            return
        
        for prop_name, prop_def in schema['properties'].items():
            prop_type = prop_def.get('type')
            
            # 修复 array 类型：必须有 items 定义
            if prop_type == 'array' and 'items' not in prop_def:
                # 根据参数名推断元素类型
                if any(keyword in prop_name.lower() for keyword in ['range', 'position', 'point', 'coord', 'size', 'extent']):
                    # 数值范围类参数，元素是数字
                    prop_def['items'] = {"type": "number"}
                elif any(keyword in prop_name.lower() for keyword in ['name', 'label', 'category', 'field', 'column']):
                    # 名称/标签类参数，元素是字符串
                    prop_def['items'] = {"type": "string"}
                else:
                    # 默认为数字类型（大多数可视化参数是数值）
                    prop_def['items'] = {"type": "number"}
            
            # 修复 object 类型：需要 properties 或 additionalProperties
            elif prop_type == 'object':
                if 'properties' not in prop_def and 'additionalProperties' not in prop_def:
                    prop_def['additionalProperties'] = True
    
    def to_anthropic_format(self, chart_type: Optional[ChartType] = None) -> List[Dict[str, Any]]:
        """
        转换为Anthropic (Claude) tool use格式
        
        Args:
            chart_type: 图表类型
            
        Returns:
            Anthropic格式的工具列表
        """
        tools = []
        
        # 获取工具列表
        if chart_type:
            tool_names = self.registry.list_tools_for_chart(chart_type)
        else:
            tool_names = self.registry.list_all_tools()
        
        for tool_name in tool_names:
            tool_info = self.registry.get_tool(tool_name)
            if not tool_info:
                continue
            
            # 转换参数
            params_schema = self._convert_params_to_json_schema(tool_info['params'])
            
            # 移除 vega_spec
            if 'properties' in params_schema and 'vega_spec' in params_schema['properties']:
                del params_schema['properties']['vega_spec']
            if 'required' in params_schema and 'vega_spec' in params_schema['required']:
                params_schema['required'].remove('vega_spec')
            
            # 修复 schema 类型
            self._fix_schema_types(params_schema)
            
            anthropic_tool = {
                "name": tool_name,
                "description": tool_info['description'],
                "input_schema": params_schema
            }
            tools.append(anthropic_tool)
        
        return tools
    
    def to_generic_format(self, chart_type: Optional[ChartType] = None) -> List[Dict[str, Any]]:
        """
        转换为通用格式（可用于提示词描述）
        
        Args:
            chart_type: 图表类型
            
        Returns:
            通用格式的工具列表
        """
        tools = []
        
        # 获取工具列表
        if chart_type:
            tool_names = self.registry.list_tools_for_chart(chart_type)
        else:
            tool_names = self.registry.list_all_tools()
        
        for tool_name in tool_names:
            tool_info = self.registry.get_tool(tool_name)
            if not tool_info:
                continue
            
            # 构建参数描述
            params_desc = []
            for param_name, param_spec in tool_info['params'].items():
                # 跳过 vega_spec
                if param_name == 'vega_spec':
                    continue
                    
                param_type = param_spec.get('type', 'any')
                required = param_spec.get('required', False)
                default = param_spec.get('default', 'N/A')
                
                param_str = f"  - {param_name} ({param_type})"
                if required:
                    param_str += " [REQUIRED]"
                elif default != 'N/A':
                    param_str += f" [default={default}]"
                
                params_desc.append(param_str)
            
            tool_desc = {
                "name": tool_name,
                "category": tool_info.get('category', 'unknown'),
                "description": tool_info['description'],
                "parameters": "\n".join(params_desc) if params_desc else "No parameters"
            }
            tools.append(tool_desc)
        
        return tools
    
    def to_prompt_string(self, chart_type: Optional[ChartType] = None) -> str:
        """
        转换为提示词字符串格式（用于不支持function calling的VLM）
        
        Args:
            chart_type: 图表类型
            
        Returns:
            格式化的工具描述字符串
        """
        tools = self.to_generic_format(chart_type)
        
        prompt_parts = ["# Available Tools\n"]
        
        # 按类别分组
        categories = {}
        for tool in tools:
            cat = tool['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
        
        # 生成提示词
        for category, cat_tools in categories.items():
            prompt_parts.append(f"\n## {category.upper()} Tools\n")
            
            for tool in cat_tools:
                prompt_parts.append(f"\n### {tool['name']}")
                prompt_parts.append(f"\n{tool['description']}")
                prompt_parts.append(f"\n**Parameters:**\n{tool['parameters']}\n")
        
        prompt_parts.append("\n## Tool Usage Format\n")
        prompt_parts.append("To use a tool, respond with JSON in this format:\n")
        prompt_parts.append("```json\n")
        prompt_parts.append('{\n')
        prompt_parts.append('  "tool": "tool_name",\n')
        prompt_parts.append('  "params": {\n')
        prompt_parts.append('    "param1": "value1",\n')
        prompt_parts.append('    "param2": "value2"\n')
        prompt_parts.append('  },\n')
        prompt_parts.append('  "reason": "Why you are calling this tool"\n')
        prompt_parts.append('}\n')
        prompt_parts.append("```\n")
        
        return "".join(prompt_parts)
    
    def _convert_params_to_json_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数规范转换为JSON Schema格式
        
        Args:
            params: 参数定义字典
            
        Returns:
            JSON Schema 格式的参数定义
        """
        properties = {}
        required = []
        
        for param_name, param_spec in params.items():
            # 跳过 vega_spec 参数（在这里就过滤掉）
            if param_name == 'vega_spec':
                continue
            
            param_type = param_spec.get('type', 'string')
            
            # Python类型 -> JSON Schema类型 映射
            type_mapping = {
                'str': 'string',
                'string': 'string',
                'int': 'integer',
                'integer': 'integer',
                'float': 'number',
                'number': 'number',
                'bool': 'boolean',
                'boolean': 'boolean',
                'list': 'array',
                'array': 'array',
                'dict': 'object',
                'object': 'object',
                'tuple': 'array',
                'any': 'string'  # 默认为 string
            }
            
            json_type = type_mapping.get(param_type, 'string')
            
            # 构建属性定义
            prop_def = {
                "type": json_type,
                "description": param_spec.get('description', f"{param_name} 参数")
            }
            
            # 🔧 关键修复：为 array 类型添加 items
            if json_type == 'array':
                # 尝试从参数规格中获取元素类型
                item_type = param_spec.get('item_type', param_spec.get('items_type', 'number'))
                item_type_mapping = {
                    'str': 'string',
                    'string': 'string',
                    'int': 'integer',
                    'integer': 'integer',
                    'float': 'number',
                    'number': 'number',
                    'bool': 'boolean',
                    'boolean': 'boolean'
                }
                prop_def['items'] = {
                    "type": item_type_mapping.get(item_type, 'number')
                }
            
            # 🔧 关键修复：为 object 类型添加 additionalProperties
            if json_type == 'object':
                prop_def['additionalProperties'] = True
            
            # 添加默认值
            if 'default' in param_spec:
                prop_def['default'] = param_spec['default']
            
            # 添加枚举值（如果有）
            if 'enum' in param_spec:
                prop_def['enum'] = param_spec['enum']
            
            properties[param_name] = prop_def
            
            # 收集必需参数
            if param_spec.get('required', False):
                required.append(param_name)
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    def generate_tool_execution_guide(self) -> str:
        """生成工具执行指南"""
        guide = """
# Tool Execution Guide

## Overview
This system provides interactive tools for visual analysis. All tools operate on Vega-Lite specifications.

## Core Principles

1. **Tools are automatically connected to the visualization**: You don't need to pass vega_spec, it's handled automatically
2. **Tools return updated state**: Action tools return an updated visualization
3. **Tools are composable**: You can chain multiple tool calls in sequence

## Tool Categories

### Perception Tools
These tools READ the current state:
- `get_data_summary`: Get statistical summary of data
- `get_tooltip_data`: Get data at specific position

### Action Tools  
These tools MODIFY the visualization:
- `zoom`: Zoom to a specific area
- `filter`: Filter data by dimension
- `brush`: Select/brush an area
- `change_encoding`: Change visual encoding
- `highlight`: Highlight specific categories
- `render_chart`: Render the visualization

### Analysis Tools
These tools ANALYZE patterns:
- `identify_clusters`: Find clusters in scatter plots
- `calculate_correlation`: Calculate correlation

## Usage Pattern

1. **Understand the task**: Parse user query
2. **Plan tool usage**: Decide which tools to use
3. **Execute tools**: Call tools with proper parameters
4. **Interpret results**: Analyze tool outputs
5. **Respond to user**: Provide insights based on results

## Example Workflow

```python
# 1. Get data summary to understand the data
result = get_data_summary(scope='all')

# 2. Identify interesting patterns
clusters = identify_clusters(n_clusters=3)

# 3. Highlight findings
updated = highlight(category='cluster_0')

# 4. Return insights to user
```

## Error Handling

- Always check tool result['success']
- If a tool fails, try alternative approaches
- Validate parameters before calling tools
"""
        return guide
    
    def validate_tools(self) -> List[str]:
        """
        验证所有工具的 schema 是否正确
        
        Returns:
            错误信息列表，如果为空则表示所有工具都正确
        """
        errors = []
        tools = self.to_openai_format()
        
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            params = func.get('parameters', {})
            
            if 'properties' in params:
                for prop_name, prop_def in params['properties'].items():
                    prop_type = prop_def.get('type')
                    
                    # 检查 array 类型是否有 items
                    if prop_type == 'array' and 'items' not in prop_def:
                        errors.append(f"工具 '{name}' 的参数 '{prop_name}' 是 array 类型但缺少 items 定义")
                    
                    # 检查 object 类型是否有 properties 或 additionalProperties
                    if prop_type == 'object':
                        if 'properties' not in prop_def and 'additionalProperties' not in prop_def:
                            errors.append(f"工具 '{name}' 的参数 '{prop_name}' 是 object 类型但缺少 properties 或 additionalProperties")
        
        return errors


# 创建全局实例
vlm_adapter = VLMToolAdapter()


# 便捷函数：验证工具定义
def validate_all_tools() -> bool:
    """验证所有工具定义是否正确"""
    errors = vlm_adapter.validate_tools()
    if errors:
        print("❌ 工具定义验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ 所有工具定义验证通过")
        return True