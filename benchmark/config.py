"""
Benchmark 模型配置

统一管理所有可用模型的 API 配置，
benchmark 脚本可以通过 --model 参数选择。
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""
    name: str                      # 模型显示名
    base_url: str                  # API 端点
    model: str                     # 模型 ID
    api_key_env: Optional[str]     # API Key 环境变量名
    max_tokens: int = 2000         # 最大输出 token
    temperature: float = 0.0       # 温度
    timeout: int = 180             # 超时时间（秒）
    max_iterations: int = 8        # 多轮迭代最大次数
    tool_choice_format: str = "dict"  # "dict" → {"type": "auto"}, "string" → "auto"
    supports_external_tools: bool = True  # Whether model uses external MCP tools (False for system API)
    

# ============================================================
# 可用模型配置
# ============================================================

MODELS: Dict[str, ModelConfig] = {
    # 自己的系统（本地服务）
    "system": ModelConfig(
        name="My System",
        base_url="http://localhost:8000/v1",
        model="my-system",
        api_key_env=None,  # 本地不需要 API Key
        timeout=300,  # 系统可能需要更长时间
        supports_external_tools=False,  # System has internal tool logic
    ),
    
    # Claude (via OAI Pro)
    "claude": ModelConfig(
        name="Claude Sonnet 4",
        base_url="https://api.oaipro.com/v1",
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    
    # GPT-4o
    "gpt": ModelConfig(
        name="GPT-4o",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        tool_choice_format="string",  # OpenAI native API accepts string
    ),
    
    # GPT-5 (via OAI Pro)
    "gpt5": ModelConfig(
        name="GPT-5",
        base_url="https://api.oaipro.com/v1",
        model="gpt-5",
        api_key_env="OPENAI_API_KEY",
        tool_choice_format="string",  # OAI Pro expects string format like native OpenAI
    ),
    
    # Gemini (via OpenRouter)
    "gemini": ModelConfig(
        name="Gemini 2.5 Flash",
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash-preview-09-2025",
        api_key_env="OPENROUTER_API_KEY",
    ),
    
    # Llama (via OpenRouter)
    "llama": ModelConfig(
        name="Llama 4 Maverick",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-4-maverick",
        api_key_env="OPENROUTER_API_KEY",
    ),
    
    # Qwen (via OpenRouter)
    "qwen": ModelConfig(
        name="Qwen 3 VL 235B",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-vl-235b-a22b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        tool_choice_format="string",  # Qwen 需要字符串格式
    ),
    
    # Grok (via OAI Pro)
    "grok": ModelConfig(
        name="Grok 4",
        base_url="https://openrouter.ai/api/v1",
        model="x-ai/grok-4",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """获取模型配置"""
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODELS[model_name]


def get_api_key(config: ModelConfig) -> Optional[str]:
    """获取 API Key"""
    if config.api_key_env is None:
        return None
    return os.environ.get(config.api_key_env)


def list_available_models() -> Dict[str, str]:
    """列出所有可用模型"""
    return {name: cfg.name for name, cfg in MODELS.items()}
