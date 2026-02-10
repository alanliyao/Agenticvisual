"""
全局配置文件
包含 DashScope API 配置、系统参数等
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """全局配置类"""
    
    # ==================== API 配置 ====================
    DASHSCOPE_API_KEY: str = os.getenv('DASHSCOPE_API_KEY', '')
    VLM_MODEL: str = os.getenv('VLM_MODEL', 'qwen-vl-max')
    
    # ==================== 系统配置 ====================
    MAX_ITERATIONS: int = int(os.getenv('MAX_ITERATIONS', '8'))
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR: Path = Path(os.getenv('LOG_DIR', './logs'))
    
    # ==================== 会话配置 ====================
    SESSION_TIMEOUT: int = int(os.getenv('SESSION_TIMEOUT', '3600'))
    
    # ==================== Vega 配置 ====================
    VEGA_RENDERER: str = os.getenv('VEGA_RENDERER', 'canvas')
    IMAGE_FORMAT: str = os.getenv('IMAGE_FORMAT', 'png')
    
    # ==================== 项目路径配置 ====================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    PROMPTS_DIR: Path = PROJECT_ROOT / 'prompts'
    LOGS_DIR: Path = PROJECT_ROOT / 'logs'
    
    # ==================== VLM 参数配置 ====================
    VLM_TEMPERATURE: float = 0
    VLM_MAX_TOKENS: int = 2000
    VLM_TOP_P: float = 0.9
    
    # ==================== 工具执行配置 ====================
    TOOL_EXECUTION_TIMEOUT: int = 30  # 秒
    MAX_TOOL_RETRIES: int = 3
    
    # ==================== 日志配置 ====================
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    APP_LOG_FILE: Path = PROJECT_ROOT / 'logs' / 'app.log'
    ERROR_LOG_FILE: Path = PROJECT_ROOT / 'logs' / 'error.log'
    
    # ==================== 模式配置 ====================
    MAX_GOAL_ORIENTED_ITERATIONS: int = int(os.getenv('MAX_GOAL_ORIENTED_ITERATIONS', '8'))
    GOAL_ACHIEVEMENT_THRESHOLD: float = float(os.getenv('GOAL_ACHIEVEMENT_THRESHOLD', '0.9'))
    MAX_EXPLORATION_ITERATIONS: int = int(os.getenv('MAX_EXPLORATION_ITERATIONS', '8'))
    
    # ==================== Vega 默认尺寸配置 ====================
    VEGA_DEFAULT_WIDTH: int = int(os.getenv('VEGA_DEFAULT_WIDTH', '800'))
    VEGA_DEFAULT_HEIGHT: int = int(os.getenv('VEGA_DEFAULT_HEIGHT', '600'))
    
    # ==================== Vega 渲染配置 ====================
    VEGA_REQUIRE_CLI: bool = os.getenv('VEGA_REQUIRE_CLI', 'false').lower() in ('true', '1', 'yes')
    # 如果设置为 True，将只使用 vega-cli，不使用 mock 渲染
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置的有效性"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is not set in environment variables")
        
        # 创建必要的目录
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_api_key(cls) -> str:
        """获取 API Key"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is not configured")
        return cls.DASHSCOPE_API_KEY
    
    @classmethod
    def get_model_name(cls) -> str:
        """获取模型名称"""
        return cls.VLM_MODEL
    
    @classmethod
    def to_dict(cls) -> dict:
        """转换为字典格式"""
        return {
            'api_key': cls.DASHSCOPE_API_KEY[:10] + '...' if cls.DASHSCOPE_API_KEY else 'Not set',
            'model': cls.VLM_MODEL,
            'max_iterations': cls.MAX_ITERATIONS,
            'log_level': cls.LOG_LEVEL,
            'session_timeout': cls.SESSION_TIMEOUT,
            'vega_renderer': cls.VEGA_RENDERER,
        }


# 在模块加载时验证配置（可选）
# Settings.validate()
