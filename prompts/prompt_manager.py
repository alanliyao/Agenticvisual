"""
提示词管理器
负责加载、组装和缓存提示词模板
"""

from pathlib import Path
from typing import Dict, Optional
from config.chart_types import ChartType
from config.intent_types import IntentType


class PromptManager:
    """提示词管理器类"""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        初始化提示词管理器
        
        Args:
            prompts_dir: 提示词文件目录，默认为当前模块的prompts目录
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, str] = {}
    
    def _load_prompt_file(self, file_path: Path) -> str:
        """
        从文件加载提示词
        
        Args:
            file_path: 提示词文件路径
            
        Returns:
            提示词内容
        """
        cache_key = str(file_path)
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 读取文件
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 缓存内容
        self._cache[cache_key] = content
        
        return content
    
    def get_base_system_role(self) -> str:
        """
        获取基础系统角色提示词
        
        Returns:
            基础系统角色提示词
        """
        file_path = self.prompts_dir / 'base' / 'system_role.txt'
        return self._load_prompt_file(file_path)
    
    def get_chitchat_prompt(self) -> str:
        """
        获取闲聊模式提示词
        
        Returns:
            闲聊提示词
        """
        file_path = self.prompts_dir / 'base' / 'chitchat.txt'
        return self._load_prompt_file(file_path)
    
    def get_intent_classifier_prompt(self) -> str:
        """
        获取意图识别提示词
        
        Returns:
            意图识别提示词
        """
        file_path = self.prompts_dir / 'intent_recognition' / 'intent_classifier.txt'
        return self._load_prompt_file(file_path)
    
    def get_chart_specific_prompt(self, chart_type: ChartType) -> str:
        """
        获取图表类型专属提示词
        
        Args:
            chart_type: 图表类型
            
        Returns:
            图表专属提示词
        """
        # 映射图表类型到文件名
        chart_file_mapping = {
            ChartType.BAR_CHART: 'bar_chart.txt',
            ChartType.LINE_CHART: 'line_chart.txt',
            ChartType.SCATTER_PLOT: 'scatter_plot.txt',
            ChartType.PARALLEL_COORDINATES: 'parallel_coordinates.txt',
            ChartType.HEATMAP: 'heatmap.txt',
            ChartType.SANKEY_DIAGRAM: 'sankey_diagram.txt',
        }
        
        filename = chart_file_mapping.get(chart_type)
        if not filename:
            return ""
        
        file_path = self.prompts_dir / 'chart_specific' / filename
        return self._load_prompt_file(file_path)
    
    def get_goal_oriented_prompt(self) -> str:
        """
        获取目标导向模式提示词
        
        Returns:
            目标导向提示词
        """
        file_path = self.prompts_dir / 'modes' / 'goal_oriented.txt'
        return self._load_prompt_file(file_path)
    
    def get_autonomous_exploration_prompt(self) -> str:
        """
        获取自主探索模式提示词
        
        Returns:
            自主探索提示词
        """
        file_path = self.prompts_dir / 'modes' / 'autonomous_exploration.txt'
        return self._load_prompt_file(file_path)
    
    def get_benchmark_answer_instruction(self) -> str:
        """
        获取 Benchmark 评估格式要求提示词
        
        Returns:
            Benchmark 评估格式要求提示词
        """
        file_path = self.prompts_dir / 'modes' / 'benchmark_answer_instruction.txt'
        return self._load_prompt_file(file_path)
    
    def assemble_system_prompt(
        self,
        chart_type: Optional[ChartType] = None,
        intent_type: Optional[IntentType] = None,
        mode: Optional[str] = None,
        include_tools: bool = False,
        tools_description: str = "",
        benchmark_mode: bool = False
    ) -> str:
        """
        组装完整的系统提示词
        
        Args:
            chart_type: 图表类型
            intent_type: 意图类型
            mode: 模式字符串 ("goal_oriented" 或 "autonomous_exploration")，会被转换为intent_type
            include_tools: 是否包含工具描述
            tools_description: 工具描述文本
            benchmark_mode: 是否在 benchmark 评估模式（会添加 ANSWER 字段要求）
            
        Returns:
            组装后的完整系统提示词
        """
        # 如果提供了mode字符串，转换为intent_type
        if mode and not intent_type:
            if mode == "goal_oriented":
                intent_type = IntentType.EXPLICIT_ANALYSIS
            elif mode == "autonomous_exploration":
                intent_type = IntentType.VAGUE_EXPLORATION
        
        parts = []
        
        # 1. 基础系统角色
        parts.append(self.get_base_system_role())
        
        # 2. 图表类型专属提示词
        if chart_type and chart_type != ChartType.UNKNOWN:
            chart_prompt = self.get_chart_specific_prompt(chart_type)
            if chart_prompt:
                parts.append("\n\n" + "="*60)
                parts.append("# 当前图表类型专属指导")
                parts.append("="*60)
                parts.append(chart_prompt)
        
        # 3. 分析模式提示词
        if intent_type:
            if intent_type == IntentType.CHITCHAT:
                parts.append("\n\n" + "="*60)
                parts.append("# 当前模式：闲聊对话")
                parts.append("="*60)
                parts.append(self.get_chitchat_prompt())
            
            elif intent_type == IntentType.EXPLICIT_ANALYSIS:
                parts.append("\n\n" + "="*60)
                parts.append("# 当前模式：目标导向分析")
                parts.append("="*60)
                parts.append(self.get_goal_oriented_prompt())
            
            elif intent_type == IntentType.VAGUE_EXPLORATION:
                parts.append("\n\n" + "="*60)
                parts.append("# 当前模式：自主探索分析")
                parts.append("="*60)
                parts.append(self.get_autonomous_exploration_prompt())
        
        # 4. 工具描述
        if include_tools and tools_description:
            parts.append("\n\n" + "="*60)
            parts.append("# 可用工具列表")
            parts.append("="*60)
            parts.append(tools_description)
        
        # 5. Benchmark 评估格式要求（仅在 benchmark_mode 时添加）
        if benchmark_mode:
            parts.append("\n\n" + "="*60)
            parts.append("# Benchmark 评估格式要求")
            parts.append("="*60)
            parts.append(self.get_benchmark_answer_instruction())
        
        # 组装所有部分
        full_prompt = "\n".join(parts)
        
        return full_prompt
    
    def get_intent_recognition_prompt(
        self,
        user_query: str,
        chart_type: Optional[ChartType] = None
    ) -> str:
        """
        获取用于意图识别的完整提示词
        
        Args:
            user_query: 用户查询
            chart_type: 当前图表类型
            
        Returns:
            意图识别提示词
        """
        parts = []
        
        # 基础意图识别提示词
        parts.append(self.get_intent_classifier_prompt())
        
        # 添加当前上下文
        parts.append("\n\n" + "="*60)
        parts.append("# 当前上下文")
        parts.append("="*60)
        
        if chart_type and chart_type != ChartType.UNKNOWN:
            parts.append(f"当前图表类型：{chart_type.value}")
        
        parts.append(f"\n用户查询：{user_query}")
        
        parts.append("\n请分析上述用户查询的意图类型，并按照指定的JSON格式输出结果。")
        
        return "\n".join(parts)
    
    def clear_cache(self):
        """清空提示词缓存"""
        self._cache.clear()
    
    def preload_all_prompts(self):
        """预加载所有提示词到缓存"""
        # 加载基础提示词
        self.get_base_system_role()
        self.get_chitchat_prompt()
        self.get_intent_classifier_prompt()
        
        # 加载所有图表类型提示词
        for chart_type in ChartType:
            if chart_type != ChartType.UNKNOWN:
                try:
                    self.get_chart_specific_prompt(chart_type)
                except FileNotFoundError:
                    pass
        
        # 加载模式提示词
        self.get_goal_oriented_prompt()
        self.get_autonomous_exploration_prompt()


# 创建全局实例
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """获取提示词管理器单例"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
