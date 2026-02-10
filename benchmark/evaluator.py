"""
Benchmark评估器
评估维度：洞察质量(60%)、推理过程(40%)
使用语义相似度进行洞察匹配
"""

import json
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer


class BenchmarkEvaluator:
    """Benchmark评估器
    
    评估维度：
    1. 洞察质量 (60%): Recall + Precision + Depth
    2. 推理过程 (40%): 连贯性 + 工具调用 + 工具路径 + 推理对齐
    """
    
    def __init__(self, ground_truth: Dict):
        self.gt = ground_truth
        print("📦 加载语义相似度模型...")
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ 模型加载完成")
    
    def evaluate(self, agent_result: Dict) -> Dict:
        """完整评估"""
        explorations = agent_result.get('explorations', [])
        
        insight_score = self.evaluate_insight_quality(explorations)
        reasoning_score = self.evaluate_reasoning_process(explorations)
        
        weights = {'insight_quality': 0.60, 'reasoning_process': 0.40}
        total_score = insight_score * weights['insight_quality'] + reasoning_score * weights['reasoning_process']
        
        return {
            'total_score': round(total_score, 2),
            'dimension_scores': {
                'insight_quality': round(insight_score, 2),
                'reasoning_process': round(reasoning_score, 2)
            },
            'weights': weights,
            'details': {
                'total_explorations': len(explorations),
                'insights_found': self._count_insights(explorations),
                'tools_used': self._get_tools_used(explorations),
                'insights_before_dedup': getattr(self, '_dedup_stats', (0, 0))[0],
                'insights_after_dedup': getattr(self, '_dedup_stats', (0, 0))[1],
            }
        }
    
    def evaluate_insight_quality(self, explorations: List[Dict]) -> float:
        """评估洞察质量：Recall + Precision + Depth"""
        gt_insights = self.gt['insight_quality']['critical_insights']
        criteria = self.gt['insight_quality']['evaluation_criteria']
        
        # 收集所有 key_insights
        agent_insights = []
        for exp in explorations:
            summary = exp.get('analysis_summary', {})
            agent_insights.extend(summary.get('key_insights', []))
        
        # 语义去重，降低重复洞察对Precision的稀释
        before_cnt = len(agent_insights)
        agent_insights = self._dedup_insights(agent_insights)
        after_cnt = len(agent_insights)
        if before_cnt != after_cnt:
            print(f"🧹 洞察去重: {before_cnt} -> {after_cnt}")
        self._dedup_stats = (before_cnt, after_cnt)
        
        if not agent_insights:
            return 0.0
        
        # Recall - 每个GT洞察的最佳匹配分数
        recall_scores = [self._calc_match_score(gt, agent_insights) for gt in gt_insights]
        recall = np.mean(recall_scores)
        
        # Precision - agent洞察中有效的比例
        valid_count = sum(1 for ins in agent_insights if self._is_valid_insight(ins, gt_insights))
        precision = valid_count / len(agent_insights)
        
        # Depth - 洞察深度
        depth_scores = [self._assess_depth(ins) for ins in agent_insights]
        avg_depth = np.mean(depth_scores) / 3.0  # 归一化到0-1
        
        score = (
            recall * criteria['recall_weight'] * 100 +
            precision * criteria['precision_weight'] * 100 +
            avg_depth * criteria['depth_weight'] * 100
        )
        
        return min(100, score)
    
    def evaluate_reasoning_process(self, explorations: List[Dict]) -> float:
        """评估推理过程：连贯性 + 工具调用 + 工具路径 + 推理对齐"""
        coherence = self._eval_coherence(explorations)
        tool_usage = self._eval_tool_usage(explorations)
        tool_path = self._eval_tool_path(explorations)
        reasoning_alignment = self._eval_reasoning_alignment(explorations)
        
        return (
            coherence * 0.15 +
            tool_usage * 0.35 +
            tool_path * 0.30 +
            reasoning_alignment * 0.20
        )
    
    # ========================================
    # 洞察评估辅助方法
    # ========================================
    
    def _calc_match_score(self, gt_insight: Dict, agent_insights: List[str]) -> float:
        """计算GT洞察与agent洞察的最佳匹配分数（余弦相似度）"""
        gt_content = gt_insight['content']
        gt_emb = self.semantic_model.encode(gt_content, convert_to_numpy=True)
        
        max_sim = 0.0
        for agent_ins in agent_insights:
            if not agent_ins or len(agent_ins.strip()) < 5:
                continue
            agent_emb = self.semantic_model.encode(agent_ins, convert_to_numpy=True)
            sim = np.dot(gt_emb, agent_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(agent_emb) + 1e-8)
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def _is_valid_insight(self, agent_insight: str, gt_insights: List[Dict]) -> bool:
        """检查agent洞察是否匹配任一GT洞察（阈值0.5）"""
        if not agent_insight or len(agent_insight.strip()) < 5:
            return False
        
        agent_emb = self.semantic_model.encode(agent_insight, convert_to_numpy=True)
        
        for gt in gt_insights:
            gt_emb = self.semantic_model.encode(gt['content'], convert_to_numpy=True)
            sim = np.dot(agent_emb, gt_emb) / (np.linalg.norm(agent_emb) * np.linalg.norm(gt_emb) + 1e-8)
            if sim > 0.5:
                return True
        
        return False
    
    def _assess_depth(self, insight: str) -> int:
        """评估洞察深度：1=描述性, 2=诊断性, 3=预测性"""
        if not insight:
            return 1
        
        insight_lower = insight.lower()
        
        level3_kw = ['预测', '预期', '将会', '会导致', '预计', 'will', 'forecast', 'predict', 'expect', '如果', 'if']
        level2_kw = ['因为', '由于', '导致', '原因', '造成', 'because', 'due to', 'caused by', '所以', 'therefore', '表明']
        
        if any(kw in insight_lower for kw in level3_kw):
            return 3
        if any(kw in insight_lower for kw in level2_kw):
            return 2
        return 1

    def _dedup_insights(self, insights: List[str], threshold: float = 0.80) -> List[str]:
        """基于语义相似度的去重，保留代表性洞察"""
        deduped: List[str] = []
        if not insights:
            return deduped
        
        for ins in insights:
            if not ins or len(ins.strip()) < 5:
                continue
            ins_clean = ins.strip()
            ins_emb = self.semantic_model.encode(ins_clean, convert_to_numpy=True)
            
            is_dup = False
            for kept in deduped:
                kept_emb = self.semantic_model.encode(kept, convert_to_numpy=True)
                sim = float(np.dot(ins_emb, kept_emb) / (np.linalg.norm(ins_emb) * np.linalg.norm(kept_emb) + 1e-8))
                if sim >= threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                deduped.append(ins_clean)
        
        return deduped
    
    # ========================================
    # 推理过程评估辅助方法
    # ========================================
    
    def _eval_coherence(self, explorations: List[Dict]) -> float:
        """评估推理连贯性"""
        if len(explorations) <= 1:
            return 100.0
        
        score = 100.0
        
        for i in range(1, len(explorations)):
            prev = explorations[i-1]
            curr = explorations[i]
            
            # 获取前一步洞察
            prev_insights = prev.get('analysis_summary', {}).get('key_insights', [])
            
            # 获取当前步的 reasoning
            curr_reasoning = curr.get('analysis_summary', {}).get('reasoning', '')
            curr_tool = (curr.get('tool_execution') or {}).get('tool_name', '')
            
            # 如果前一步有洞察，检查当前reasoning是否有引用
            if prev_insights and curr_reasoning:
                has_ref = any(self._concept_referenced(ins, curr_reasoning) for ins in prev_insights)
                if not has_ref:
                    score -= 5
            
            # 检查重复工具调用
            if curr_tool == 'identify_clusters':
                used_before = any(
                    (exp.get('tool_execution') or {}).get('tool_name') == 'identify_clusters'
                    for exp in explorations[:i]
                )
                if used_before:
                    score -= 10
        
        return max(0, score)
    
    def _concept_referenced(self, insight: str, reasoning: str) -> bool:
        """检查洞察概念是否在reasoning中被引用"""
        if not insight or not reasoning:
            return False
        
        insight_words = set(insight.lower().split())
        reasoning_lower = reasoning.lower()
        
        overlap = sum(1 for w in insight_words if w in reasoning_lower)
        return overlap / len(insight_words) > 0.3 if insight_words else False
    
    def _eval_tool_usage(self, explorations: List[Dict]) -> float:
        """评估工具调用覆盖率"""
        if 'required_tools' not in self.gt.get('reasoning_process', {}):
            return 100.0
        
        required = set(self.gt['reasoning_process']['required_tools'])
        if not required:
            return 100.0
        
        used = set()
        for exp in explorations:
            tool_exec = exp.get('tool_execution') or {}
            tool_name = tool_exec.get('tool_name')
            if tool_name:
                used.add(tool_name)
        
        coverage = len(required & used) / len(required)
        return coverage * 100
    
    def _eval_tool_path(self, explorations: List[Dict]) -> float:
        """评估工具调用路径（LCS相似度）"""
        if 'reference_optimal_path' not in self.gt.get('reasoning_process', {}):
            return 100.0
        
        gt_path = self.gt['reasoning_process']['reference_optimal_path']
        if not gt_path:
            return 100.0
        
        gt_seq = [step['tool'] for step in gt_path]
        
        agent_seq = []
        for exp in explorations:
            tool_exec = exp.get('tool_execution') or {}
            tool_name = tool_exec.get('tool_name')
            if tool_name:
                agent_seq.append(tool_name)
        
        if not agent_seq:
            return 0.0
        
        lcs_len = self._lcs(gt_seq, agent_seq)
        return (lcs_len / len(gt_seq)) * 100

    def _eval_reasoning_alignment(self, explorations: List[Dict]) -> float:
        """评估agent推理与GT参考推理的对齐程度（余弦相似度均值）"""
        gt_reasonings = self._get_gt_reasonings()
        if not gt_reasonings:
            return 100.0
        
        agent_reasonings: List[str] = []
        for exp in explorations:
            reasoning = exp.get('analysis_summary', {}).get('reasoning', '')
            if isinstance(reasoning, list):
                reasoning_text = "\n".join(reasoning)
            else:
                reasoning_text = reasoning
            if reasoning_text and reasoning_text.strip():
                agent_reasonings.append(reasoning_text.strip())
        
        if not agent_reasonings:
            return 0.0
        
        gt_embs = [self.semantic_model.encode(r, convert_to_numpy=True) for r in gt_reasonings]
        
        scores = []
        for ar in agent_reasonings:
            ar_emb = self.semantic_model.encode(ar, convert_to_numpy=True)
            sims = [
                float(np.dot(ar_emb, gt_emb) / (np.linalg.norm(ar_emb) * np.linalg.norm(gt_emb) + 1e-8))
                for gt_emb in gt_embs
            ]
            scores.append(max(sims) if sims else 0.0)
        
        return min(100.0, float(np.mean(scores) * 100))

    def _get_gt_reasonings(self) -> List[str]:
        """提取GT参考推理文本"""
        reasoning_process = self.gt.get('reasoning_process', {})
        ref_path = reasoning_process.get('reference_optimal_path', []) or []
        texts = []
        for step in ref_path:
            text = step.get('reasoning')
            if text and isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return texts
    
    def _lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # ========================================
    # 统计方法
    # ========================================
    
    def _count_insights(self, explorations: List[Dict]) -> int:
        """统计洞察数量"""
        count = 0
        for exp in explorations:
            count += len(exp.get('analysis_summary', {}).get('key_insights', []))
        return count
    
    def _get_tools_used(self, explorations: List[Dict]) -> List[str]:
        """获取使用的工具列表"""
        tools = []
        for exp in explorations:
            tool_exec = exp.get('tool_execution') or {}
            tool_name = tool_exec.get('tool_name')
            if tool_name:
                tools.append(tool_name)
        return tools


def format_evaluation_report(eval_result: Dict, task_id: str) -> str:
    """格式化评估报告"""
    report = []
    report.append("=" * 60)
    report.append(f"Benchmark评估报告 - {task_id}")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"📊 总分: {eval_result['total_score']}/100")
    report.append("")
    
    scores = eval_result['dimension_scores']
    weights = eval_result['weights']
    
    report.append("📈 各维度得分:")
    report.append(f"  1. 洞察质量 ({int(weights['insight_quality']*100)}%): {scores['insight_quality']}/100")
    report.append(f"  2. 推理过程 ({int(weights['reasoning_process']*100)}%): {scores['reasoning_process']}/100")
    report.append("")
    
    details = eval_result['details']
    report.append("📋 探索详情:")
    report.append(f"  - 探索轮次: {details['total_explorations']}")
    report.append(f"  - 发现洞察: {details['insights_found']}个")
    report.append(f"  - 使用工具: {', '.join(details['tools_used']) if details['tools_used'] else '无'}")
    report.append("")
    
    total = eval_result['total_score']
    if total >= 85:
        rating = "🌟 优秀"
    elif total >= 70:
        rating = "✅ 良好"
    elif total >= 60:
        rating = "⚠️ 及格"
    else:
        rating = "❌ 不及格"
    
    report.append(f"评级: {rating}")
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Benchmark评估器已就绪")
    print("支持字段: explorations[].analysis_summary.key_insights")
    print("支持字段: explorations[].analysis_summary.reasoning")
    print("支持字段: explorations[].tool_execution.tool_name")