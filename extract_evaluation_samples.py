#!/usr/bin/env python3
"""
从真实 benchmark 结果中提取 NLI/LLM 评估样本。
生成待人工标注的数据文件。

支持两种模式：
1. 生成 60 样本（原始 5 模型）
2. 扩展到 100 样本（保留已标注的 60 + 新增 40）
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ground Truth insights (English)
GT_INSIGHTS = [
    {
        "id": "gt_001",
        "content": "The scatter plot shows a clear negative correlation: higher horsepower leads to lower MPG, with a calculated correlation coefficient of -0.778."
    },
    {
        "id": "gt_002", 
        "content": "Japanese cars are characterized by low horsepower (50-120 HP) and high fuel efficiency (25-46 MPG), with densely distributed data points."
    },
    {
        "id": "gt_003",
        "content": "American cars feature high horsepower (100-230 HP) and low fuel efficiency (10-25 MPG), prioritizing power performance."
    },
    {
        "id": "gt_004",
        "content": "European cars occupy the middle ground with 60-140 HP and 20-35 MPG, balancing performance and efficiency."
    },
    {
        "id": "gt_005",
        "content": "The 60-120 horsepower range shows significant data point overlap, requiring zooming for detailed analysis."
    },
    {
        "id": "gt_006",
        "content": "In the 60-120 HP and 25-40 MPG region, Japanese cars clearly dominate with compact distribution."
    },
    {
        "id": "gt_007",
        "content": "Some Japanese cars maintain 30+ MPG in the 80-95 HP range, showing superior engine efficiency."
    },
    {
        "id": "gt_008",
        "content": "European cars have wide distribution in the dense region, with diversified product strategies."
    },
    {
        "id": "gt_009",
        "content": "In the high-efficiency zone, American cars are scattered and less efficient than Japanese and European cars."
    },
    {
        "id": "gt_010",
        "content": "Japanese and American cars show the most distinct separation at opposite ends of the performance-efficiency spectrum."
    },
    {
        "id": "gt_011",
        "content": "European and Japanese cars overlap in the low-to-mid horsepower range."
    },
    {
        "id": "gt_012",
        "content": "Some ultra-high horsepower American cars (200+ HP) have extremely low fuel efficiency (<15 MPG), representing muscle cars."
    }
]

# ============================================================================
# Model Agent Insights (原始 5 模型，手动整理的英文翻译)
# ============================================================================

MODEL_INSIGHTS = {
    "claude-sonnet-4": [
        "The dataset contains 392 cars from three origins: USA (245), Japan (79), and Europe (68), with uneven sample distribution.",
        "Horsepower ranges from 46-230 HP, MPG from 9.0-46.6, showing a clear inverse relationship between performance and efficiency.",
        "The chart shows different origin cars exhibit distinct distribution patterns in the scatter plot.",
        "Horsepower and MPG show strong negative correlation (r=-0.778), confirming the universal performance-efficiency tradeoff.",
        "The correlation coefficient of -0.778 indicates about 60% of MPG variation can be explained by horsepower.",
        "The car market shows three distinct technical clusters: high-efficiency type (70hp/31mpg), high-performance type (164hp/14mpg), and mainstream balanced type (100hp/22mpg).",
        "Cluster analysis reveals different design strategies: 134 high-efficiency cars, 90 high-performance cars, 168 mainstream cars.",
        "The high-efficiency region (46-90hp, 25-47mpg) contains 144 cars, likely concentrating small-displacement efficient models from Japan and Europe.",
        "Region selection highlights the concentrated distribution pattern of high-efficiency models, contrasting with American high-horsepower models.",
        "The high-performance region (130-230hp, 8-20mpg) contains 94 cars, dominated by high-performance models.",
        "The contrast between two extreme regions demonstrates market technical differentiation: 144 cars in high-efficiency zone vs 94 cars in high-performance zone."
    ],
    
    "gpt-5": [
        "Three tradeoff bands quantified: efficiency band C1 has 112 samples (32%), compromise band C2 has 154 samples (43%), high-performance band C3 has 89 samples (25%).",
        "Threshold effect: when horsepower exceeds about 130 HP, samples almost all fall into MPG<20 low-efficiency zone, showing significant fuel penalty for high power.",
        "In the middle band (HP 80-135, MPG 18-30) there are 154 cars (39.3%), indicating this is the mainstream performance-efficiency balance point.",
        "At the same horsepower level, fuel consumption difference is significant (8-10 MPG spread): European/Japanese samples more often at upper edge (25-29 MPG), American samples concentrated at lower edge (18-23 MPG)."
    ],
    
    "gemini-2.5-flash": [
        "Horsepower and MPG have very strong negative correlation (r=-0.778), quantifying the fundamental performance-efficiency tradeoff.",
        "Origin technical characteristics differ significantly: American cars average HP (119 HP) much higher than Japan (80 HP) and Europe (81 HP), with lowest average fuel efficiency (20.08 MPG).",
        "Japanese cars are efficiency champions: highest average fuel efficiency (30.45 MPG), significantly higher than Europe (27.60 MPG) and USA.",
        "American cars dominate high-performance market: all cars with HP over 133 HP are from USA.",
        "Weight correlates with efficiency: American cars average weight (3372 lbs) far exceeds Japan and Europe, explaining lower fuel efficiency.",
        "In high-efficiency/low-HP market (40-100 HP, 30-50 MPG), Japanese and European cars absolutely dominate with 89 cars.",
        "Japanese cars excel at top fuel efficiency: Japanese and European models dominate among MPG>40 vehicles.",
        "American cars weak in high-efficiency domain: in the selected high-efficiency region, American car count and density significantly lower than Japan and Europe."
    ],
    
    "llama-4-maverick": [
        "Horsepower and MPG have a strong negative correlation, indicating performance-efficiency tradeoff.",
        "Different origins show different characteristics: American cars tend toward high HP/low MPG, while European and Japanese cars balance performance and efficiency.",
        "Cluster analysis identifies three main car categories: low HP/high MPG, moderate performance, and high performance/low efficiency.",
        "In the HP 150-200 and MPG 15-25 region, mainly high-performance cars, with cars from all three origins distributed."
    ],
    
    "qwen3-vl-235b": [
        "Overall dataset shows strong negative correlation (Pearson r=-0.778), indicating performance improvement typically sacrifices fuel efficiency.",
        "Cluster 0 (blue) represents low HP, high MPG vehicles, concentrated in Europe origin, showing preference for economy and efficiency-oriented design.",
        "Cluster 1 (orange) represents high HP, low MPG vehicles, mainly corresponding to USA origin, reflecting pursuit of high performance sacrificing fuel efficiency.",
        "Cluster 2 (green) represents moderate HP and moderate MPG balanced vehicles, mainly from Japan, embodying the compromise between performance and efficiency.",
        "In low HP (40-100) and high MPG (25-50) region, mainly dominated by cluster 0 (blue), highly overlapping with European origin.",
        "In high HP (150-240) and low MPG (5-15) region, mainly dominated by cluster 1 (orange), highly overlapping with USA origin, confirming American cars pursue high performance accepting low fuel efficiency."
    ]
}


def extract_insights_from_result_json(result_path: Path) -> Optional[List[str]]:
    """
    从 benchmark result.json 中提取 insights。
    优先使用 summary.all_insights，否则从 explorations 中收集。
    """
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 优先使用 summary.all_insights
        if "summary" in data and "all_insights" in data["summary"]:
            insights = data["summary"]["all_insights"]
            # 去重
            unique_insights = list(dict.fromkeys(insights))
            if unique_insights:
                return unique_insights
        
        # 否则从 explorations 中收集
        insights = []
        for exp in data.get("explorations", []):
            summary = exp.get("analysis_summary", {})
            for ins in summary.get("key_insights", []):
                if ins and ins not in insights:
                    insights.append(ins)
        
        return insights if insights else None
    except Exception as e:
        print(f"  ❌ 读取失败 {result_path}: {e}")
        return None


def find_scatter_results(base_dir: Path) -> Dict[str, List[str]]:
    """
    从 benchmark/results 中找到所有散点图任务的结果，提取 insights。
    返回 {model_name: [insights]}
    """
    results = {}
    
    # 搜索所有子目录
    for mcp_dir in base_dir.iterdir():
        if not mcp_dir.is_dir():
            continue
        
        for task_dir in mcp_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            # 只处理 cars_multivariate 或 scatter 相关任务
            if "cars_multivariate" not in task_dir.name and "scatter" not in task_dir.name:
                continue
            
            result_file = task_dir / "result.json"
            if not result_file.exists():
                continue
            
            insights = extract_insights_from_result_json(result_file)
            if not insights:
                continue
            
            # 提取模型名称
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                model_name = data.get("model", task_dir.name)
            except:
                model_name = task_dir.name
            
            # 简化模型名称
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
            
            if model_name not in results:
                results[model_name] = insights
                print(f"  ✅ 提取 {model_name}: {len(insights)} insights")
    
    return results


def generate_60_samples():
    """生成待人工标注的数据文件 - 60 样本版本"""
    
    samples = []
    sample_id = 0
    
    for model_name, agent_insights in MODEL_INSIGHTS.items():
        for gt in GT_INSIGHTS:
            sample_id += 1
            sample_entry = {
                "id": f"sample_{sample_id:03d}",
                "model": model_name,
                "agent_insights": agent_insights,
                "gt_id": gt["id"],
                "gt_insight": gt["content"],
                "human_label": None  # 待标注
            }
            samples.append(sample_entry)
    
    # 构建输出
    output = {
        "metadata": {
            "description": "NLI/LLM Evaluation Samples - 60 samples from 5 models",
            "task": "cars_multivariate_002",
            "evaluation_logic": "Determine if each GT insight is entailed by all agent insights",
            "total_samples": len(samples),
            "models": list(MODEL_INSIGHTS.keys()),
            "gt_insights_count": len(GT_INSIGHTS),
            "label_options": ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
            "label_meaning": {
                "ENTAILMENT": "GT insight is clearly covered/entailed by agent insights (3)",
                "NEUTRAL": "GT insight is partially mentioned or implied (2)", 
                "CONTRADICTION": "GT insight contradicts or is not mentioned (1)"
            }
        },
        "samples": samples
    }
    
    # 保存
    output_path = Path(__file__).parent / "evaluation_samples_60.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(samples)} samples from {len(MODEL_INSIGHTS)} models")
    print(f"📄 Saved to: {output_path}")
    
    return output


def extend_to_100_samples():
    """
    扩展到 100 样本：
    1. 读取已标注的 60 样本
    2. 从 benchmark/results 中提取新模型的 insights
    3. 合并生成 100 样本文件
    """
    base_dir = Path(__file__).parent
    
    # 1. 读取已标注的 60 样本
    existing_file = base_dir / "evaluation_samples_60.json"
    if not existing_file.exists():
        print("❌ 未找到 evaluation_samples_60.json，请先运行 generate_60_samples()")
        return None
    
    with open(existing_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    
    existing_samples = existing_data["samples"]
    existing_models = set(s["model"] for s in existing_samples)
    print(f"📂 已加载 {len(existing_samples)} 个已标注样本")
    print(f"   已有模型: {', '.join(existing_models)}")
    
    # 2. 从 benchmark/results 中提取新模型
    print("\n🔍 搜索新的散点图任务结果...")
    benchmark_dir = base_dir / "benchmark" / "results"
    new_model_insights = find_scatter_results(benchmark_dir)
    
    # 过滤掉已有的模型
    new_models = {}
    for model, insights in new_model_insights.items():
        # 检查是否已存在（模糊匹配）
        is_existing = any(
            existing.lower() in model.lower() or model.lower() in existing.lower()
            for existing in existing_models
        )
        if not is_existing:
            new_models[model] = insights
    
    print(f"\n📊 新增模型: {list(new_models.keys())}")
    
    # 3. 生成新样本
    new_samples = []
    sample_id = len(existing_samples)
    
    for model_name, agent_insights in new_models.items():
        for gt in GT_INSIGHTS:
            sample_id += 1
            sample_entry = {
                "id": f"sample_{sample_id:03d}",
                "model": model_name,
                "agent_insights": agent_insights,
                "gt_id": gt["id"],
                "gt_insight": gt["content"],
                "human_label": None  # 待标注
            }
            new_samples.append(sample_entry)
    
    # 4. 合并样本
    all_samples = existing_samples + new_samples
    all_models = list(existing_models) + list(new_models.keys())
    
    # 构建输出
    output = {
        "metadata": {
            "description": f"NLI/LLM Evaluation Samples - {len(all_samples)} samples from {len(all_models)} models",
            "task": "cars_multivariate_002",
            "evaluation_logic": "Determine if each GT insight is entailed by all agent insights",
            "total_samples": len(all_samples),
            "models": all_models,
            "gt_insights_count": len(GT_INSIGHTS),
            "label_options": ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
            "label_meaning": {
                "ENTAILMENT": "GT insight is clearly covered/entailed by agent insights (3)",
                "NEUTRAL": "GT insight is partially mentioned or implied (2)", 
                "CONTRADICTION": "GT insight contradicts or is not mentioned (1)"
            },
            "annotated_count": len(existing_samples),
            "pending_annotation_count": len(new_samples)
        },
        "samples": all_samples
    }
    
    # 保存
    output_path = base_dir / "evaluation_samples_100.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 生成 {len(all_samples)} 个样本（{len(existing_samples)} 已标注 + {len(new_samples)} 待标注）")
    print(f"📄 保存到: {output_path}")
    
    # 打印待标注样本的统计
    print("\n" + "="*80)
    print("待标注样本（新增）:")
    print("="*80)
    for model in new_models.keys():
        count = sum(1 for s in new_samples if s["model"] == model)
        print(f"  {model}: {count} 个样本")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="提取评估样本")
    parser.add_argument("--extend", action="store_true", help="扩展到 100 样本")
    args = parser.parse_args()
    
    if args.extend:
        extend_to_100_samples()
    else:
        generate_60_samples()
