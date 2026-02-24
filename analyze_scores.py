#!/usr/bin/env python3
"""分析任务分数分布，找出容易触发 LLM 二次评分的任务"""

import json
import os
from collections import defaultdict

# 收集所有评分数据
task_scores = defaultdict(list)
task_llm_triggered = defaultdict(int)

results_dir = 'benchmark/results/batch'
for root, dirs, files in os.walk(results_dir):
    if 'eval_result.json' in files:
        path = os.path.join(root, 'eval_result.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task_id = data.get('task_id', '')
            results = data.get('results', [])
            
            if results:
                scores = results[0].get('scores', {})
                total = scores.get('total')
                if total is not None:
                    task_scores[task_id].append(total)
                    
                # 检查是否触发了 LLM 评分
                agent_judge_triggered = results[0].get('agent_judge_triggered', False)
                if agent_judge_triggered:
                    task_llm_triggered[task_id] += 1
        except Exception as e:
            pass

# 分析每个任务的分数分布
print('任务分数分析 (寻找容易产生 [0.4, 0.7] 分数的任务):')
print('=' * 80)

task_analysis = []
for task_id, scores in sorted(task_scores.items()):
    if len(scores) >= 2:  # 至少2个模型跑过
        avg = sum(scores) / len(scores)
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s
        
        # 统计落在 [0.4, 0.7] 区间内的比例
        in_range = sum(1 for s in scores if 0.4 <= s <= 0.7)
        in_range_ratio = in_range / len(scores)
        
        # 判断任务类型
        parts = task_id.split('_')
        category = parts[2] if len(parts) > 2 else 'unknown'
        
        task_analysis.append({
            'task_id': task_id,
            'category': category,
            'count': len(scores),
            'avg': avg,
            'min': min_s,
            'max': max_s,
            'range': range_s,
            'in_range': in_range,
            'in_range_ratio': in_range_ratio,
            'llm_triggered': task_llm_triggered.get(task_id, 0)
        })

# 按类别分组
by_category = defaultdict(list)
for t in task_analysis:
    by_category[t['category']].append(t)

print("\n[按类别推荐任务] (* = 高概率触发 LLM 评分):\n")

for cat in ['cm', 'cs', 'vm', 'vs']:
    if cat in by_category:
        print(f'[{cat.upper()} 类别] ({len(by_category[cat])} 个任务有数据)')
        # 排序：优先显示 in_range_ratio 高、llm_triggered 多的任务
        tasks = sorted(by_category[cat], 
                      key=lambda x: (-x['in_range_ratio'], -x['llm_triggered'], -x['range']))
        for t in tasks[:6]:
            indicator = '**' if t['in_range_ratio'] > 0.5 or t['llm_triggered'] >= 2 else \
                       '*' if t['in_range_ratio'] > 0.2 or t['llm_triggered'] >= 1 else '  '
            llm_info = f", LLM触发:{t['llm_triggered']}次" if t['llm_triggered'] > 0 else ""
            print(f"  {indicator} {t['task_id']}: 平均{t['avg']:.2f}, 范围[{t['min']:.2f}, {t['max']:.2f}], "
                  f"中区间比例{t['in_range_ratio']:.0%}{llm_info}")
        print()

# 找出最佳组合
print("\n[推荐最小任务组合] (覆盖所有类别 + 高 LLM 触发率):\n")

selected_tasks = []
for cat in ['cm', 'cs', 'vm', 'vs']:
    if cat in by_category:
        # 选择标准：in_range_ratio 最高，且有 llm_triggered 的优先
        tasks = sorted(by_category[cat], 
                      key=lambda x: (-x['in_range_ratio'] * 10 - x['llm_triggered'] * 5 - x['range']))
        if tasks:
            best = tasks[0]
            selected_tasks.append(best)
            print(f"  {cat.upper()}: {best['task_id']} (中区间比例: {best['in_range_ratio']:.0%}, "
                  f"历史 LLM 触发: {best['llm_triggered']}次)")

print(f"\n>> 共 {len(selected_tasks)} 个任务 x 7 个模型 = {len(selected_tasks) * 7} 次调用")
print(f"   预计 LLM 二次评分触发率: {sum(t['in_range_ratio'] for t in selected_tasks)/len(selected_tasks):.0%}")

# 额外推荐一些高波动性任务
print("\n[额外推荐的高波动性任务] (适合验证 LLM 评分差异):")
all_tasks = sorted(task_analysis, key=lambda x: -x['range'])
for t in all_tasks[:5]:
    if t not in selected_tasks:
        print(f"  - {t['task_id']} ({t['category']}): 分数范围 [{t['min']:.2f}, {t['max']:.2f}], "
              f"差异 {t['range']:.2f}")
