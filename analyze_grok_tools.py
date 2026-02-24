#!/usr/bin/env python3
"""分析 Grok tool 评分为 0 的原因"""

import json
import os
from pathlib import Path

results_dir = 'benchmark/results/batch/20260223_092752'

print("=" * 80)
print("Grok Tool 评分分析")
print("=" * 80)

for root, dirs, files in os.walk(results_dir):
    if 'grok' in root.lower():
        result_path = os.path.join(root, 'result.json')
        eval_path = os.path.join(root, 'eval_result.json')
        
        if not os.path.exists(result_path) or not os.path.exists(eval_path):
            continue
            
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            with open(eval_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            task_id = eval_data.get('task_id', 'unknown')
            first_result = eval_data['results'][0]
            scores = first_result['scores']
            
            print(f"\n【{task_id}】")
            print(f"  Tool Score: {scores['tool']}")
            print(f"  Tool Details 键: {list(first_result.get('tool_details', {}).keys()) if 'tool_details' in first_result else 'Not found'}")
            
            # 检查 agent 输出
            output = result.get('output', '')
            tool_calls = result.get('tool_calls', [])
            
            print(f"  Agent Output (前200字): {output[:200].replace(chr(10), ' ')}")
            print(f"  Tool Calls 数量: {len(tool_calls)}")
            
            if tool_calls:
                for tc in tool_calls[:5]:
                    tool_name = tc.get('tool', 'unknown') if isinstance(tc, dict) else str(tc)
                    print(f"    - {tool_name}")
            else:
                print("    ⚠️ 没有工具调用！")
                
        except Exception as e:
            import traceback
            print(f"Error processing {root}: {e}")
            traceback.print_exc()
