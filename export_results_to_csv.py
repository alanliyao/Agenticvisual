#!/usr/bin/env python3
"""
Smart CSV exporter for benchmark results.

Auto-detects run mode (single/multi model/task) and generates appropriate output.

Usage:
    python export_results_to_csv.py [--batch-dir PATH]

Output structure:
    benchmark/results/csv_export/{timestamp}/
        - Generated files based on detected mode
"""

import json
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def extract_category(task_id: str) -> str:
    """Extract category (cs/cm/vm/vs) from task_id like '07_bar_cm_01'"""
    parts = task_id.split('_')
    if len(parts) >= 3:
        category = parts[2]
        if category in ['cm', 'cs', 'vm', 'vs']:
            return category
    return 'unknown'


def parse_eval_result(eval_path: Path) -> Optional[Dict[str, Any]]:
    """Parse eval_result.json and extract relevant fields."""
    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get('results'):
            return None
        
        result = data['results'][0]
        task_id = data.get('task_id', '')
        model = data.get('model', '')
        
        scores = result.get('scores', {})
        agent_judge = result.get('agent_judge_result')
        
        # Get scores
        answer_score = scores.get('answer', 0)
        tool_score = scores.get('tool', 0)
        reasoning_score = scores.get('reasoning', 0)
        tool_reasoning_score = scores.get('tool_reasoning', 0)
        state_score = scores.get('state', 0)
        total_score = scores.get('total', 0)
        
        row = {
            'task_id': task_id,
            'model': model,
            'category': extract_category(task_id),
            'answer': answer_score,
            'tool': tool_score,
            'reasoning': reasoning_score,
            'tool_reason': tool_reasoning_score,
            'state': state_score,
        }
        
        # Three total scores: original, llm_adjusted, final
        if agent_judge and result.get('agent_judge_triggered'):
            # LLM was triggered - we have all three scores
            dimension_scores = agent_judge.get('dimension_scores', {})
            
            # Original score (before LLM adjustment)
            original_total = agent_judge.get('original_score', total_score)
            # LLM calculated score
            llm_total = agent_judge.get('adjusted_total_score') or agent_judge.get('adjusted_score', '')
            # Final score (after verdict applied)
            final_total = agent_judge.get('final_score', total_score)
            
            row['original_total'] = original_total
            row['llm_total'] = llm_total
            row['final_total'] = final_total
            
            # LLM dimension scores
            if dimension_scores:
                row['llm_answer'] = dimension_scores.get('answer', '')
                row['llm_tool'] = dimension_scores.get('tool', '')
                row['llm_reasoning'] = dimension_scores.get('reasoning', '')
                row['llm_state'] = dimension_scores.get('state', '')
            else:
                row['llm_answer'] = ''
                row['llm_tool'] = ''
                row['llm_reasoning'] = ''
                row['llm_state'] = ''
            
            row['llm_tool_r'] = dimension_scores.get('reasoning', '') if dimension_scores else ''
            row['llm_reason'] = agent_judge.get('reasoning', '')
        else:
            # No LLM triggered - all three are the same (original rule-based score)
            row['original_total'] = total_score
            row['llm_total'] = ''  # No LLM evaluation
            row['final_total'] = total_score
            
            row['llm_answer'] = ''
            row['llm_tool'] = ''
            row['llm_reasoning'] = ''
            row['llm_tool_r'] = ''
            row['llm_state'] = ''
            row['llm_reason'] = ''
        
        return row
    except Exception as e:
        print(f"Error parsing {eval_path}: {e}")
        return None


def format_float(value, decimals=2):
    """Format float to specified decimals, handle empty/None."""
    if value is None or value == '':
        return ''
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def detect_mode(all_rows: List[Dict]) -> Tuple[str, List[str], List[str]]:
    """Detect run mode based on data distribution."""
    models = sorted(set(r['model'] for r in all_rows))
    tasks = sorted(set(r['task_id'] for r in all_rows))
    
    model_count = len(models)
    task_count = len(tasks)
    
    if model_count == 1 and task_count == 1:
        return "single_model_single_task", models, tasks
    elif model_count == 1 and task_count > 1:
        return "single_model_multi_task", models, tasks
    elif model_count > 1 and task_count == 1:
        return "multi_model_single_task", models, tasks
    else:
        return "multi_model_multi_task", models, tasks


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print ASCII table."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def export_single_model_single_task(rows: List[Dict], output_dir: Path):
    """Export for single model, single task mode."""
    row = rows[0]
    model = row['model']
    task = row['task_id']
    
    # Save detailed CSV
    output_file = output_dir / f"{model}_{task}_detailed.csv"
    fieldnames = ['dimension', 'original_score', 'llm_score', 'change']
    
    dimensions = ['answer', 'tool', 'reasoning', 'state', 'original_total', 'llm_total', 'final_total']
    csv_rows = []
    for dim in dimensions:
        orig = row.get(dim, '')
        llm = row.get(f'llm_{dim}', '') if 'llm' in dim else ''
        change = ''
        if orig != '' and llm != '':
            change = format_float(float(llm) - float(orig))
        csv_rows.append({
            'dimension': dim,
            'original_score': format_float(orig),
            'llm_score': format_float(llm),
            'change': change
        })
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"  Saved: {output_file.name}")
    
    # Terminal output
    print(f"\nTask: {task} | Model: {model}")
    print("-" * 50)
    print(f"{'Dimension':<12} {'Original':>10} {'LLM Adj':>10} {'Change':>10}")
    print("-" * 50)
    for r in csv_rows:
        print(f"{r['dimension']:<12} {r['original_score']:>10} {r['llm_score']:>10} {r['change']:>10}")
    print("-" * 50)
    
    # LLM Reason
    if row.get('llm_reason'):
        print(f"\nLLM Reason:")
        print(f"  {row['llm_reason'][:200]}...")


def export_single_model_multi_task(rows: List[Dict], output_dir: Path):
    """Export for single model, multiple tasks mode."""
    model = rows[0]['model']
    
    # Save CSV
    output_file = output_dir / f"{model}_results.csv"
    fieldnames = [
        'task_id', 'category', 'answer', 'tool', 'reasoning', 'tool_reason',
        'state', 'original_total', 'llm_total', 'final_total', 
        'llm_answer', 'llm_tool', 'llm_reasoning',
        'llm_tool_r', 'llm_state', 'llm_reason'
    ]
    
    rows_sorted = sorted(rows, key=lambda x: x['task_id'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            row_copy = {k: format_float(v) if k not in ['task_id', 'category', 'llm_reason'] else v 
                       for k, v in row.items() if k != 'model'}
            writer.writerow(row_copy)
    
    print(f"  Saved: {output_file.name} ({len(rows)} tasks)")
    
    # Terminal summary
    print(f"\nModel: {model}")
    print(f"Tasks: {len(rows)}")
    print("-" * 60)
    
    # Category breakdown
    cat_scores = defaultdict(list)
    for row in rows:
        cat_scores[row['category']].append(float(row.get('final_total', 0) or 0))
    
    table_data = []
    for cat in ['cs', 'cm', 'vm', 'vs']:
        if cat in cat_scores:
            scores = cat_scores[cat]
            avg = sum(scores) / len(scores)
            table_data.append([cat.upper(), str(len(scores)), format_float(avg)])
    
    print_table(['Category', 'Count', 'Avg Total'], table_data, "Performance by Category")
    
    # Overall stats
    all_scores = [float(r.get('original_total', 0) or 0) for r in rows]
    final_scores = [float(r.get('final_total', 0) or 0) for r in rows]
    llm_scores = [float(r.get('llm_total', 0) or 0) for r in rows if r.get('llm_total') not in ['', None]]
    
    print(f"Original Avg: {format_float(sum(all_scores)/len(all_scores))}")
    print(f"Final Avg: {format_float(sum(final_scores)/len(final_scores))}")
    if llm_scores:
        print(f"LLM Calculated Avg: {format_float(sum(llm_scores)/len(llm_scores))}")
    print(f"Tasks with LLM Review: {len(llm_scores)}/{len(rows)}")


def export_multi_model_single_task(rows: List[Dict], output_dir: Path):
    """Export for multiple models, single task mode."""
    task = rows[0]['task_id']
    
    # Save comparison CSV
    output_file = output_dir / f"{task}_comparison.csv"
    fieldnames = ['model', 'answer', 'tool', 'reasoning', 'state', 'original_total', 
                  'llm_total', 'final_total', 'llm_adjusted']
    
    rows_sorted = sorted(rows, key=lambda x: float(x.get('final_total', 0) or 0), reverse=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({
                'model': row['model'],
                'answer': format_float(row.get('answer', '')),
                'tool': format_float(row.get('tool', '')),
                'reasoning': format_float(row.get('reasoning', '')),
                'state': format_float(row.get('state', '')),
                'original_total': format_float(row.get('original_total', '')),
                'llm_total': format_float(row.get('llm_total', '')),
                'final_total': format_float(row.get('final_total', '')),
                'llm_adjusted': 'Yes' if row.get('llm_total') not in ['', None] else 'No'
            })
    
    print(f"  Saved: {output_file.name}")
    
    # Terminal comparison table
    print(f"\nTask: {task}")
    print(f"Models compared: {len(rows)}")
    print("-" * 80)
    
    table_data = []
    for row in rows_sorted:
        table_data.append([
            row['model'][:15],
            format_float(row.get('answer', '')),
            format_float(row.get('tool', '')),
            format_float(row.get('state', '')),
            format_float(row.get('original_total', '')),
            format_float(row.get('final_total', '')),
            'Yes' if row.get('llm_total') not in ['', None] else 'No'
        ])
    
    print_table(['Model', 'Answer', 'Tool', 'State', 'Original', 'Final', 'LLM Adj?'], 
                table_data, "Model Comparison")


def export_multi_model_multi_task(rows: List[Dict], models: List[str], output_dir: Path):
    """Export for multiple models, multiple tasks mode (full benchmark)."""
    
    # Individual model CSVs
    print("\n  [1/4] Exporting per-model CSVs...")
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        model_rows.sort(key=lambda x: x['task_id'])
        
        output_file = output_dir / f'{model}_results.csv'
        fieldnames = [
            'task_id', 'category', 'answer', 'tool', 'reasoning', 'tool_reason',
            'state', 'original_total', 'llm_total', 'final_total',
            'llm_answer', 'llm_tool', 'llm_reasoning',
            'llm_tool_r', 'llm_state', 'llm_reason'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in model_rows:
                row_copy = {k: format_float(v) if k not in ['task_id', 'category', 'llm_reason'] else v 
                           for k, v in row.items() if k != 'model'}
                writer.writerow(row_copy)
        
        print(f"    {model}: {len(model_rows)} tasks")
    
    # Master CSV
    print("\n  [2/4] Exporting combined CSV...")
    output_file = output_dir / 'all_models_results.csv'
    fieldnames = [
        'model', 'task_id', 'category', 'answer', 'tool', 'reasoning', 'tool_reason',
        'state', 'original_total', 'llm_total', 'final_total',
        'llm_answer', 'llm_tool', 'llm_reasoning',
        'llm_tool_r', 'llm_state', 'llm_reason'
    ]
    
    rows_sorted = sorted(rows, key=lambda x: (x['model'], x['task_id']))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            row_copy = {k: format_float(v) if k not in ['model', 'task_id', 'category', 'llm_reason'] else v 
                       for k, v in row.items()}
            writer.writerow(row_copy)
    
    print(f"    Combined: {len(rows)} rows")
    
    # Category stats
    print("\n  [3/4] Exporting category statistics...")
    stats = defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'original_sum': 0, 'final_sum': 0, 'llm_count': 0, 'llm_total_sum': 0
    }))
    
    for row in rows:
        s = stats[row['model']][row['category']]
        s['count'] += 1
        s['original_sum'] += float(row.get('original_total', 0) or 0)
        s['final_sum'] += float(row.get('final_total', 0) or 0)
        llm_total = row.get('llm_total')
        if llm_total not in ['', None]:
            s['llm_count'] += 1
            s['llm_total_sum'] += float(llm_total)
    
    output_file = output_dir / 'category_stats.csv'
    fieldnames = ['model', 'category', 'count', 'avg_original', 'avg_final', 'llm_count', 'avg_llm_total']
    
    stat_rows = []
    for model in models:
        for cat in ['cs', 'cm', 'vm', 'vs']:
            s = stats[model][cat]
            if s['count'] > 0:
                stat_rows.append({
                    'model': model,
                    'category': cat,
                    'count': s['count'],
                    'avg_original': format_float(s['original_sum'] / s['count']),
                    'avg_final': format_float(s['final_sum'] / s['count']),
                    'llm_count': s['llm_count'],
                    'avg_llm_total': format_float(s['llm_total_sum'] / s['llm_count']) if s['llm_count'] > 0 else ''
                })
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stat_rows)
    
    print(f"    {len(stat_rows)} entries")
    
    # Summary Markdown
    print("\n  [4/4] Generating summary report...")
    summary_file = output_dir / 'summary.md'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Results Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total tasks: {len(rows)}\n")
        f.write(f"- Models: {', '.join(models)}\n\n")
        
        # Model ranking
        f.write("## Model Ranking (by Avg Final Score)\n\n")
        model_scores = defaultdict(list)
        for row in rows:
            model_scores[row['model']].append(float(row.get('final_total', 0) or 0))
        
        ranking = [(m, sum(scores)/len(scores), len(scores)) for m, scores in model_scores.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        f.write("| Rank | Model | Avg Total | Tasks |\n")
        f.write("|:----:|:------|:---------:|:-----:|\n")
        for i, (model, avg_score, count) in enumerate(ranking, 1):
            f.write(f"| {i} | {model} | {format_float(avg_score)} | {count} |\n")
        
        f.write("\n")
        
        # Category breakdown
        f.write("## Category Breakdown\n\n")
        for cat in ['cs', 'cm', 'vm', 'vs']:
            cat_data = [r for r in stat_rows if r['category'] == cat]
            if cat_data:
                f.write(f"### {cat.upper()}\n\n")
                f.write("| Model | Avg Original | Avg Final | Tasks |\n")
                f.write("|:------|:------------:|:---------:|:-----:|\n")
                for r in cat_data:
                    f.write(f"| {r['model']} | {r['avg_original']} | {r['avg_final']} | {r['count']} |\n")
                f.write("\n")
        
        # LLM adjusted
        adjusted = [r for r in rows if r.get('llm_total') not in ['', None]]
        f.write(f"## LLM Adjusted Tasks\n\n")
        f.write(f"Tasks with Agent-as-Judge: {len(adjusted)} / {len(rows)} ({format_float(len(adjusted)/len(rows)*100)}%)\n\n")
        
        if adjusted:
            f.write("| Model | Task | Original | Adjusted | Change |\n")
            f.write("|:------|:-----|:--------:|:--------:|:------:|\n")
            for row in adjusted[:20]:
                orig = float(row.get('original_total', 0) or 0)
                adj = float(row.get('llm_total', 0) or 0)
                change = adj - orig
                change_str = f"+{format_float(change)}" if change > 0 else format_float(change)
                f.write(f"| {row['model']} | {row['task_id']} | {format_float(orig)} | {format_float(adj)} | {change_str} |\n")
            if len(adjusted) > 20:
                f.write(f"\n*Showing 20 of {len(adjusted)} adjusted tasks*\n")
    
    print(f"    Saved: summary.md")
    
    # Terminal summary
    print(f"\n{'=' * 60}")
    print(f"Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"Total tasks: {len(rows)}")
    print(f"Models: {len(models)}")
    print(f"\nModel Rankings:")
    for i, (model, avg, _) in enumerate(ranking, 1):
        print(f"  {i}. {model:<20} {format_float(avg)}")
    print(f"\nTasks with LLM review: {len(adjusted)}/{len(rows)} ({format_float(len(adjusted)/len(rows)*100)}%)")


def main():
    parser = argparse.ArgumentParser(description='Export benchmark results to CSV (auto-detects mode)')
    parser.add_argument('--batch-dir', type=str, help='Path to batch results directory (default: latest)')
    args = parser.parse_args()
    
    # Determine batch directory
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
    else:
        batch_base = Path('benchmark/results/batch')
        batch_dirs = sorted(batch_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not batch_dirs:
            print("No batch results found!")
            return
        batch_dir = batch_dirs[0]
    
    if not batch_dir.exists():
        print(f"Batch directory not found: {batch_dir}")
        return
    
    timestamp = batch_dir.name
    
    # Create output directory
    output_base = Path('benchmark/results/csv_export')
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("Scanning Results")
    print(f"Source: {batch_dir}")
    
    # Scan all results
    eval_files = list(batch_dir.rglob('*/eval_result.json'))
    print(f"Found: {len(eval_files)} result files")
    
    # Parse all results
    all_rows = []
    for eval_file in eval_files:
        row = parse_eval_result(eval_file)
        if row:
            all_rows.append(row)
    
    if not all_rows:
        print("No valid results found!")
        return
    
    # Detect mode
    mode, models, tasks = detect_mode(all_rows)
    
    print(f"\nDetected mode: {mode}")
    print(f"  Models: {len(models)} ({', '.join(models)})")
    print(f"  Tasks:  {len(tasks)}")
    
    print_header("Exporting Results")
    print(f"Output: {output_dir}\n")
    
    # Export based on mode
    if mode == "single_model_single_task":
        export_single_model_single_task(all_rows, output_dir)
    elif mode == "single_model_multi_task":
        export_single_model_multi_task(all_rows, output_dir)
    elif mode == "multi_model_single_task":
        export_multi_model_single_task(all_rows, output_dir)
    else:
        export_multi_model_multi_task(all_rows, models, output_dir)
    
    print_header("Export Complete")
    print(f"Files saved to: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
