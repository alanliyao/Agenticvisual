#!/usr/bin/env python3
"""
Benchmark Evaluation Runner

Runs evaluation on agent results against task configurations.
Supports both objective and subjective task types.

Usage:
    python run_evaluation.py --task task_path --result result_path   # evaluates all questions
    python run_evaluation.py --task task_path --result result_path -o out.json   # save full result to file
    python run_evaluation.py --batch --results-dir ./results --tasks-dir ./tasks
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.evaluators import UnifiedEvaluator
from benchmark.evaluators.unified_evaluator import UnifiedEvalResult


def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str):
    """Save JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def result_to_dict(result: UnifiedEvalResult) -> Dict:
    """Convert UnifiedEvalResult to dict"""
    d = {
        "task_type": result.task_type,
        "question_id": result.question_id,
        "scores": {
            "answer": result.answer_score,
            "tool": result.tool_score,
            "reasoning": result.reasoning_score,
            "tool_reasoning": result.tool_reasoning_score,
            "state": result.state_score,
            "total": result.total_score
        },
        "details": {
            "answer": result.answer_details,
            "tool": result.tool_details,
            "reasoning": result.reasoning_details,
            "state": result.state_details
        },
        "eval_weights": result.eval_weights,
        "agent_judge_triggered": result.agent_judge_triggered,
        "agent_judge_result": result.agent_judge_result
    }
    return d


def evaluate_single(task_path: str,
                   result_path: str,
                   question_idx: int = 0,
                   verbose: bool = True) -> UnifiedEvalResult:
    """
    Evaluate a single question's agent result against the task.
    
    Args:
        task_path: Path to task configuration JSON
        result_path: Path to agent result JSON ("results" list or single result)
        question_idx: Index of the question to evaluate
        verbose: Print detailed output
        
    Returns:
        UnifiedEvalResult for that question
    """
    task_config = load_json(task_path)
    data = load_json(result_path)
    
    if "results" in data and isinstance(data["results"], list):
        results_list = data["results"]
        agent_result = results_list[question_idx] if question_idx < len(results_list) else {}
    else:
        agent_result = data
    
    evaluator = UnifiedEvaluator()
    r = evaluator.evaluate_task(task_config, agent_result, question_idx)
    
    if verbose:
        _print_eval_result(r, Path(result_path).stem, question_idx)
    
    return r


def _print_eval_result(r: UnifiedEvalResult, stem: str = "", qidx: int = 0) -> None:
    """Print evaluation result with scores and Agent-as-Judge reasoning when present."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {stem} (Q{qidx})" if stem else f"Evaluation Results (Q{qidx})")
    print(f"{'='*60}")
    print(f"Task Type: {r.task_type}")
    print(f"Question ID: {r.question_id}")
    print(f"\nScores:")
    print(f"  Answer:          {r.answer_score:.3f}")
    print(f"  Tool:            {r.tool_score:.3f}")
    print(f"  Reasoning:       {r.reasoning_score:.3f}")
    print(f"  Tool+Reasoning:  {r.tool_reasoning_score:.3f}")
    print(f"  State:           {r.state_score:.3f}")
    print(f"  Total:           {r.total_score:.3f}")
    if r.agent_judge_triggered and r.agent_judge_result:
        j = r.agent_judge_result
        print(f"\nAgent-as-Judge:")
        print(f"  Verdict: {j.get('verdict', 'N/A')}")
        print(f"  Adjusted: {j.get('adjusted_score', j.get('final_score', 0)):.3f}")
        if j.get("reasoning"):
            print(f"  Reasoning: {j['reasoning']}")
    print(f"\nWeights: {r.eval_weights}")


def evaluate(task_path: str,
             result_path: str,
             verbose: bool = True) -> List[Dict]:
    """
    Evaluate all questions: loop over indices and call evaluate_single for each.
    
    Loads task and result to determine number of questions, then repeatedly
    calls evaluate_single(..., question_idx=i) and aggregates results.
    
    Args:
        task_path: Path to task configuration JSON
        result_path: Path to agent result JSON
        verbose: Print per-question lines and summary
        
    Returns:
        List of result dicts (one per question)
    """
    task_config = load_json(task_path)
    data = load_json(result_path)
    
    if "results" in data and isinstance(data["results"], list):
        results_list = data["results"]
    else:
        results_list = [data]
    
    questions = task_config.get("questions", [])
    n = min(len(results_list), len(questions)) if questions else len(results_list)
    if n == 0:
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation: {Path(result_path).stem}")
        print(f"{'='*60}")
        print(f"Evaluating {n} questions...\n")
    
    out: List[Dict] = []
    scores: Dict[str, List[float]] = {"answer": [], "tool": [], "reasoning": [], "tool_reasoning": [], "state": [], "total": []}
    
    for i in range(n):
        r = evaluate_single(task_path, result_path, question_idx=i, verbose=False)
        d = result_to_dict(r)
        d["question_idx"] = i
        d["qid"] = questions[i].get("qid", f"q{i}") if i < len(questions) else f"q{i}"
        out.append(d)
        for k in scores:
            scores[k].append(d["scores"][k])
        
        if verbose:
            sid = d["qid"]
            line = f"  {sid}: answer={d['scores']['answer']:.2f}, tool={d['scores']['tool']:.2f}, " \
                   f"reasoning={d['scores']['reasoning']:.2f}, state={d['scores']['state']:.2f}, " \
                   f"total={d['scores']['total']:.2f}"
            jr = d.get("agent_judge_result")
            if jr and jr.get("reasoning"):
                line += f"\n    [Judge] {jr.get('verdict', '')} | {jr.get('reasoning', '')[:120]}..."
            print(line)
    
    avg = {k: (sum(scores[k]) / len(scores[k])) if scores[k] else 0.0 for k in scores}
    
    if verbose:
        print(f"\n--- Summary ({n} questions) ---")
        print(f"  Avg Answer: {avg['answer']:.3f}  Avg Tool: {avg['tool']:.3f}  "
              f"Avg Reasoning: {avg['reasoning']:.3f}  Avg State: {avg['state']:.3f}  "
              f"Avg Total: {avg['total']:.3f}")
    
    return out


def evaluate_batch(task_dir: str,
                  results_dir: str,
                  output_dir: Optional[str] = None,
                  verbose: bool = True) -> List[Dict]:
    """
    Batch evaluate multiple results against tasks.
    
    Matches result files to tasks by naming convention:
    - Task: tasks/objective/clear+single/scatter_cars_001.json
    - Result: results/claude_mcp/scatter_cars_001.json
    
    Args:
        task_dir: Directory containing task configurations
        results_dir: Directory containing agent results
        output_dir: Directory to save evaluation outputs
        verbose: Print detailed output
        
    Returns:
        List of evaluation result dicts
    """
    evaluator = UnifiedEvaluator()
    all_results = []
    
    # Find all task files
    task_dir = Path(task_dir)
    task_files = list(task_dir.rglob("*.json"))
    
    # Find all result files
    results_dir = Path(results_dir)
    result_files = list(results_dir.rglob("*.json"))
    
    # Create result file lookup by stem
    result_lookup = {f.stem: f for f in result_files}
    
    if verbose:
        print(f"Found {len(task_files)} task files")
        print(f"Found {len(result_files)} result files")
    
    for task_file in task_files:
        task_stem = task_file.stem
        
        # Try to find matching result
        if task_stem in result_lookup:
            result_file = result_lookup[task_stem]
        else:
            # Try fuzzy matching
            matched = None
            for stem, rfile in result_lookup.items():
                if task_stem in stem or stem in task_stem:
                    matched = rfile
                    break
            if not matched:
                if verbose:
                    print(f"No result file found for task: {task_stem}")
                continue
            result_file = matched
        
        try:
            task_config = load_json(str(task_file))
            full_agent_result = load_json(str(result_file))
            
            # Handle nested "results" format from benchmark scripts
            results_list = None
            if "results" in full_agent_result and isinstance(full_agent_result["results"], list):
                results_list = full_agent_result["results"]
            
            # Evaluate each question in the task
            questions = task_config.get("questions", [])
            if isinstance(questions, list):
                for i in range(len(questions)):
                    # Extract the correct result for this question
                    if results_list and i < len(results_list):
                        agent_result = results_list[i]
                    else:
                        agent_result = full_agent_result
                    
                    result = evaluator.evaluate_task(task_config, agent_result, i)
                    result_dict = result_to_dict(result)
                    result_dict["task_file"] = str(task_file)
                    result_dict["result_file"] = str(result_file)
                    all_results.append(result_dict)
                    
                    if verbose:
                        print(f"[{task_stem}] Q{i}: total={result.total_score:.3f}")
            else:
                agent_result = results_list[0] if results_list else full_agent_result
                result = evaluator.evaluate_task(task_config, agent_result, 0)
                result_dict = result_to_dict(result)
                result_dict["task_file"] = str(task_file)
                result_dict["result_file"] = str(result_file)
                all_results.append(result_dict)
                
                if verbose:
                    print(f"[{task_stem}]: total={result.total_score:.3f}")
                    
        except Exception as e:
            if verbose:
                print(f"Error evaluating {task_stem}: {e}")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_results_{timestamp}.json"
        
        save_json({
            "timestamp": timestamp,
            "task_dir": str(task_dir),
            "results_dir": str(results_dir),
            "total_evaluations": len(all_results),
            "results": all_results
        }, str(output_file))
        
        if verbose:
            print(f"\nSaved results to: {output_file}")
    
    return all_results


def compute_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics from evaluation results"""
    if not results:
        return {"error": "No results to summarize"}
    
    # Aggregate by task type
    by_type = {}
    for r in results:
        task_type = r.get("task_type", "unknown")
        if task_type not in by_type:
            by_type[task_type] = []
        by_type[task_type].append(r)
    
    summary = {
        "total_evaluations": len(results),
        "by_task_type": {}
    }
    
    for task_type, type_results in by_type.items():
        scores = {
            "answer": [r["scores"]["answer"] for r in type_results],
            "tool": [r["scores"]["tool"] for r in type_results],
            "reasoning": [r["scores"]["reasoning"] for r in type_results],
            "tool_reasoning": [r["scores"]["tool_reasoning"] for r in type_results],
            "state": [r["scores"]["state"] for r in type_results],
            "total": [r["scores"]["total"] for r in type_results]
        }
        
        summary["by_task_type"][task_type] = {
            "count": len(type_results),
            "scores": {
                k: {
                    "mean": sum(v) / len(v),
                    "min": min(v),
                    "max": max(v)
                }
                for k, v in scores.items()
            }
        }
    
    return summary





def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation Runner")
    parser.add_argument("--task", type=str, help="Path to task configuration JSON")
    parser.add_argument("--result", type=str, help="Path to agent result JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Override default eval save path (default: <result_stem>_eval.json)")
    
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    parser.add_argument("--tasks-dir", type=str, help="Directory with task configs")
    parser.add_argument("--results-dir", type=str, help="Directory with agent results")
    parser.add_argument("--output-dir", type=str, help="Directory for evaluation outputs")
    
    parser.add_argument("--demo", action="store_true", help="Run demo test")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    verbose = not args.quiet
        
    if args.batch:
        if not args.tasks_dir or not args.results_dir:
            print("Error: --batch requires --tasks-dir and --results-dir")
            return
        
        results = evaluate_batch(
            args.tasks_dir,
            args.results_dir,
            args.output_dir,
            verbose=verbose
        )
        
        if results:
            summary = compute_summary(results)
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(json.dumps(summary, indent=2))
            
    elif args.task and args.result:
        results = evaluate(
            args.task,
            args.result,
            verbose=verbose
        )
        # Always save to default path if no -o specified; -o overrides
        result_path = Path(args.result)
        default_eval_path = result_path.parent / f"{result_path.stem}_eval.json"
        out_path = Path(args.output) if args.output else default_eval_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, str(out_path))
        if verbose:
            print(f"\nEvaluation results saved to: {out_path.resolve()}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()