"""
批量跑多模型基准脚本（重构版）
调用统一的 run_benchmark.py，支持任务筛选和详细汇总

特性：
- 任务筛选：--task-filter 指定特定任务，--task-pattern 通配符匹配
- 统一调用：使用 run_benchmark.py --eval 进行标准化评估
- 结果汇总：自动读取 eval_result.json，生成平均分统计
- 并发控制：支持同时跑多个任务
- 失败重试：自动重试失败的作业
"""

import argparse
import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

# 导入 benchmark.config 以触发 .env 加载（必须在其他导入之前）
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark.config import *  # 这会触发 .env 加载
except Exception as e:
    print(f"[警告] 加载 benchmark.config 失败: {e}")

# 支持的模型列表（不再映射到脚本，统一用 run_benchmark.py）
DEFAULT_MODELS = ["gpt", "claude", "gemini", "grok", "qwen", "llama", "mistral"]


def load_tasks(task_path: str) -> List[str]:
    """加载任务文件列表"""
    p = Path(task_path)
    if p.is_file():
        return [str(p)]
    tasks = []
    for f in sorted(p.glob("*.json")):
        tasks.append(str(f))
    return tasks


def filter_tasks(
    tasks: List[str],
    task_filter: List[str] = None,
    task_pattern: str = None
) -> List[str]:
    """筛选任务
    
    Args:
        tasks: 原始任务列表
        task_filter: 文件名前缀列表（如：["07_bar", "34_scatter"]）
        task_pattern: glob 通配符（如："*_cm_*.json"）
    """
    filtered = tasks
    
    if task_filter:
        # 支持多个前缀，用逗号或空格分隔
        allowed = set()
        for f in task_filter:
            allowed.update(f.replace(',', ' ').split())
        
        # 使用前缀匹配：文件名以允许的前缀开头
        def matches_filter(filename: str) -> bool:
            name = Path(filename).stem  # 07_bar_cm_01
            for prefix in allowed:
                if name.startswith(prefix) or filename.startswith(prefix):
                    return True
            return False
        
        filtered = [t for t in filtered if matches_filter(t)]
        print(f"[筛选] 按名称过滤: {len(filtered)}/{len(tasks)} 个任务")
    
    if task_pattern:
        import fnmatch
        filtered = [
            t for t in filtered 
            if fnmatch.fnmatch(Path(t).name, task_pattern)
        ]
        print(f"[筛选] 按模式过滤: {len(filtered)}/{len(tasks)} 个任务")
    
    return filtered


async def run_one(
    cmd: List[str],
    log_path: Path,
    sem: asyncio.Semaphore,
    retries: int
) -> Tuple[bool, str, Dict[str, Any]]:
    """运行单个任务
    
    Returns:
        (success, error_msg, eval_result)
    """
    attempt = 0
    eval_result = {}
    
    # 准备环境变量（传递 OPENROUTER_API_KEY）
    env = os.environ.copy()
    
    # 确保 OPENROUTER_API_KEY 被传递（Windows 兼容性修复）
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
    else:
        print("  [警告] 未找到 OPENROUTER_API_KEY 环境变量")
        return False, "OPENROUTER_API_KEY not set", {}
    
    while attempt <= retries:
        attempt += 1
        async with sem:
            # 创建日志目录
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=lf,
                    stderr=lf,
                    env=env,  # 传递环境变量
                )
                ret = await proc.wait()
            
            if ret == 0:
                # 尝试读取 eval_result.json
                try:
                    # 从 cmd 中找到 output-dir 参数
                    output_dir = None
                    for i, arg in enumerate(cmd):
                        if arg == "--output-dir" and i + 1 < len(cmd):
                            output_dir = Path(cmd[i + 1])
                            break
                    
                    if output_dir:
                        eval_file = output_dir / "eval_result.json"
                        if eval_file.exists():
                            with open(eval_file, "r", encoding="utf-8") as f:
                                eval_result = json.load(f)
                except Exception as e:
                    print(f"  [警告] 读取 eval_result 失败: {e}")
                
                return True, "", eval_result
            else:
                err_msg = f"exit {ret}, attempt {attempt}/{retries+1}"
                if attempt > retries:
                    return False, err_msg, {}
                await asyncio.sleep(1.0)
    
    return False, "max retries exceeded", {}


def calculate_summary(results: List[Tuple]) -> Dict[str, Any]:
    """计算汇总统计
    
    Args:
        results: [(model_name, task_file, log_path, success, error, eval_result), ...]
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success": 0,
        "fail": 0,
        "by_model": {},
        "by_task": {},
        "scores": {
            "overall": {"answer": [], "tool": [], "reasoning": [], "state": [], "total": []},
            "by_model": {}
        }
    }
    
    for model_name, task_file, log_path, success, error, eval_result in results:
        task_name = Path(task_file).stem
        
        # 统计成功/失败
        if success:
            summary["success"] += 1
        else:
            summary["fail"] += 1
        
        # 按模型统计
        if model_name not in summary["by_model"]:
            summary["by_model"][model_name] = {"success": 0, "fail": 0, "tasks": []}
        summary["by_model"][model_name]["success" if success else "fail"] += 1
        summary["by_model"][model_name]["tasks"].append({
            "task": task_name,
            "success": success,
            "error": error if not success else None,
            "scores": (eval_result.get("results", [{}])[0].get("scores", {}) if eval_result.get("results") else eval_result.get("scores", {})) if success else {}
        })
        
        # 按任务统计
        if task_name not in summary["by_task"]:
            summary["by_task"][task_name] = {"success": 0, "fail": 0, "models": []}
        summary["by_task"][task_name]["success" if success else "fail"] += 1
        summary["by_task"][task_name]["models"].append({
            "model": model_name,
            "success": success,
            "scores": (eval_result.get("results", [{}])[0].get("scores", {}) if eval_result.get("results") else eval_result.get("scores", {})) if success else {}
        })
        
        # 收集分数
        if success and eval_result:
            # 处理嵌套格式 {"results": [{"scores": {...}}]}
            if "results" in eval_result and eval_result["results"]:
                scores = eval_result["results"][0].get("scores", {})
            else:
                scores = eval_result.get("scores", {})
            for key in ["answer", "tool", "reasoning", "state", "total"]:
                if key in scores:
                    summary["scores"]["overall"][key].append(scores[key])
                    
                    # 按模型收集
                    if model_name not in summary["scores"]["by_model"]:
                        summary["scores"]["by_model"][model_name] = {
                            "answer": [], "tool": [], "reasoning": [], "state": [], "total": []
                        }
                    summary["scores"]["by_model"][model_name][key].append(scores[key])
    
    # 计算平均分
    for category in ["overall", "by_model"]:
        target = summary["scores"][category]
        if category == "by_model":
            for model in target:
                for key in target[model]:
                    if target[model][key]:
                        target[model][key] = {
                            "mean": sum(target[model][key]) / len(target[model][key]),
                            "count": len(target[model][key])
                        }
        else:
            for key in target:
                if target[key]:
                    target[key] = {
                        "mean": sum(target[key]) / len(target[key]),
                        "count": len(target[key])
                    }
    
    return summary


def print_summary(summary: Dict[str, Any]):
    """打印汇总结果"""
    print("\n" + "="*60)
    print(" 批量跑分完成")
    print("="*60)
    print(f"总任务: {summary['total']}, 成功: {summary['success']}, 失败: {summary['fail']}")
    
    # 按模型统计
    print("\n[按模型统计]")
    for model, stats in summary["by_model"].items():
        print(f"  {model:12s}: 成功 {stats['success']:3d}, 失败 {stats['fail']:3d}")
        
        # 显示该模型的平均分
        if model in summary["scores"]["by_model"]:
            scores = summary["scores"]["by_model"][model]
            if scores.get("total"):
                print(f"                平均分: answer={scores.get('answer', {}).get('mean', 0):.2f}, "
                      f"tool={scores.get('tool', {}).get('mean', 0):.2f}, "
                      f"total={scores['total']['mean']:.2f}")
    
    # 总体平均分
    print("\n[总体平均分]")
    overall = summary["scores"]["overall"]
    for key in ["answer", "tool", "reasoning", "state", "total"]:
        if key in overall and overall[key]:
            print(f"  {key:12s}: {overall[key]['mean']:.3f} (n={overall[key]['count']})")
    
    # 失败清单
    if summary["fail"] > 0:
        print("\n[失败清单]")
        for model, stats in summary["by_model"].items():
            for task in stats["tasks"]:
                if not task["success"]:
                    print(f"  - {model}/{task['task']}: {task.get('error', 'unknown')}")
    
    print("="*60)


async def main():
    parser = argparse.ArgumentParser(
        description="批量跑多模型基准脚本（重构版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 跑所有任务的所有模型
  python run_all_benchmarks.py benchmark/tasks/
  
  # 只跑 Qwen 和 Claude
  python run_all_benchmarks.py benchmark/tasks/ --models qwen claude
  
  # 只跑特定几个任务
  python run_all_benchmarks.py benchmark/tasks/ --models qwen --task-filter 07_bar 34_scatter
  
  # 按通配符筛选（只跑明确多步骤任务）
  python run_all_benchmarks.py benchmark/tasks/ --models qwen --task-pattern "*_cm_*.json"
        """
    )
    parser.add_argument("tasks", help="任务文件或目录（目录下 *.json）")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=DEFAULT_MODELS,
        help=f"选择模型（可选：{' '.join(DEFAULT_MODELS)}），默认全部"
    )
    parser.add_argument(
        "--task-filter",
        nargs="*",
        help="只跑特定任务（文件名前缀，如：07_bar 34_scatter）"
    )
    parser.add_argument(
        "--task-pattern",
        help="任务文件名通配符（如：*_cm_*.json）"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发数（默认1，建议不超过3以避免MCP冲突）"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="失败重试次数（默认1）"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/results/batch",
        help="结果输出目录"
    )
    parser.add_argument(
        "--log-dir",
        default="benchmark/logs/batch",
        help="运行日志目录"
    )
    
    args = parser.parse_args()
    
    # 1. 加载任务
    print(f"[加载] 从 {args.tasks} 加载任务...")
    task_list = load_tasks(args.tasks)
    if not task_list:
        print("[错误] 未找到任务文件")
        sys.exit(1)
    print(f"[加载] 找到 {len(task_list)} 个任务")
    
    # 2. 筛选任务
    task_list = filter_tasks(task_list, args.task_filter, args.task_pattern)
    if not task_list:
        print("[错误] 筛选后没有剩余任务")
        sys.exit(1)
    print(f"[筛选] 最终任务数: {len(task_list)}")
    for t in task_list[:5]:  # 只显示前5个
        print(f"  - {Path(t).name}")
    if len(task_list) > 5:
        print(f"  ... 还有 {len(task_list)-5} 个")
    
    # 3. 选择模型
    chosen_models = args.models if args.models else DEFAULT_MODELS
    print(f"\n[模型] 将使用: {', '.join(chosen_models)}")
    
    # 4. 准备目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / timestamp
    log_base = Path(args.log_dir) / timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    log_base.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 结果目录: {output_base}")
    print(f"[输出] 日志目录: {log_base}")
    
    # 5. 构造任务列表
    sem = asyncio.Semaphore(args.concurrency)
    jobs = []
    
    for model in chosen_models:
        for task_file in task_list:
            task_name = Path(task_file).stem
            task_output_dir = output_base / f"{task_name}_{model}"
            log_path = log_base / f"{task_name}_{model}.log"
            
            # 构造命令：调用 run_benchmark.py
            cmd = [
                sys.executable,
                "benchmark/run_benchmark.py",
                "--task", task_file,
                "--model", model,
                "--eval",
                "--output-dir", str(task_output_dir),
                "--no-save-views"  # 批量跑时节省空间
            ]
            
            job = asyncio.create_task(
                run_one(cmd, log_path, sem, args.retries)
            )
            jobs.append((model, task_file, log_path, job))
    
    total_jobs = len(jobs)
    print(f"\n[启动] 共 {total_jobs} 个作业 (任务×模型)")
    print(f"[配置] 并发数: {args.concurrency}, 重试: {args.retries}")
    print("="*60)
    
    # 6. 执行
    results = []
    completed = 0
    
    for model, task_file, log_path, job in jobs:
        task_name = Path(task_file).stem
        print(f"[{completed+1}/{total_jobs}] {model}/{task_name} ... ", end="", flush=True)
        
        ok, err, eval_result = await job
        results.append((model, task_file, log_path, ok, err, eval_result))
        completed += 1
        
        if ok:
            score_str = ""
            if eval_result and "scores" in eval_result:
                total_score = eval_result["scores"].get("total", 0)
                score_str = f" [score={total_score:.2f}]"
            print(f"OK{score_str}")
        else:
            print(f"FAIL [{err}]")
    
    # 7. 汇总
    summary = calculate_summary(results)
    print_summary(summary)
    
    # 8. 保存汇总
    summary_file = output_base / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 详细汇总已保存到: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
