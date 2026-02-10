"""
批量跑多模型基准脚本
按模型列表批量调用已有的 *_mcp_benchmark 脚本，支持：
- 任务列表文件或目录
- 模型列表选择
- 并发控制
- 重试
- 独立日志
"""

import argparse
import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# 默认模型与脚本映射（可在命令行覆盖模型列表）
DEFAULT_MODELS = [
    ("gpt", "gpt5_mcp_benchmark.py"),
    ("claude", "claude_mcp_benchmark.py"),
    ("gemini", "gemini_mcp_benchmark.py"),
    ("grok", "grok_mcp_benchmark.py"),
    ("qwen", "qwen_mcp_benchmark.py"),
    ("llama", "llama_mcp_benchmark.py"),
    ("mistral", "mistral_mcp_benchmark.py"),
]


def load_tasks(task_path: str) -> List[str]:
    p = Path(task_path)
    if p.is_file():
        return [str(p)]
    tasks = []
    for f in sorted(p.glob("*.json")):
        tasks.append(str(f))
    return tasks


async def run_one(cmd: List[str], log_path: Path, sem: asyncio.Semaphore, retries: int) -> Tuple[bool, str]:
    attempt = 0
    while attempt <= retries:
        attempt += 1
        async with sem:
            with open(log_path, "w") as lf:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=lf,
                    stderr=lf,
                )
                ret = await proc.wait()
            if ret == 0:
                return True, ""
            else:
                err_msg = f"exit {ret}, attempt {attempt}/{retries+1}"
                if attempt > retries:
                    return False, err_msg
                await asyncio.sleep(1.0)


async def main():
    parser = argparse.ArgumentParser(description="批量跑多模型基准脚本")
    parser.add_argument("tasks", help="任务文件或目录（目录下 *.json）")
    parser.add_argument(
        "--models",
        nargs="*",
        help="选择模型简称（默认全部）：gpt claude gemini grok qwen llama mistral"
    )
    parser.add_argument("--concurrency", type=int, default=3, help="并发数")
    parser.add_argument("--retries", type=int, default=1, help="失败重试次数")
    parser.add_argument("--python", default=sys.executable, help="python 解释器路径")
    parser.add_argument("--log-dir", default="benchmark/logs", help="日志目录")
    args = parser.parse_args()

    # 任务列表
    task_list = load_tasks(args.tasks)
    if not task_list:
        print("未找到任务文件")
        sys.exit(1)

    # 模型/脚本列表
    chosen = set(args.models) if args.models else None
    model_script_pairs = []
    for name, script in DEFAULT_MODELS:
        if chosen is None or name in chosen:
            model_script_pairs.append((name, script))
    if not model_script_pairs:
        print("未选择任何模型")
        sys.exit(1)

    # 准备日志目录
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    tasks_jobs = []

    for model_name, script in model_script_pairs:
        for task_file in task_list:
            task_base = Path(task_file).stem
            log_path = log_dir / f"{task_base}_{model_name}.log"
            cmd = [args.python, script, task_file]
            job = asyncio.create_task(run_one(cmd, log_path, sem, args.retries))
            tasks_jobs.append((model_name, task_file, log_path, job))

    # 执行
    results = []
    for model_name, task_file, log_path, job in tasks_jobs:
        ok, err = await job
        results.append((model_name, task_file, log_path, ok, err))

    # 汇总
    success = [r for r in results if r[3]]
    fail = [r for r in results if not r[3]]

    print("\n=== 运行完成 ===")
    print(f"总任务: {len(results)}, 成功: {len(success)}, 失败: {len(fail)}")
    if fail:
        print("\n失败清单：")
        for model_name, task_file, log_path, ok, err in fail:
            print(f"- 模型: {model_name}, 任务: {task_file}, 日志: {log_path}, 错误: {err}")


if __name__ == "__main__":
    asyncio.run(main())

