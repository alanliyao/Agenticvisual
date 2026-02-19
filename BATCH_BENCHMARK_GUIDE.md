# 批量评测使用指南 (Batch Benchmark Guide)

## 📋 简介

`run_all_benchmarks.py` 是 AgenticVisual 项目的批量评测脚本，用于自动化运行多个可视化分析任务，支持多模型对比和详细的分数汇总。

### 主要特性
- ✅ **批量任务执行**：自动遍历目录下所有测试任务
- ✅ **多模型支持**：支持 7 个主流 VLM 模型（GPT、Claude、Gemini、Qwen 等）
- ✅ **任务筛选**：支持按名称或通配符筛选特定任务
- ✅ **并发控制**：可调节并发数提高效率
- ✅ **自动评估**：集成统一评估器（Unified Evaluator）自动打分
- ✅ **详细汇总**：生成多维度统计报告

---

## 🚀 快速开始

### 1. 环境准备

确保已安装依赖并配置 API Key：

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 OpenRouter API Key（必须）
$env:OPENROUTER_API_KEY="sk-or-v1-..."  # Windows PowerShell
export OPENROUTER_API_KEY="sk-or-v1-..." # Linux/Mac
```

### 2. 基本运行

```bash
# 最简单的用法：跑所有任务的所有模型
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark/

# 推荐：先小批量测试
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark/ --models qwen --task-filter 07_bar_cm_01
```

---

## 🎛️ 参数详解

### 位置参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `tasks` | 任务文件路径或目录 | `benchmark/tasks/` 或 `tasks/01.json` |

### 模型选择

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--models` | 选择要跑的模型（空格分隔） | 全部7个模型 | `--models qwen` 或 `--models qwen claude gpt` |

**支持的模型**：
- `gpt` - GPT-4/GPT-5
- `claude` - Claude 3.5/4
- `gemini` - Google Gemini
- `grok` - xAI Grok
- `qwen` - 阿里通义千问（推荐，国内可用）
- `llama` - Meta Llama
- `mistral` - Mistral AI

### 任务筛选

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task-filter` | 按文件名前缀筛选（空格分隔） | `--task-filter 07_bar 34_scatter` |
| `--task-pattern` | 按通配符模式筛选 | `--task-pattern "*_cm_*.json"` |

**筛选示例**：
```bash
# 只跑特定几个任务
--task-filter 07_bar_cm_01 34_scatter_cm_01 40_heatmap_cm_01

# 只跑明确多步骤任务（cm = clear multi）
--task-pattern "*_cm_*.json"

# 只跑散点图任务
--task-pattern "*scatter*.json"

# 只跑3开头的任务
--task-pattern "3*.json"
```

### 执行控制

| 参数 | 说明 | 默认值 | 建议 |
|------|------|--------|------|
| `--concurrency` | 并发数（同时跑几个任务） | 1 | 测试时用1，正式跑用2-3 |
| `--retries` | 失败重试次数 | 1 | 网络不稳时设为2-3 |

**并发建议**：
- `--concurrency 1`：串行执行，最安全，适合调试
- `--concurrency 2`：速度翻倍，适合大多数情况
- `--concurrency 3`：速度最快，可能有MCP冲突风险

### 输出配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 结果输出目录 | `benchmark/results/batch` |
| `--log-dir` | 运行日志目录 | `benchmark/logs/batch` |

---

## 💡 使用示例

### 示例 1：快速测试（推荐首次使用）

```powershell
# Windows PowerShell
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen `
    --task-filter 07_bar_cm_01 34_scatter_cm_01 `
    --concurrency 1
```

**说明**：只跑2个任务，1个模型，串行执行，用于验证流程是否正常。

### 示例 2：单模型全量评测

```powershell
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen `
    --concurrency 3
```

**说明**：跑全部34个任务，只用Qwen模型，同时跑3个加速。

### 示例 3：特定题型批量测试

```powershell
# 只跑明确多步骤任务（cm = clear multi）
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen `
    --task-pattern "*_cm_*.json" `
    --concurrency 2

# 只跑明确单步骤任务（cs = clear single）
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen claude `
    --task-pattern "*_cs_*.json" `
    --concurrency 2
```

### 示例 4：多模型横向对比

```powershell
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen claude gemini `
    --task-filter 07_bar_cm_01 34_scatter_cm_01 40_heatmap_cm_01 `
    --concurrency 2
```

**说明**：3个模型 × 3个任务 = 9个作业，对比不同模型在相同任务上的表现。

### 示例 5：全模型完整评测（最全面）

```powershell
python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --concurrency 3
```

**说明**：7个模型 × 34个任务 = 238个作业，需要数小时完成。

---

## 📊 输出结果说明

### 实时输出

运行时会显示进度：
```
[加载] 从 benchmark_annotation_system/annotated_task/benchmark/ 加载任务...
[加载] 找到 34 个任务
[筛选] 按名称过滤: 2/34 个任务
[筛选] 最终任务数: 2
  - 07_bar_cm_01.json
  - 34_scatter_cm_01.json

[模型] 将使用: qwen
[输出] 结果目录: benchmark\results\batch\20260220_014200
[输出] 日志目录: benchmark\logs\batch\20260220_014200

[启动] 共 2 个作业 (任务×模型)
[配置] 并发数: 1, 重试: 1
============================================================
[1/2] qwen/07_bar_cm_01 ... OK [score=0.85]
[2/2] qwen/34_scatter_cm_01 ... OK [score=1.00]
```

### 最终汇总

```
============================================================
 批量跑分完成
============================================================
总任务: 2, 成功: 2, 失败: 0

[按模型统计]
  qwen        : 成功 2, 失败 0
                平均分: answer=0.92, tool=0.98, total=0.93

[总体平均分]
  answer      : 0.925 (n=2)
  tool        : 0.980 (n=2)
  reasoning   : 0.950 (n=2)
  state       : 0.900 (n=2)
  total       : 0.935 (n=2)

[失败清单]
  (无)
============================================================

[保存] 详细汇总已保存到: benchmark\results\batch\20260220_014200\summary.json
```

### 生成的文件结构

```
benchmark/results/batch/20260220_014200/
├── 07_bar_cm_01_qwen/
│   ├── result.json              # 模型输出结果
│   └── eval_result.json         # 评估分数详情
├── 34_scatter_cm_01_qwen/
│   ├── result.json
│   └── eval_result.json
└── summary.json                 # 完整汇总统计

benchmark/logs/batch/20260220_014200/
├── 07_bar_cm_01_qwen.log        # 运行日志
└── 34_scatter_cm_01_qwen.log
```

### summary.json 结构

```json
{
  "timestamp": "2026-02-20T01:42:00",
  "total": 2,
  "success": 2,
  "fail": 0,
  "by_model": {
    "qwen": {
      "success": 2,
      "fail": 0,
      "tasks": [...]
    }
  },
  "scores": {
    "overall": {
      "total": {"mean": 0.935, "count": 2}
    },
    "by_model": {
      "qwen": {
        "total": {"mean": 0.935, "count": 2}
      }
    }
  }
}
```

---

## ⚠️ 常见问题

### Q1: 提示 "未找到任务文件"
**原因**：路径错误或目录下没有 `.json` 文件
**解决**：
```bash
# 检查路径
ls benchmark_annotation_system/annotated_task/benchmark/

# 确认有 .json 文件
```

### Q2: 提示 "Provider returned error" 或 403
**原因**：API Key 无效或模型地区受限
**解决**：
- 检查 `OPENROUTER_API_KEY` 是否设置正确
- 某些模型（如 Claude、GPT）在国内可能受限，建议用 `qwen`

### Q3: 任务失败（FAIL）
**排查步骤**：
1. 查看日志文件：`benchmark/logs/batch/时间戳/任务名_模型.log`
2. 常见原因：
   - MCP 工具调用错误
   - API 超时
   - 格式解析错误

### Q4: 如何只跑部分任务测试？
**解决**：使用 `--task-filter` 或 `--task-pattern`
```bash
# 只跑前3个任务
--task-filter 07_bar_cm_01 07_bar_cs_01 07_bar_vm_01

# 或按模式筛选
--task-pattern "07_*.json"
```

### Q5: 并发数设置多少合适？
**建议**：
- 调试/开发：`--concurrency 1`（串行，易排查）
- 正式跑分：`--concurrency 2` 或 `3`（平衡速度和稳定性）
- 不要超过 3，可能引发 MCP 服务器冲突

---

## 🔧 高级用法

### 结合其他工具分析结果

```bash
# 1. 批量跑完
python run_all_benchmarks.py ... --models qwen

# 2. 查看 summary.json
cat benchmark/results/batch/最新时间戳/summary.json

# 3. 用 Python 分析
python -c "import json; d=json.load(open('summary.json')); print(d['scores']['overall'])"
```

### 自动化脚本示例

创建 `run_batch.ps1`（Windows）：
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

python run_all_benchmarks.py `
    benchmark_annotation_system/annotated_task/benchmark/ `
    --models qwen `
    --concurrency 3 `
    --output-dir results/qwen_full

Write-Host "批量跑分完成！"
```

---

## 📞 支持

如有问题，请检查：
1. 日志文件：`benchmark/logs/batch/时间戳/*.log`
2. 结果文件：`benchmark/results/batch/时间戳/summary.json`
3. 原始任务文件：`benchmark_annotation_system/annotated_task/benchmark/*.json`

---

**最后更新**：2026-02-20
**版本**：v2.0（重构版）
