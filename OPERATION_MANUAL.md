# AgenticVisual Benchmark 系统操作说明书

## 📖 目录
1. [项目概述](#一项目概述)
2. [环境准备](#二环境准备)
3. [配置文件](#三配置文件)
4. [运行 Benchmark](#四运行-benchmark)
5. [CSV 结果导出](#五csv-结果导出)
6. [结果分析](#六结果分析)
7. [故障排查](#七故障排查)
8. [附录](#八附录)

---

## 一、项目概述

### 1.1 系统功能
AgenticVisual 是一个多模型可视化分析能力评测系统，支持：
- **7 个主流模型**：Qwen, GPT-4o, Claude, Mistral, Llama, Gemini, Grok
- **4 类任务**：CS (Clear-Single), CM (Clear-Multi), VM (Vague-Multi), VS (Vague-Single)
- **5 维评分**：Answer, Tool, Reasoning, State, Total
- **Agent-as-Judge**：LLM 二次评估机制

### 1.2 工作流程
```
准备任务 → 运行 Benchmark → 自动评估 → 导出 CSV → 分析结果
```

### 1.3 关键文件
| 文件 | 用途 |
|------|------|
| `run_all_benchmarks.py` | 批量运行评测 |
| `export_results_to_csv.py` | 结果导出工具 |
| `benchmark/config.py` | 模型配置 |
| `.env` | API Key 配置 |

---

## 二、环境准备

### 2.1 必要环境
- **Python**: 3.10+
- **操作系统**: Windows 10/11 或 Linux
- **网络**: 可访问 OpenRouter API

### 2.2 安装依赖
```bash
pip install -r requirements.txt
```

核心依赖：
- `openai>=1.0.0` - API 客户端
- `mcp>=1.26.0` - MCP 工具协议
- `python-dotenv` - 环境变量管理

### 2.3 配置 API Key
编辑 `.env` 文件：
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**获取方式**：https://openrouter.ai/settings/credits

---

## 三、配置文件

### 3.1 模型配置 (`benchmark/config.py`)
```python
MODELS = {
    "qwen": ModelConfig(
        name="Qwen 3 VL 235B",
        model="qwen/qwen3-vl-235b-a22b-instruct",
        tool_choice_format="string",
    ),
    "gpt": ModelConfig(
        name="GPT-4o", 
        model="openai/gpt-4o",
        tool_choice_format="string",
    ),
    # ... 其他模型
}
```

**常用配置项**：
- `max_tokens`: 最大输出长度（默认 2000）
- `temperature`: 温度（默认 0.0）
- `timeout`: API 超时（默认 180 秒）

### 3.2 任务文件结构
任务文件位于：`benchmark_annotation_system/annotated_task/`

子文件夹：
- `benchmark/` - 精选 34 个任务（推荐用于快速评测）
- `clear+multi/` - 100+ CM 任务
- `clear+single/` - 100+ CS 任务  
- `vague+multi/` - 90+ VM 任务
- `vague+single/` - 90+ VS 任务

---

## 四、运行 Benchmark

### 4.1 基础命令格式
```bash
python run_all_benchmarks.py <任务目录> [选项]
```

### 4.2 运行模式详解

#### 模式 A：完整 Benchmark（推荐）
**场景**：评估所有模型在标准任务集上的表现

```bash
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --concurrency 7
```

**参数说明**：
- `--concurrency 7`: 7 个并发任务（根据 API 额度调整）
- 预计耗时：15-30 分钟（34 任务 × 7 模型 = 238 个作业）

**输出示例**：
```
[启动] 共 238 个作业 (任务×模型)
[1/238] qwen/07_bar_cm_01 ... OK
[2/238] qwen/07_bar_cs_01 ... OK
...
[保存] 详细汇总已保存到: benchmark\results\batch\20260222_222818\summary.json
```

---

#### 模式 B：单模型多任务
**场景**：测试单个模型在多个任务上的表现

```bash
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --models qwen --concurrency 2
```

**可选模型**：`qwen`, `gpt`, `claude`, `mistral`, `llama`, `gemini`, `grok`

---

#### 模式 C：单模型单任务（快速测试）
**场景**：验证特定任务或调试

```bash
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --models qwen --task-filter 34_scatter_cm_01 --concurrency 1
```

**参数说明**：
- `--task-filter 34_scatter_cm_01`: 只匹配包含该字符串的任务
- `--concurrency 1`: 单线程最稳定

---

#### 模式 D：多模型单任务（横向对比）
**场景**：对比多个模型在同一任务上的表现

```bash
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --models qwen gpt claude --task-filter 34_scatter_cm_01 --concurrency 3
```

---

### 4.3 高级参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--models` | 指定模型列表 | `--models qwen gpt` |
| `--task-filter` | 任务名过滤（部分匹配） | `--task-filter scatter` |
| `--task-pattern` | 正则匹配 | `--task-pattern "*scatter*"` |
| `--concurrency` | 并发数（1-10） | `--concurrency 3` |
| `--retries` | 失败重试次数 | `--retries 2` |
| `--no-eval` | 只运行不评估（调试用） | `--no-eval` |

---

### 4.4 运行中监控

**正常状态**：
- 终端显示 `[x/总] 模型/任务 ... OK`
- `benchmark/logs/batch/{时间戳}/` 生成日志文件

**异常处理**：
- `exit 1`: 任务失败，会自动重试（默认 2 次）
- `API error 402`: API 额度不足，需充值
- `Empty response`: 模型无响应，可能是网络问题

---

## 五、CSV 结果导出

### 5.1 基础导出
**默认导出最新批次**：
```bash
python export_results_to_csv.py
```

**自动识别模式**：
- 单模型单任务 → 生成详细对比 CSV
- 单模型多任务 → 生成模型完整报告
- 多模型单任务 → 生成模型对比表
- 多模型多任务 → 生成完整 Benchmark 报告

---

### 5.2 指定历史批次
查看所有批次：
```bash
ls benchmark/results/batch/
# 输出: 20260221_233554  20260222_085916  20260222_222818
```

导出特定批次：
```bash
python export_results_to_csv.py --batch-dir benchmark/results/batch/20260221_233554
```

---

### 5.3 输出文件说明

导出后会生成时间戳子文件夹：`benchmark/results/csv_export/{时间戳}/`

#### 多模型多任务输出（完整 Benchmark）
```
20260222_222818/
├── qwen_results.csv              # 各模型详细得分
├── gpt_results.csv
├── claude_results.csv
├── mistral_results.csv
├── llama_results.csv
├── gemini_results.csv
├── grok_results.csv
├── all_models_results.csv        # 所有模型汇总（231 行）
├── category_stats.csv            # CS/CM/VM/VS 分类统计（28 行）
└── summary.md                    # 美观的 Markdown 报告
```

#### CSV 列说明
| 列名 | 说明 | 示例 |
|------|------|------|
| `task_id` | 任务标识 | `34_scatter_cm_01` |
| `category` | 任务类型 | `cm` (Clear-Multi) |
| `answer` | 原始答案得分 | `0.73` |
| `tool` | 原始工具得分 | `0.00` |
| `reasoning` | 原始推理得分 | `0.80` |
| `state` | 原始状态得分 | `0.67` |
| `total` | 原始总分 | `0.12` |
| `llm_answer` | LLM 调整后的 answer | `0.05` |
| `llm_tool` | LLM 调整后的 tool | `0.00` |
| `llm_reasoning` | LLM 调整后的 reasoning | `0.25` |
| `llm_state` | LLM 调整后的 state | `0.10` |
| `llm_total` | LLM 调整后的总分 | `0.12` |
| `llm_reason` | LLM 调整原因 | `"The agent provides no substantive..."` |

**注意**：`llm_*` 列为空表示该任务未触发 Agent-as-Judge（总分不在 0.4-0.7 区间）

---

## 六、结果分析

### 6.1 查看 Markdown 摘要
```bash
code benchmark/results/csv_export/20260222_222818/summary.md
```

**包含内容**：
- 模型排名（按平均分）
- 分类统计（CS/CM/VM/VS）
- LLM 调整任务列表
- 详细评分分布

### 6.2 Excel 分析技巧

**筛选特定模型**：
```
筛选 model 列 = "qwen"
```

**查看 LLM 调整过的任务**：
```
筛选 llm_total 列不为空
```

**按任务类型分组**：
```
数据透视表: 行=category, 列=model, 值=平均值 of total
```

### 6.3 关键指标解读

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| Answer | 答案准确性 | 0-1 |
| Tool | 工具调用准确性 | 0-1 |
| Reasoning | 推理过程质量 | 0-1 |
| State | 图表状态保持 | 0-1 |
| Total | 加权总分 | 0-1 |

**评分标准**：
- `> 0.7`: 优秀
- `0.4 - 0.7`: 一般（可能触发 LLM 二次评估）
- `< 0.4`: 较差

---

## 七、故障排查

### 7.1 API 402 错误（额度不足）
**症状**：
```
API error: Error code: 402 - This request requires more credits...
```

**解决**：
1. 访问 https://openrouter.ai/settings/credits 充值
2. 降低并发数：`--concurrency 1`
3. 减少模型数量：`--models qwen`（先跑单个模型测试）

---

### 7.2 工具调用失败（Tool=0）
**症状**：大量任务的 `tool` 得分为 0

**原因**：
- API 额度不足导致模型未响应
- 模型调用了错误工具（与 Ground Truth 不匹配）

**诊断**：
```bash
# 查看日志
cat benchmark/logs/batch/{时间戳}/{任务}_{模型}.log
```

---

### 7.3 任务失败（exit 1）
**症状**：`success: false` 或 `exit 1`

**解决**：
- 检查 eval_result.json 中的 error 字段
- 查看详细日志
- 可能是 evaluator 解析错误，检查任务文件格式

---

### 7.4 CSV 导出失败
**症状**：`No valid results found!`

**检查**：
```bash
# 确认结果文件夹存在
ls benchmark/results/batch/{时间戳}/

# 确认有 eval_result.json 文件
find benchmark/results/batch/{时间戳}/ -name "eval_result.json"
```

---

## 八、附录

### 8.1 快速命令参考

```bash
# 1. 快速测试（1 模型 1 任务）
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --models qwen --task-filter 34_scatter_cm_01 --concurrency 1

# 2. 单模型完整测试
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --models qwen --concurrency 2

# 3. 完整 Benchmark（7 模型）
python run_all_benchmarks.py benchmark_annotation_system/annotated_task/benchmark --concurrency 7

# 4. 导出 CSV（自动识别最新批次）
python export_results_to_csv.py

# 5. 导出指定批次
python export_results_to_csv.py --batch-dir benchmark/results/batch/20260222_222818
```

### 8.2 文件路径速查

| 内容 | 路径 |
|------|------|
| 原始结果 | `benchmark/results/batch/{时间戳}/{任务}_{模型}/eval_result.json` |
| 汇总 JSON | `benchmark/results/batch/{时间戳}/summary.json` |
| 日志文件 | `benchmark/logs/batch/{时间戳}/{任务}_{模型}.log` |
| CSV 导出 | `benchmark/results/csv_export/{时间戳}/` |
| 模型配置 | `benchmark/config.py` |
| 任务文件 | `benchmark_annotation_system/annotated_task/benchmark/` |

### 8.3 任务 ID 格式解析

示例：`34_scatter_cm_01`

| 部分 | 含义 |
|------|------|
| `34` | 任务编号 |
| `scatter` | 图表类型（bar/scatter/line/heatmap...） |
| `cm` | 任务类别（cs/cm/vm/vs） |
| `01` | 子任务序号 |

**类别说明**：
- `cs`: Clear-Single（明确问题，单步解决）
- `cm`: Clear-Multi（明确问题，多步解决）
- `vm`: Vague-Multi（模糊问题，多步解决）
- `vs`: Vague-Single（模糊问题，单步解决）

---

## 九、最佳实践

### 9.1 测试流程建议
1. **先单模型单任务测试**：`--models qwen --task-filter 34_scatter_cm_01`
2. **确认正常后单模型多任务**：`--models qwen`
3. **最后完整 Benchmark**：`--concurrency 7`

### 9.2 API 额度管理
- 完整 238 任务 × 7 模型 ≈ 1666 次 API 调用
- 建议预算：$10-20（取决于模型选择）
- 优先测试 Qwen/Mistral（价格较低且效果较好）

### 9.3 结果版本管理
- 每次运行生成新的时间戳文件夹
- 使用 `export_results_to_csv.py --batch-dir` 导出历史版本
- 使用 Git 管理 `csv_export/` 文件夹（排除原始结果）

---

**文档版本**: v1.0  
**最后更新**: 2026-02-23  
**项目地址**: D:\Proj\AgenticVisual
