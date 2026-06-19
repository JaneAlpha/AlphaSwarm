# AlphaSwarm

[English](README.md)

一个面向量化因子发现的多智能体 LLM 系统，包含智能体编排、工具调用式因子校验、因子假设生成、表现评估与迭代排序，模拟受控研究员式分析过程。

AlphaSwarm 是一个多智能体量化因子发现系统，覆盖因子假设生成、表达式校验、因子计算、表现评估和迭代排序。

这个项目面向两类读者：

- 想了解 LLM agent 如何参与量化研究系统的开发者。
- 想观察因子发现、计算、评估、排序闭环的量化研究者。

## 项目能力

- 从 ETF 日频面板构建基础因子池。
- 使用 LLM agent 生成候选因子假设。
- 通过工具调用校验因子表达式。
- 计算因子值和因子表现指标。
- 使用 IC、ICIR、多周期表现、分层收益等指标评估因子。
- 对候选因子排序，并将评估反馈用于后续探索。
- 将运行结果落盘，支持 dashboard 查看和审计。

## 技术亮点

### 受控 Agent Loop

AlphaSwarm 使用四个 agent 组成因子研究 loop。每个 agent 负责一个清晰的研究环节，pipeline 在 agent 之间传递结构化结果。

整体循环如下：

```text
Ideator -> Calculator -> Evaluator -> Optimizer -> 下一轮 Ideator
```

这个设计对应真实研究动作：提出因子假设，确认表达式可计算，评价因子表现，再根据反馈调整下一轮探索方向。

### 智能体编排

系统将因子研究拆成四类 agent：

| Agent | 职责 |
| --- | --- |
| `FactorIdeator` | 在研究约束下生成因子假设和候选表达式。 |
| `FactorCalculator` | 校验表达式，必要时修复，计算因子值和表现指标。 |
| `FactorEvaluator` | 结合量化指标和 LLM 分析评估因子质量。 |
| `FactorOptimizer` | 阅读评估反馈，指导下一轮因子探索方向。 |

这种拆分避免把因子构想、计算、评价和优化混在一个不可解释的模型回答里。

### 工具调用式校验

因子想法在进入计算前需要经过工具校验。agent loop 会使用工具检查表达式、字段、算子和可计算性，减少无效表达式，并让 agent 输出更容易追踪。

### 评估驱动的迭代

候选因子需要经过计算和评估，指标包括 IC、ICIR、多周期表现和分层收益。评估反馈会进入后续轮次，用于持续收敛后续探索方向。

### 研究记忆边界

系统区分研究状态和运行追踪。ResearchMemory 用于保存一轮生命周期内的研究方向，并保留最小跨轮状态；`status.json` 用于定位 agent 运行进度和失败位置。研究反馈和运行调试记录不混用。

### Skill 驱动的因子构想

提示词层支持 skill 文档，用于因子假设生成和因子任务转换。skill 为 agent 提供领域研究模式，pipeline 负责执行校验、计算和评估。

## 目录结构

```text
config/                 运行配置
dashboard/              正式 dashboard 入口
data/                   本地数据工作区
prompts/skills/         agent 使用的 skill 文档
src/                    agent、pipeline、数据和 web 服务代码
run.py                  命令入口
```

以下目录属于本地运行产物，不进入仓库：

```text
checkpoints/            最新检查点文件
results/                按时间戳保存的历史运行结果
data/processed/         本地行情数据
```

## 数据要求

默认数据文件：

```text
data/processed/etf_daily_panel.parquet
```

至少需要包含：

```text
date, symbol, nav
```

如果存在 `open`、`high`、`low`、`close`、`volume`、`turnover_amount`、`return`、`flow`、`flow_ratio` 等字段，系统会在因子构建和计算中使用。

也可以通过环境变量指定数据根目录：

```text
ETF_DATA_ROOT=/path/to/data
```

## 环境变量

复制环境变量模板：

```powershell
copy .env.example .env
```

填写 DeepSeek 配置：

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`.env` 已被 Git 忽略。不要提交 API key 或本地数据。

## 安装

```powershell
pip install -e .
```

## 使用方式

构建基础因子池：

```powershell
python run.py build-base-factors
```

运行一次因子挖掘：

```powershell
python run.py run-loop-once --max-iterations 3 --factors-per-round 3
```

启动 dashboard：

```powershell
python run.py serve-dashboard --port 8021
```

浏览器打开：

```text
http://127.0.0.1:8021/dashboard/index.html
```

## 输出文件

每次完整运行会写入：

```text
results/<timestamp>/
```

典型文件包括：

```text
base_factors.json
factors_list.json
factor_cal_results.json
cal_report.md
evaluation_report.md
ranked_factors.json
base_factors.diff
status.json
cal_visualizations.json
cal_visualizations/
```

Dashboard 会读取本地运行结果，展示因子排序、计算结果、报告和因子表现可视化。

## 注意事项

- 本项目是研究原型，不构成投资建议。
- LLM 生成的因子需要独立验证后才能用于真实交易研究。
- Agent loop 依赖 LLM 调用；LLM 调用失败会中止本轮运行并记录运行错误。
- `checkpoints/`、`results/` 和本地市场数据默认不进入版本控制。

## License

MIT
