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

## 关键概念

本 README 使用以下项目术语：

| 术语 | 含义 |
| --- | --- |
| 因子假设 | 一个可检验的研究想法，用来说明某个可度量的 ETF 特征为什么可能预测未来收益。 |
| 因子表达式 | 使用可用行情字段、算子和可选基础因子引用写成的可计算公式。 |
| 基础因子池 | agent loop 开始前已经生成的一组基础因子。 |
| 研究上下文 | 某个 agent 在当前步骤可以读取的信息，包括历史因子、计算结果、评价指标和约束条件。 |
| ResearchMemory | 一份精简的研究状态，保存活跃假设、推广方向、拒绝方向、阻断因子、阻断因子家族和观察名单。 |
| 研究身份 | 附着在因子上的元信息，包括因子家族、生成策略、父因子、研究假设、预期方向和风险提示。 |
| 因子家族 | 因子的研究类别，例如动量、反转、波动风险、流动性压力、量价确认。 |
| 候选蓝图 | Optimizer 给下一轮准备的因子方案，包含策略、父因子、表达式意图、预期方向和风险提示。 |
| Optimizer 反馈钩子 | Evaluator 给 Optimizer 准备的结构化反馈，包括精炼、取反、变异、阻断候选和家族反馈。 |
| 运行状态 | 写入 `status.json` 的运行进度，用于定位当前 agent、步骤、轮次和失败位置。 |
| JSON 字段名 | `candidate_blueprints`、`refinement_blocked` 等名称是结果文件和 dashboard 使用的字段名。 |

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

## 四个 Agent 设计拆解

### `FactorIdeator`

`FactorIdeator` 负责把研究上下文转化为经过工具校验的候选因子想法。

设计目标：

- 生成可检验的因子假设。
- 使用 ResearchMemory、Optimizer 反馈、历史因子、阻断方向和基础因子池上下文。
- 为每个因子生成研究身份字段，包括 `family_tag`、`strategy`、`parent_factors`、`hypothesis`、`expected_direction` 和 `risk_note`。

操作材料：

- `checkpoints/base_factors.json` 中的基础因子池。
- 数据加载器识别出的 ETF 面板字段。
- ResearchMemory 中的活跃假设、推广方向、拒绝方向、阻断因子、阻断家族和观察名单。
- 上一轮 `FactorOptimizer` 给出的反馈。
- `prompts/skills/` 下的 skill 文档。

可用工具：

| Tool | 作用 |
| --- | --- |
| `list_skills` | 列出允许读取的研究 skill。 |
| `load_skill` | 加载 RD-Agent 派生的假设生成和因子任务转换 skill。 |
| `query_base_factor_pool` | 查询基础因子名称、公式、类别、方向和经济含义。 |
| `describe_data_fields` | 查询可用数据字段、类型和覆盖率。 |
| `analyze_factor_orthogonality` | 检查候选表达式与基础因子或参考表达式的相关性。 |
| `validate_and_compute_factor` | 校验候选因子的结构和可计算性。 |
| `evaluate_factor_quick` | 在接受候选因子前快速评估 IC 和分层收益。 |

主要产出：

- `factors_list.json`，包含表达式、家族标签、研究身份、工具校验结果和快速评估结果。

### `FactorCalculator`

`FactorCalculator` 负责在 ETF 面板上执行候选表达式，生成可用于评估的因子计算结果。

设计目标：

- 在计算前再次校验表达式。
- 在算子执行前展开基础因子引用。
- 对可安全修复的无效表达式进行修复。
- 保留原始表达式、修复表达式和计算结果之间的关系。

操作材料：

- `FactorIdeator` 生成的候选因子。
- `ETFDataLoader` 加载的 ETF 日频面板。
- 表达式解析器和算子引擎。
- 基础因子引用注册表。

可用能力：

| 能力 | 作用 |
| --- | --- |
| `validate_and_compute_factor` | 解析表达式、展开基础因子引用、计算因子值并返回覆盖率和统计信息。 |
| 表达式修复 | 在表达式校验失败时尝试受控修复。 |
| `evaluate_factor_quick` | 对成功计算的表达式生成快速表现指标。 |
| 可视化构建 | 生成多周期表现和分层收益的因子可视化数据。 |

主要产出：

- `factor_cal_results.json`
- `cal_report.md`
- `cal_visualizations.json`
- `cal_visualizations/<factor>.json`

### `FactorEvaluator`

`FactorEvaluator` 负责把计算结果转化为研究判断。

设计目标：

- 使用量化标准评价每个成功计算的因子。
- 根据 score、quality、direction、IC、ICIR、IC 胜率、多空收益和分层单调性进行排序。
- 生成可被 `FactorOptimizer` 消费的结构化反馈。
- 使用 LLM 做解释性分析，但以指标证据作为判断基础。

操作材料：

- 成功计算的因子结果。
- `FactorCalculator` 生成的快速表现指标。
- 多周期指标和分层回测结果。
- 因子家族标签和研究身份字段。

评价维度：

| 维度 | 含义 |
| --- | --- |
| `mean_ic` | 日度截面 Spearman IC 的平均值。 |
| `icir` | IC 均值相对 IC 波动的稳定性指标。 |
| `ic_win_rate` | IC 方向正确的样本占比。 |
| `long_short_return` | 最高分组和最低分组之间的收益差。 |
| `directional_monotonicity` | 分层收益是否符合预期方向的单调性。 |
| `score_breakdown` | 因子排序使用的加权证据。 |

主要产出：

- `evaluation_report.md`
- 写入 `ranked_factors.json` 的排序因子内容
- `optimizer_feedback_hook`，即给 Optimizer 使用的结构化反馈包，包含精炼、取反、变异、阻断候选和家族反馈。

### `FactorOptimizer`

`FactorOptimizer` 负责把评估证据转化为下一轮研究指令。

设计目标：

- 诊断因子有效或失败的原因。
- 判断哪些模式应该推广、拒绝、精炼、取反、变异或阻断。
- 生成下一轮候选因子蓝图。
- 在返回 loop 前用工具校验优化想法。
- 执行阻断因子精炼和阻断家族探索等硬约束。

操作材料：

- Pipeline 构建的完整研究上下文。
- 当前候选因子及其研究身份。
- 计算摘要和表达式失败信息。
- 评估摘要、排序因子和 Optimizer 反馈包。
- 基础因子池摘要。
- ResearchMemory 和因子家族集中度状态。
- 近期历史因子名称和表现。

可用工具：

| Tool | 作用 |
| --- | --- |
| `query_base_factor_pool` | 查询可支持下一轮蓝图的基础因子。 |
| `describe_data_fields` | 确认可使用的数据输入。 |
| `validate_and_compute_factor` | 检查候选蓝图是否能转化为可计算表达式。 |
| `evaluate_factor_quick` | 快速测试候选表达式的初步表现。 |
| `analyze_factor_orthogonality` | 检查候选蓝图与基础因子或父因子的重合程度。 |

主要产出：

- `diagnostics`
- `next_round_plan`
- `candidate_blueprints`：Optimizer 提出的下一轮候选因子蓝图。
- `guardrails`：下一轮构想需要遵守的约束。
- `refinement_blocked`：后续禁止继续精炼的因子。
- `family_exploration_blocked`：后续禁止继续探索的因子家族。
- `reject_patterns`：证据较弱的研究方向。
- `promote_patterns`：证据较强的研究方向。
- 追加到 `evaluation_report.md` 的最终研究建议

### Pipeline 层编排

Pipeline 负责协调四个 agent，并保持职责边界。

每轮执行顺序：

1. `FactorIdeator` 发现经过工具校验的候选因子。
2. `FactorCalculator` 计算表达式并记录修复情况。
3. `FactorEvaluator` 排序因子表现并生成 Optimizer 反馈包。
4. `FactorOptimizer` 在非最终轮生成下一轮反馈，在最终轮写入最终研究建议。

Pipeline 管理的衔接动作：

- 将计算修复结果回填到因子记录。
- 将研究来源写入排序因子。
- 在轮次之间维护 ResearchMemory。
- 追踪因子家族集中度并阻断过度集中的家族。
- 写入 `status.json`，用于运行追踪和失败定位。
- 将输出同时写入最新检查点和按时间戳保存的结果目录。

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

## 项目局限

本项目是研究原型，存在以下重要局限：

- **LLM 因子挖掘的数学基础仍需验证。** 因子表达式发现本质上包含在已知数学表达式空间中的搜索问题。传统量化研究常用遗传规划、稀疏模型或其他具有明确数学性质的方法。LLM 生成因子表达式的实际优势仍需要实证验证。LLM 更自然的能力是从研报、论文和领域文本中提取研究思路，单独承担高效数学搜索仍需要谨慎评估。
- **LLM 数学推理可靠性有限。** 因子表达式需要精确的数学运算和逻辑验证。工具调用可以捕捉语法错误、字段缺失、算子不支持和明显不可计算的表达式。更细微的问题仍需要额外控制和人工审查，例如未来函数、过拟合、逻辑等价但数值不稳定的表达式，以及经济逻辑薄弱的表达式。
- **过拟合控制尚不完整。** 因子挖掘容易受到多重比较影响并产生假阳性。在较大的因子空间中反复搜索，可能偶然得到样本内 IC 较高的因子。当前 README 尚未给出完整的交叉验证、样本外测试、滚动窗口评估或多重假设检验校正方案。
- **ETF 日频数据存在天然限制。** ETF 日频面板的因子信噪比可能较低。分散化效应会削弱一些在个股或更高频数据中更明显的因子家族。基于 ETF 日频数据挖掘出的因子，其实际预测能力可能有限。
- **缺少实盘或样本外案例。** 因子挖掘系统最终需要通过样本外表现和回测证据评价。当前仓库尚未提供一组公开整理的案例，用来展示已挖掘因子在样本外具有稳健表现。

## 注意事项

- 本项目是研究原型，不构成投资建议。
- LLM 生成的因子需要独立验证后才能用于真实交易研究。
- Agent loop 依赖 LLM 调用；LLM 调用失败会中止本轮运行并记录运行错误。
- `checkpoints/`、`results/` 和本地市场数据默认不进入版本控制。

## License

MIT
