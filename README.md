# AlphaSwarm

[中文版](README.zh-CN.md)

Multi-agent LLM system for quantitative factor discovery, featuring agent orchestration, tool-calling factor validation, hypothesis generation, performance evaluation, and iterative ranking under a controlled analyst-style research process.

AlphaSwarm is a quantitative factor discovery system built with multiple LLM agents. It covers factor hypothesis generation, expression validation, factor calculation, performance evaluation, and iterative ranking.

The project is aimed at developers and researchers who want to study agent-based quantitative research systems with explicit tool use and measurable factor evaluation.

## What It Does

- Builds a base factor pool from an ETF daily panel.
- Uses LLM agents to generate candidate factor hypotheses.
- Validates factor expressions through tool calls before calculation.
- Computes factor values and performance metrics.
- Evaluates factors with IC, ICIR, multi-period behavior, and stratified returns.
- Ranks candidate factors and feeds evaluation feedback into later exploration.
- Records local run outputs for audit and dashboard review.

## Technical Highlights

### Controlled Agent Loop

AlphaSwarm uses a four-agent loop rather than a single prompt that asks the model to list factors. Each agent has a separate research responsibility, and the pipeline passes structured results between them.

The loop is designed around a simple research cycle:

```text
Ideator -> Calculator -> Evaluator -> Optimizer -> next round Ideator
```

This makes factor discovery closer to an analyst process: propose a hypothesis, test whether it can be computed, evaluate the result, then use feedback to guide the next attempt.

### Agent Orchestration

The system separates agent responsibilities:

| Agent | Responsibility |
| --- | --- |
| `FactorIdeator` | Generates factor hypotheses and candidate expressions under research constraints. |
| `FactorCalculator` | Validates expressions, repairs when possible, computes factor values, and calculates metrics. |
| `FactorEvaluator` | Reviews calculated factors using quantitative metrics and LLM analysis. |
| `FactorOptimizer` | Interprets evaluation feedback and guides the next round of exploration. |

This separation was chosen to keep ideation, calculation, evaluation, and optimization from collapsing into one opaque model response.

### Tool-Calling Validation

Factor ideas are expected to pass through tools before becoming calculation targets. The agent loop uses tool calls to check factor expressions, available fields, supported operators, and computability. This reduces invalid expressions and makes the agent output easier to inspect.

### Evaluation-Driven Iteration

The system does not treat all generated factors equally. Candidate factors are calculated and evaluated with quantitative metrics such as IC, ICIR, multi-period performance, and stratified returns. Evaluation feedback is passed into later rounds so that the loop can refine exploration direction instead of repeatedly generating unrelated ideas.

### Research Memory Boundary

The project distinguishes research state from runtime tracing. Research memory is used to guide one loop lifecycle and retain minimal cross-round direction, while runtime status is used to locate agent progress and failures. This keeps research feedback separate from operational debugging records.

### Skill-Guided Ideation

The prompt layer supports skill documents for factor hypothesis generation and factor-task conversion. These skills give the agents domain-specific research patterns without hard-coding every candidate factor into the pipeline.

## Repository Layout

```text
config/                 Runtime configuration
dashboard/              Project dashboard entry
data/                   Local data workspace
prompts/skills/         Skill documents used by the agents
src/                    Agent, pipeline, data, and web-server code
run.py                  Command entry point
```

Local runtime directories are intentionally not part of the repository:

```text
checkpoints/            Latest local checkpoint files
results/                Timestamped run outputs
data/processed/         Local market data
```

## Data

The default data file is:

```text
data/processed/etf_daily_panel.parquet
```

The expected panel contains at least:

```text
date, symbol, nav
```

Optional fields such as `open`, `high`, `low`, `close`, `volume`, `turnover_amount`, `return`, `flow`, and `flow_ratio` are used when available.

You can also set a custom data root:

```text
ETF_DATA_ROOT=/path/to/data
```

## Environment

Copy the example environment file:

```powershell
copy .env.example .env
```

Fill in your own DeepSeek credentials:

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`.env` is ignored by Git. Do not commit API keys or local data files.

## Installation

```powershell
pip install -e .
```

## Usage

Build the base factor pool:

```powershell
python run.py build-base-factors
```

Run one factor-mining loop:

```powershell
python run.py run-loop-once --max-iterations 3 --factors-per-round 3
```

Start the dashboard:

```powershell
python run.py serve-dashboard --port 8021
```

Then open:

```text
http://127.0.0.1:8021/dashboard/index.html
```

## Outputs

Each completed run is written to:

```text
results/<timestamp>/
```

Typical outputs include:

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

The dashboard reads local run outputs and provides a project-level view of factor rankings, calculation results, reports, and visualized performance.

## Notes

- This is a research prototype, not an investment advisory system.
- Generated factors require independent validation before any real trading use.
- LLM availability is required for the agent loop; the system does not silently replace failed LLM calls with rule-based fallback generation.
- Runtime checkpoints, historical results, and local market data are excluded from version control by default.

## License

MIT
