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

## Key Concepts

This README uses a small set of project terms:

| Term | Meaning |
| --- | --- |
| Factor hypothesis | A testable research idea about why a measurable ETF feature may predict future returns. |
| Factor expression | A computable formula built from available market fields, operators, and optional base-factor references. |
| Base factor pool | The initial set of known factors generated before the agent loop starts. |
| Research context | The information given to an agent at a specific step, including prior factors, calculation results, evaluation metrics, and active constraints. |
| ResearchMemory | A compact loop-level state that keeps active hypotheses, promoted directions, rejected directions, blocked factors, blocked factor families, and watchlist factors. |
| Research identity | Metadata attached to each factor, including family, strategy, parent factors, hypothesis, expected direction, and risk note. |
| Factor family | A research category such as momentum, reversal, volatility risk, liquidity pressure, or volume-price confirmation. |
| Candidate blueprint | An optimizer proposal for a next-round factor, including its research strategy, parent factors, expression intent, expected direction, and risk note. |
| Optimizer feedback hook | Structured evaluation feedback prepared for the optimizer, including candidates for refinement, inversion, mutation, blocking, and family-level feedback. |
| Runtime status | Operational progress written to `status.json`, used to locate the active agent, step, iteration, and failure point. |
| JSON field names | Names such as `candidate_blueprints` and `refinement_blocked` are output fields used by result files and downstream dashboard views. |

## Technical Highlights

### Controlled Agent Loop

AlphaSwarm uses a four-agent loop for factor research. Each agent has a separate research responsibility, and the pipeline passes structured results between them.

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

Candidate factors are calculated and evaluated with quantitative metrics such as IC, ICIR, multi-period performance, and stratified returns. Evaluation feedback is passed into later rounds so that the loop can refine exploration direction across iterations.

### Research Memory Boundary

The project distinguishes research state from runtime tracing. Research memory is used to guide one loop lifecycle and retain minimal cross-round direction, while runtime status is used to locate agent progress and failures. This keeps research feedback separate from operational debugging records.

### Skill-Guided Ideation

The prompt layer supports skill documents for factor hypothesis generation and factor-task conversion. These skills give the agents domain-specific research patterns without hard-coding every candidate factor into the pipeline.

## Agent Design Breakdown

### `FactorIdeator`

`FactorIdeator` is responsible for turning research context into tool-validated candidate factor ideas.

Design goal:

- Generate testable factor hypotheses.
- Use ResearchMemory, optimizer feedback, prior factors, blocked directions, and base factor context.
- Assign each factor a research identity, including `family_tag`, `strategy`, `parent_factors`, `hypothesis`, `expected_direction`, and `risk_note`.

Operating materials:

- Base factor pool from `checkpoints/base_factors.json`.
- Available ETF panel fields from the data loader.
- ResearchMemory, including active hypotheses, promoted directions, rejected directions, blocked factors, blocked families, and watchlist factors.
- Previous round feedback from `FactorOptimizer`.
- Skill documents under `prompts/skills/`.

Available tools:

| Tool | Purpose |
| --- | --- |
| `list_skills` | Lists allowed research skills. |
| `load_skill` | Loads the RD-Agent-derived hypothesis and factor-task skills. |
| `query_base_factor_pool` | Reads base factor definitions, formulas, categories, directions, and economic meanings. |
| `describe_data_fields` | Describes available data fields, types, and coverage. |
| `analyze_factor_orthogonality` | Checks correlation against base factors or reference expressions. |
| `validate_and_compute_factor` | Validates candidate factor structure and computability. |
| `evaluate_factor_quick` | Runs a quick IC and stratified-return evaluation before accepting a candidate. |

Main output:

- `factors_list.json`, including validated factor expressions, family tags, research identity, tool validation, and quick evaluation.

### `FactorCalculator`

`FactorCalculator` is responsible for executing candidate expressions on the ETF panel and producing factor results for evaluation.

Design goal:

- Re-validate expressions before calculation.
- Expand base-factor references before operator evaluation.
- Repair invalid expressions when a safe repair is possible.
- Preserve the link between original expression, repaired expression, and calculation result.

Operating materials:

- Candidate factors from `FactorIdeator`.
- ETF daily panel loaded by `ETFDataLoader`.
- Operator engine and expression parser.
- Base factor reference registry.

Available tools and internal capabilities:

| Capability | Purpose |
| --- | --- |
| `validate_and_compute_factor` | Parses expressions, expands base-factor references, computes factor values, and returns coverage/statistics. |
| Expression repair | Attempts controlled repair when an expression fails validation. |
| `evaluate_factor_quick` | Computes quick performance for successful factor expressions. |
| Visualization builder | Creates per-factor visualization payloads for multi-period metrics and stratified returns. |

Main outputs:

- `factor_cal_results.json`
- `cal_report.md`
- `cal_visualizations.json`
- `cal_visualizations/<factor>.json`

### `FactorEvaluator`

`FactorEvaluator` is responsible for converting calculation results into research judgments.

Design goal:

- Evaluate each successful factor with quantitative standards.
- Rank factors using score, quality, direction, IC, ICIR, IC win rate, long-short return, and stratified monotonicity.
- Produce structured feedback that can be consumed by `FactorOptimizer`.
- Require LLM analysis for interpretation while keeping metric evidence as the basis.

Operating materials:

- Successful factor calculation results.
- Quick performance output from `FactorCalculator`.
- Multi-period metrics and stratified backtest results.
- Factor family tags and research identity.

Evaluation dimensions:

| Dimension | Meaning |
| --- | --- |
| `mean_ic` | Average cross-sectional Spearman IC. |
| `icir` | IC stability measured by mean IC divided by IC volatility. |
| `ic_win_rate` | Share of periods where IC direction is favorable. |
| `long_short_return` | Return spread between top and bottom quantile groups. |
| `directional_monotonicity` | Whether stratified returns are monotonic in the expected direction. |
| `score_breakdown` | Weighted evidence used for factor ranking. |

Main outputs:

- `evaluation_report.md`
- Ranked factor payload used later in `ranked_factors.json`
- `optimizer_feedback_hook`, including refinement, inversion, mutation, block candidates, and family feedback.

### `FactorOptimizer`

`FactorOptimizer` is responsible for turning evaluation evidence into the next research instruction set.

Design goal:

- Diagnose why factors worked or failed.
- Decide which patterns should be promoted, rejected, refined, inverted, or blocked.
- Generate candidate blueprints for the next round.
- Use tools to validate optimization ideas before returning them to the loop.
- Enforce hard constraints such as blocked factor refinement and blocked family exploration.

Operating materials:

- Full research context built by the pipeline.
- Current candidate factors and their research identities.
- Calculation summary and failed-expression information.
- Evaluation summary, ranked factors, and the optimizer feedback hook.
- Base factor pool summary.
- ResearchMemory and factor-family concentration state.
- Recent historical factor names and performance.

Available tools:

| Tool | Purpose |
| --- | --- |
| `query_base_factor_pool` | Checks base factors that may support next-round blueprints. |
| `describe_data_fields` | Confirms available data inputs. |
| `validate_and_compute_factor` | Tests whether a proposed blueprint can become a computable expression. |
| `evaluate_factor_quick` | Tests early performance evidence for a proposed expression. |
| `analyze_factor_orthogonality` | Checks whether a blueprint overlaps too much with base or parent factors. |

Main outputs:

- `diagnostics`
- `next_round_plan`
- `candidate_blueprints`: next-round factor blueprints proposed by the optimizer.
- `guardrails`: constraints that the next ideation round should follow.
- `refinement_blocked`: factors excluded from later refinement.
- `family_exploration_blocked`: factor families excluded from later exploration.
- `reject_patterns`: weak research directions.
- `promote_patterns`: research directions with stronger evidence.
- Final research advice appended to `evaluation_report.md`

### Pipeline-Level Coordination

The pipeline coordinates the four agents and keeps their responsibilities separated.

Per iteration:

1. `FactorIdeator` discovers validated candidate factors.
2. `FactorCalculator` computes expressions and records repairs.
3. `FactorEvaluator` ranks factor performance and builds the optimizer feedback package.
4. `FactorOptimizer` produces feedback for the next round, except on the final loop where it writes final research advice.

Pipeline-managed coordination:

- Attaches calculation repairs back to factor records.
- Attaches research sources to ranked factors.
- Maintains ResearchMemory across loop iterations.
- Tracks factor family concentration and blocks over-concentrated families.
- Writes `status.json` for runtime tracing and failure localization.
- Writes mirrored outputs to latest checkpoints and timestamped result directories.

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
- LLM availability is required for the agent loop; failed LLM calls stop the run and are reported as runtime errors.
- Runtime checkpoints, historical results, and local market data are excluded from version control by default.

## License

MIT
