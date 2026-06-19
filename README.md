# MAS-FactorMiner

MAS-FactorMiner is an ETF factor-mining prototype built around a multi-agent research loop.

The system starts from a local ETF daily panel, builds a base factor pool, asks LLM agents to propose candidate factors through tool calls, calculates candidate expressions, evaluates factor performance, and records historical runs under `results/`.

## Core Workflow

1. `FactorIdeator` proposes candidate factor expressions with tool validation.
2. `FactorCalculator` validates expressions, computes factor values, repairs invalid expressions when possible, and calculates performance metrics.
3. `FactorEvaluator` ranks factors with IC, ICIR, multi-period performance, stratified returns, and LLM analysis.
4. `FactorOptimizer` analyzes evaluation feedback and controls the next round of factor exploration.

## Data

The default data location is:

```text
data/processed/etf_daily_panel.parquet
```

You can also provide:

```text
data/stock_data.db
```

or set:

```text
ETF_DATA_ROOT=/path/to/data
```

The expected daily panel contains at least:

```text
date, symbol, nav
```

Optional fields such as `open`, `high`, `low`, `close`, `volume`, `turnover_amount`, `return`, `flow`, and `flow_ratio` are used when available.

## Environment

Copy the example file and fill in your own key:

```powershell
copy .env.example .env
```

Required variables:

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Install

```powershell
pip install -e .
```

## Build Base Factors

```powershell
python run.py build-base-factors
```

This creates `checkpoints/base_factors.json`. The `checkpoints/` directory is a local runtime workspace and is not intended to be committed.

## Run One Mining Pipeline

```powershell
python run.py run-loop-once --max-iterations 3 --factors-per-round 3
```

Outputs are written to:

```text
results/<timestamp>/
```

The pipeline also mirrors the latest runtime checkpoint files under `checkpoints/`.

## Dashboard

Start the formal dashboard server:

```powershell
python run.py serve-dashboard --port 8021
```

Open:

```text
http://127.0.0.1:8021/dashboard/index.html
```

The dashboard reads historical results from `results/` and can start a new pipeline run from the web UI.

## Runtime Outputs

Each completed result directory contains:

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

## Repository Hygiene

Do not commit:

```text
.env
checkpoints/
results/
data/
__pycache__/
refference/
```

Use `results/` for local experiments and keep only curated examples if needed.

## License

MIT
