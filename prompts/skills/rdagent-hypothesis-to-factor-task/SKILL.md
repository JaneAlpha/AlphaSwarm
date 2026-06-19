---
name: rdagent-hypothesis-to-factor-task
description: RD-Agent-derived workflow that converts a hypothesis into structured ETF factor tasks and screens candidates for viability, relevance, duplication, and implementability before tool validation.
---

# RD-Agent Hypothesis To Factor Task

Use this skill after loading `rdagent-hypothesis-generation` and before calling factor validation tools.

## Source

- Source project: `microsoft/RD-Agent`
- Source files:
  - `refference/RD-Agent/rdagent/components/proposal/prompts.yaml`
  - `refference/RD-Agent/rdagent/scenarios/qlib/prompts.yaml`
  - `refference/RD-Agent/rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml`
  - `refference/RD-Agent/rdagent/components/coder/factor_coder/prompts.yaml`
- Source prompt keys:
  - `hypothesis2experiment`
  - `factor_experiment_output_format`
  - `factor_viability_system`
  - `factor_relevance_system`
  - `factor_duplicate_system`
  - `select_implementable_factor_system`
- License: RD-Agent is distributed under the MIT License. See `refference/RD-Agent/LICENSE`.

## Factor Task Schema

Convert each accepted hypothesis into one or more structured factor tasks.

```json
{
  "name": "factor_name",
  "description": "[Factor Type] concise factor description",
  "formulation": "mathematical formulation or expression intent",
  "variables": {
    "variable_or_function": "meaning"
  },
  "category": "momentum | reversal | volatility | volume_price | trend | liquidity",
  "family_tag": "price_momentum | price_reversal | volatility_risk | volume_price_confirmation | liquidity_pressure | trend_strength | flow_attention | cross_family_composite | other",
  "family_rationale": "research-mechanism reason for the family label",
  "strategy": "new_hypothesis | refinement | mutation | crossover | inversion",
  "parent_factors": ["parent_factor_name"],
  "hypothesis": "source hypothesis",
  "expected_direction": "positive | negative | neutral",
  "risk_note": "main risk",
  "optimizer_instruction": "optimizer guidance used, if any"
}
```

## Screening Rules

Apply these checks before calling `validate_and_compute_factor`.

### Viability

Keep only factors that can be calculated:

- at daily frequency
- for each ETF symbol
- from available source fields
- using the allowed expression operators

Reject factors that require unavailable fundamentals, analyst expectations, news, natural language, minute data, or subjective judgment.

### Relevance

Keep only real quantitative investment factors:

- mathematical manipulation only
- no subjective classification
- no future leakage
- no target return directly embedded in the factor

### Duplicate Check

Reject candidates that are equivalent to:

- another candidate in this round
- a recent historical candidate
- a base factor with only superficial renaming
- a blocked factor under a minor expression rewrite

### Implementability

Prefer candidates with:

- simple expressions
- stable numerical operations
- available fields
- no fragile division by near-zero denominators unless protected by `add(abs(x), 1e-6)` style guards
- clear expected direction

## Required Tool Sequence

For each final candidate:

1. Call `analyze_factor_orthogonality`.
2. Call `validate_and_compute_factor`.
3. If valid, call `evaluate_factor_quick`.

Only factors passing tool validation can enter the discovered factor list.

## Constraints

- This skill does not calculate factors by itself.
- This skill does not rank factors.
- This skill does not update ResearchMemory.
- This skill does not override Optimizer hard constraints.
- This skill only shapes candidate research tasks before existing tools validate them.
