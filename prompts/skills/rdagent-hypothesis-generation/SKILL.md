---
name: rdagent-hypothesis-generation
description: RD-Agent-derived hypothesis generation workflow for FactorIdeator. Use when generating ETF factor research hypotheses from prior trials, feedback, ResearchMemory, base factor context, and optimizer advice before proposing concrete factor expressions.
---

# RD-Agent Hypothesis Generation

Use this skill before proposing candidate factors. The output of this skill is a research hypothesis, not an accepted factor.

## Source

- Source project: `microsoft/RD-Agent`
- Source files:
  - `refference/RD-Agent/rdagent/components/proposal/prompts.yaml`
  - `refference/RD-Agent/rdagent/scenarios/qlib/prompts.yaml`
- Source prompt keys:
  - `hypothesis_gen`
  - `factor_hypothesis_specification`
  - `hypothesis_output_format`
  - `factor_hypothesis_output_format`
- License: RD-Agent is distributed under the MIT License. See `refference/RD-Agent/LICENSE`.

## Procedure

1. Read the current loop context:
   - iteration number
   - ResearchMemory
   - prior factor names and performance
   - optimizer feedback
   - rejected or blocked factors
   - blocked factor families
   - base factor pool summary

2. Generate one testable hypothesis before factor design.

3. Ground the hypothesis in observed evidence:
   - prior successful factors
   - prior weak factors
   - optimizer diagnostics
   - available ETF daily data fields
   - base factor families

4. Choose the research action:
   - `new_hypothesis` when no reliable prior direction exists
   - `refinement` when a prior direction was useful but needs a controlled improvement
   - `mutation` when modifying one prior factor mechanism
   - `crossover` when combining two compatible mechanisms
   - `inversion` when testing the reverse side of a failed but economically meaningful mechanism

5. Reject directions that violate ResearchMemory:
   - blocked factors cannot be parents
   - blocked families cannot be explored
   - rejected patterns cannot be repeated without a concrete mechanism change

## Output Contract

The hypothesis must be concise and usable by the next skill.

```json
{
  "hypothesis": "A specific, testable statement that explains why the next factor set may improve predictive power.",
  "reason": "Evidence-based reason for this hypothesis.",
  "action": "new_hypothesis | refinement | mutation | crossover | inversion",
  "expected_direction": "positive | negative | neutral",
  "risk_note": "Main risk or failure mode."
}
```

## Constraints

- Do not produce final factors directly.
- Do not bypass factor tools.
- Do not ignore optimizer hard blocks.
- Do not use subjective text analysis as a factor input.
- Prefer simple and effective mechanisms early.
- Increase complexity only when prior results support it.
