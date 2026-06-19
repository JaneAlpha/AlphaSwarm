from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_calculator import FactorCalculatorAgent
from .agent_evaluator import FactorEvaluatorAgent
from .agent_ideator import FactorIdeatorAgent
from .agent_optimizer import FactorOptimizerAgent
from .agent_tools import FactorToolbox


class SingleRunFactorMiningPipeline:
    FAMILY_CONCENTRATION_THRESHOLD = 0.35
    FAMILY_CONCENTRATION_MIN_FACTORS = 3
    FAMILY_CONCENTRATION_MIN_FAMILY_COUNT = 2
    CHECKPOINT_TEXT_FILES = {
        "cal_report.md",
        "evaluation_report.md",
        "base_factors.diff",
    }
    CHECKPOINT_JSON_FILES = {
        "factors_list.json",
        "factor_cal_results.json",
        "ranked_factors.json",
        "status.json",
    }

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        max_iterations: int = 3,
        factors_per_round: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_iterations = max_iterations
        self.factors_per_round = factors_per_round
        self.toolbox = FactorToolbox()
        self.ideator = FactorIdeatorAgent(self.toolbox)
        self.calculator = FactorCalculatorAgent(self.toolbox)
        self.evaluator = FactorEvaluatorAgent()
        self.optimizer = FactorOptimizerAgent(self.toolbox)
        self.result_dir: Path | None = None

    def run(self) -> dict[str, Any]:
        started = datetime.now()
        started_at = started.isoformat(timespec="seconds")
        run_id = started.strftime("%Y%m%d_%H%M%S")
        run_id, self.result_dir = self._prepare_result_dir(run_id)
        iterations: list[dict[str, Any]] = []
        all_agent_calls: list[dict[str, Any]] = []
        all_ranked: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        history_performance: list[dict[str, Any]] = []
        feedback: dict[str, Any] | None = None
        optimizer_feedback: dict[str, Any] | None = None
        research_memory = self._init_research_memory()
        factors: list[dict[str, Any]] = []
        calculation: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        final_optimizer_result: dict[str, Any] | None = None
        final_research_context: dict[str, Any] | None = None
        status = self._init_status(started_at, run_id, self.result_dir, all_agent_calls, iterations)

        try:
            self._checkpoint_status(
                status,
                current_stage="running",
                current_iteration=0,
                current_agent=None,
                last_event="pipeline_started",
            )

            for iteration in range(1, self.max_iterations + 1):
                round_memory = self._init_round_memory(iteration)
                optimizer_call: dict[str, Any] | None = None
                self._agent_started(status, iteration, self.ideator.name, "discover")
                factors, ideator_calls = self.ideator.discover(
                    iteration=iteration,
                    factors_per_round=self.factors_per_round,
                    history=history,
                    feedback=feedback,
                    optimizer_feedback=optimizer_feedback,
                    research_memory=research_memory,
                )
                all_agent_calls.extend(ideator_calls)
                self._agent_completed(
                    status,
                    iteration,
                    self.ideator.name,
                    "discover",
                    {"factors_count": len(factors), "tool_calls": len(ideator_calls)},
                )
                factors = self._attach_research_sources(factors, iteration, optimizer_feedback)
                self._hook_ideation(round_memory, factors)

                self._agent_started(status, iteration, self.calculator.name, "calculate")
                calculation, calculator_calls = self.calculator.calculate(factors)
                all_agent_calls.extend(calculator_calls)
                self._agent_completed(
                    status,
                    iteration,
                    self.calculator.name,
                    "calculate",
                    {
                        "success_count": calculation["success_count"],
                        "failure_count": calculation["failure_count"],
                        "tool_calls": len(calculator_calls),
                    },
                )
                factors = self._attach_calculation_repairs(factors, calculation)
                self._hook_calculation(round_memory, calculation)

                self._agent_started(status, iteration, self.evaluator.name, "evaluate")
                evaluation, evaluator_calls = self.evaluator.evaluate(calculation)
                all_agent_calls.extend(evaluator_calls)
                self._agent_completed(
                    status,
                    iteration,
                    self.evaluator.name,
                    "evaluate",
                    {
                        "evaluated_factors": evaluation["factor_count"],
                        "effective_factors": evaluation["effective_count"],
                        "excellent_factors": evaluation["excellent_count"],
                        "tool_calls": len(evaluator_calls),
                    },
                )
                evaluation = self._attach_sources_to_evaluation(evaluation, factors, calculation)
                self._hook_evaluation(round_memory, evaluation)
                research_context = self._build_research_context(
                    iteration=iteration,
                    factors=factors,
                    calculation=calculation,
                    evaluation=evaluation,
                    history=history,
                    history_performance=history_performance,
                    research_memory=research_memory,
                )
                is_final_iteration = iteration == self.max_iterations
                if is_final_iteration:
                    final_research_context = research_context
                else:
                    self._agent_started(status, iteration, self.optimizer.name, "optimize")
                    optimizer_result = self.optimizer.optimize_from_context(
                        research_context=research_context,
                        target_count=self.factors_per_round,
                    )
                    all_agent_calls.extend(self._optimizer_tool_calls(optimizer_result, iteration, "optimize"))
                    optimizer_feedback = optimizer_result["optimization"]
                    self._hook_optimizer(round_memory, optimizer_feedback)
                    optimizer_feedback["refinement_blocked"] = (
                        round_memory.get("optimizer_advice") or {}
                    ).get("refinement_blocked") or []
                    optimizer_feedback["family_exploration_blocked"] = (
                        round_memory.get("optimizer_advice") or {}
                    ).get("family_exploration_blocked") or []
                    memory_delta = self._build_memory_delta(round_memory)
                    round_memory["memory_delta"] = memory_delta
                    self._apply_memory_delta(research_memory, memory_delta)
                    optimizer_call = {
                        "time": optimizer_result["created_at"],
                        "agent": self.optimizer.name,
                        "called": "FactorOptimizer.optimize_from_context",
                        "input": {
                            "iteration": iteration,
                            "target_count": self.factors_per_round,
                            "final_research_advice": False,
                            "candidate_factors": len(factors),
                            "calculation_success": calculation["success_count"],
                            "calculation_failure": calculation["failure_count"],
                            "evaluated_factors": evaluation["factor_count"],
                        },
                        "result": {
                            **self._summarize_optimizer_feedback(optimizer_feedback),
                            "tool_trace_count": len(optimizer_result.get("tool_trace") or []),
                        },
                    }
                    all_agent_calls.append(optimizer_call)
                    self._agent_completed(
                        status,
                        iteration,
                        self.optimizer.name,
                        "optimize",
                        self._summarize_optimizer_feedback(optimizer_feedback),
                    )
                history.extend(factors)
                history_performance.extend(
                    {**item, "iteration": iteration}
                    for item in evaluation["ranked_factors"]
                )
                feedback = evaluation
                all_ranked.extend(
                    {**item, "iteration": iteration}
                    for item in evaluation["ranked_factors"]
                )
                iterations.append(
                    {
                        "iteration": iteration,
                        "factors_count": len(factors),
                        "calculation_success": calculation["success_count"],
                        "calculation_failure": calculation["failure_count"],
                        "evaluated_factors": evaluation["factor_count"],
                        "effective_factors": evaluation["effective_count"],
                        "excellent_factors": evaluation["excellent_count"],
                        "top_factor": self._without_research_source(evaluation["ranked_factors"][0]) if evaluation["ranked_factors"] else None,
                    }
                )
                self._checkpoint_status(
                    status,
                    current_stage="running",
                    current_iteration=iteration,
                    current_agent=None,
                    last_event="iteration_completed",
                )
        except Exception as exc:
            self._agent_failed(status, exc)
            raise

        global_ranked = sorted(all_ranked, key=lambda x: x.get("score") or 0, reverse=True)
        ranked_payload = {
            **evaluation,
            "rank_scope": "all_iterations",
            "max_iterations": self.max_iterations,
            "factors_per_round": self.factors_per_round,
            "factor_count": len(global_ranked),
            "effective_count": sum(1 for item in global_ranked if item.get("is_effective")),
            "excellent_count": sum(1 for item in global_ranked if item.get("is_excellent")),
            "ranked_factors": global_ranked,
        }

        if final_research_context is not None:
            final_research_context = {
                **final_research_context,
                "global_evaluation_summary": {
                    "factor_count": len(global_ranked),
                    "effective_count": ranked_payload["effective_count"],
                    "excellent_count": ranked_payload["excellent_count"],
                    "ranked_factors": self._compact_ranked(global_ranked),
                },
            }
            try:
                self._agent_started(status, self.max_iterations, self.optimizer.name, "final_research_advice")
                final_optimizer_result = self.optimizer.optimize_from_context(
                    research_context=final_research_context,
                    target_count=self.factors_per_round,
                )
                all_agent_calls.extend(self._optimizer_tool_calls(final_optimizer_result, self.max_iterations, "final_research_advice"))
                all_agent_calls.append(
                    {
                        "time": final_optimizer_result["created_at"],
                        "agent": self.optimizer.name,
                        "called": "FactorOptimizer.final_research_advice",
                        "input": {
                            "iteration": self.max_iterations,
                            "target_count": self.factors_per_round,
                            "final_research_advice": True,
                            "rank_scope": "all_iterations",
                            "evaluated_factors": ranked_payload["factor_count"],
                            "effective_factors": ranked_payload["effective_count"],
                            "excellent_factors": ranked_payload["excellent_count"],
                        },
                        "result": {
                            **self._summarize_optimizer_feedback(final_optimizer_result["optimization"]),
                            "tool_trace_count": len(final_optimizer_result.get("tool_trace") or []),
                        },
                    }
                )
                self._agent_completed(
                    status,
                    self.max_iterations,
                    self.optimizer.name,
                    "final_research_advice",
                    self._summarize_optimizer_feedback(final_optimizer_result["optimization"]),
                )
            except Exception as exc:
                self._agent_failed(status, exc)
                raise

        factors_payload = {
            "schema_version": "1.0",
            "agent": self.ideator.name,
            "generation_mode": "tool_call_only",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "iteration": self.max_iterations,
            "factor_count": len(factors),
            "factors": [self._without_research_source(item) for item in factors],
        }

        status.update(
            {
                "summary": {
                    "discovered_factors": len(factors),
                    "calculation_success": calculation["success_count"],
                    "calculation_failure": calculation["failure_count"],
                    "evaluated_factors": evaluation["factor_count"],
                    "effective_factors": evaluation["effective_count"],
                    "excellent_factors": evaluation["excellent_count"],
                    "total_discovered_factors": len(history),
                    "total_ranked_factors": len(global_ranked),
                },
            }
        )
        self._checkpoint_status(
            status,
            current_stage="writing_outputs",
            current_iteration=self.max_iterations,
            current_agent=None,
            current_step="checkpoint_outputs",
            last_event="checkpoint_outputs_started",
        )
        try:
            self._write_json("factors_list.json", factors_payload)
            self._write_json("factor_cal_results.json", calculation)
            self._write_cal_visualizations(self.calculator.build_visualizations(calculation))
            self._write_text("cal_report.md", self.calculator.build_report(calculation))
            evaluation_report = self.evaluator.build_report(ranked_payload)
            if final_optimizer_result:
                evaluation_report += self.optimizer.build_final_report_section(final_optimizer_result)
            self._write_text("evaluation_report.md", evaluation_report)
            self._write_json("ranked_factors.json", ranked_payload)
            self._write_text("base_factors.diff", self._build_base_factor_diff(ranked_payload))
        except Exception as exc:
            self._checkpoint_output_failed(status, exc)
            raise
        self._checkpoint_status(
            status,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            current_stage="completed",
            current_agent=None,
            current_step=None,
            last_event="pipeline_completed",
        )
        return status

    def _init_status(
        self,
        started_at: str,
        run_id: str,
        result_dir: Path,
        all_agent_calls: list[dict[str, Any]],
        iterations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result_prefix = self._to_posix(result_dir)
        return {
            "schema_version": "1.0",
            "pipeline": "max_3_round_tool_call_factor_mining",
            "run_id": run_id,
            "result_dir": result_prefix,
            "started_at": started_at,
            "finished_at": None,
            "loop_enabled": True,
            "max_iterations": self.max_iterations,
            "factors_per_round": self.factors_per_round,
            "current_stage": "created",
            "current_iteration": 0,
            "current_agent": None,
            "current_step": None,
            "last_event": "status_created",
            "last_success_agent": None,
            "failure": None,
            "agents": [
                {"name": self.ideator.name, "role": "discover factors with tool validation"},
                {"name": self.calculator.name, "role": "calculate discovered factor expressions"},
                {"name": self.evaluator.name, "role": "evaluate factors using IC/ICIR, stratified backtest, monotonicity, and LLM analysis"},
                {"name": self.optimizer.name, "role": "diagnose evaluation results, control non-final ideation rounds, and write final research advice"},
            ],
            "agent_run_trace": [],
            "agent_calls": all_agent_calls,
            "iterations": iterations,
            "outputs": {
                "factors_list": "checkpoints/factors_list.json",
                "factor_cal_results": "checkpoints/factor_cal_results.json",
                "cal_report": "checkpoints/cal_report.md",
                "evaluation_report": "checkpoints/evaluation_report.md",
                "ranked_factors": "checkpoints/ranked_factors.json",
                "base_factors_diff": "checkpoints/base_factors.diff",
                "status": "checkpoints/status.json",
            },
            "result_outputs": {
                "factors_list": f"{result_prefix}/factors_list.json",
                "factor_cal_results": f"{result_prefix}/factor_cal_results.json",
                "cal_report": f"{result_prefix}/cal_report.md",
                "evaluation_report": f"{result_prefix}/evaluation_report.md",
                "ranked_factors": f"{result_prefix}/ranked_factors.json",
                "base_factors_diff": f"{result_prefix}/base_factors.diff",
                "cal_visualizations": f"{result_prefix}/cal_visualizations.json",
                "status": f"{result_prefix}/status.json",
            },
            "summary": {},
        }

    def _agent_started(
        self,
        status: dict[str, Any],
        iteration: int,
        agent: str,
        step: str,
    ) -> None:
        self._checkpoint_status(
            status,
            current_stage="running",
            current_iteration=iteration,
            current_agent=agent,
            current_step=step,
            last_event="agent_started",
            trace_event={
                "time": datetime.now().isoformat(timespec="seconds"),
                "iteration": iteration,
                "agent": agent,
                "step": step,
                "status": "started",
            },
        )

    def _agent_completed(
        self,
        status: dict[str, Any],
        iteration: int,
        agent: str,
        step: str,
        summary: dict[str, Any],
    ) -> None:
        self._checkpoint_status(
            status,
            current_stage="running",
            current_iteration=iteration,
            current_agent=agent,
            current_step=step,
            last_event="agent_completed",
            last_success_agent=agent,
            trace_event={
                "time": datetime.now().isoformat(timespec="seconds"),
                "iteration": iteration,
                "agent": agent,
                "step": step,
                "status": "completed",
                "summary": summary,
            },
        )

    def _agent_failed(self, status: dict[str, Any], exc: Exception) -> None:
        failure = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "failed_iteration": status.get("current_iteration"),
            "failed_agent": status.get("current_agent"),
            "failed_stage": status.get("current_step"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        self._checkpoint_status(
            status,
            current_stage="failed",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            last_event="agent_failed",
            failure=failure,
            trace_event={
                **failure,
                "iteration": failure["failed_iteration"],
                "agent": failure["failed_agent"],
                "step": failure["failed_stage"],
                "status": "failed",
            },
        )

    def _checkpoint_output_failed(self, status: dict[str, Any], exc: Exception) -> None:
        failure = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "failed_iteration": status.get("current_iteration"),
            "failed_agent": status.get("current_agent"),
            "failed_stage": status.get("current_step"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        self._checkpoint_status(
            status,
            current_stage="failed",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            last_event="checkpoint_output_failed",
            failure=failure,
            trace_event={
                **failure,
                "iteration": failure["failed_iteration"],
                "agent": failure["failed_agent"],
                "step": failure["failed_stage"],
                "status": "failed",
            },
        )

    def _checkpoint_status(
        self,
        status: dict[str, Any],
        **updates: Any,
    ) -> None:
        trace_event = updates.pop("trace_event", None)
        status.update(updates)
        if trace_event:
            status.setdefault("agent_run_trace", []).append(trace_event)
        self._write_json("status.json", status)

    @staticmethod
    def _attach_calculation_repairs(
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        by_name = {item.get("name"): item for item in calculation.get("results", [])}
        updated: list[dict[str, Any]] = []
        for factor in factors:
            item = by_name.get(factor.get("name")) or {}
            repair = item.get("repair")
            if not repair:
                updated.append(factor)
                continue
            enriched = dict(factor)
            enriched["calculation_repair"] = {
                "attempted": repair.get("attempted"),
                "success": repair.get("success"),
                "original_expression": item.get("original_expression") or factor.get("expression"),
                "repaired_expression": item.get("expression") if repair.get("success") else None,
                "error_before_repair": repair.get("error_before_repair"),
                "error_after_repair": repair.get("error_after_repair"),
                "reason": repair.get("reason"),
            }
            updated.append(enriched)
        return updated

    def _attach_research_sources(
        self,
        factors: list[dict[str, Any]],
        iteration: int,
        optimizer_feedback: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                **factor,
                "research_source": self._build_research_source(factor, iteration, optimizer_feedback),
            }
            for factor in factors
        ]

    def _attach_sources_to_evaluation(
        self,
        evaluation: dict[str, Any],
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
    ) -> dict[str, Any]:
        ranked = []
        for item in evaluation.get("ranked_factors") or []:
            source = self._find_research_source(item, factors, calculation)
            family_tag = self._find_family_tag(item, factors, calculation)
            family_rationale = self._find_family_rationale(item, factors, calculation)
            ranked.append(
                {
                    **item,
                    "family_tag": family_tag,
                    "family_rationale": family_rationale,
                    "research_source": source,
                }
            )
        return {**evaluation, "ranked_factors": ranked}

    def _find_family_tag(
        self,
        ranked_factor: dict[str, Any],
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
    ) -> str:
        ranked_keys = self._factor_match_keys(ranked_factor)
        for factor in factors:
            if ranked_keys & self._factor_match_keys(factor):
                identity = factor.get("research_identity") or {}
                return factor.get("family_tag") or identity.get("family_tag")
        for item in calculation.get("results", []):
            if ranked_keys & self._factor_match_keys(item):
                identity = item.get("research_identity") or {}
                return item.get("family_tag") or identity.get("family_tag")
        return ranked_factor.get("family_tag")

    def _find_family_rationale(
        self,
        ranked_factor: dict[str, Any],
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
    ) -> str | None:
        ranked_keys = self._factor_match_keys(ranked_factor)
        for factor in factors:
            if ranked_keys & self._factor_match_keys(factor):
                identity = factor.get("research_identity") or {}
                return factor.get("family_rationale") or identity.get("family_rationale")
        for item in calculation.get("results", []):
            if ranked_keys & self._factor_match_keys(item):
                identity = item.get("research_identity") or {}
                return item.get("family_rationale") or identity.get("family_rationale")
        return ranked_factor.get("family_rationale")

    def _find_research_source(
        self,
        ranked_factor: dict[str, Any],
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
    ) -> dict[str, Any]:
        ranked_keys = self._factor_match_keys(ranked_factor)
        for factor in factors:
            if ranked_keys & self._factor_match_keys(factor):
                return factor.get("research_source") or self._unattributed_source()

        for item in calculation.get("results", []):
            if not ranked_keys & self._factor_match_keys(item):
                continue
            for factor in factors:
                if self._factor_match_keys(item) & self._factor_match_keys(factor):
                    return factor.get("research_source") or self._unattributed_source()
        return self._unattributed_source()

    def _build_research_source(
        self,
        factor: dict[str, Any],
        iteration: int,
        optimizer_feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        identity = factor.get("research_identity") or {}
        strategy = identity.get("strategy")
        parent_factors = self._unique_texts(
            [
                *(identity.get("parent_factors") or []),
                *self._base_factor_refs(factor.get("expression") or ""),
            ]
        )
        source_note = (
            identity.get("hypothesis")
            or identity.get("optimizer_instruction")
            or factor.get("logic")
        )
        return {
            "iteration": iteration,
            "source_type": self._research_source_type(strategy, parent_factors, optimizer_feedback, source_note),
            "strategy": strategy,
            "parent_factors": parent_factors,
            "feedback_basis": self._feedback_basis(optimizer_feedback),
            "source_note": source_note,
        }

    @staticmethod
    def _research_source_type(
        strategy: str | None,
        parent_factors: list[str],
        optimizer_feedback: dict[str, Any] | None,
        source_note: str | None,
    ) -> str:
        if strategy:
            return strategy
        if parent_factors or optimizer_feedback or source_note:
            return "feedback_guided"
        return "unattributed"

    @staticmethod
    def _feedback_basis(optimizer_feedback: dict[str, Any] | None) -> dict[str, Any] | None:
        if not optimizer_feedback:
            return None
        return {
            "diagnostics": optimizer_feedback.get("diagnostics") or [],
            "next_round_plan": optimizer_feedback.get("next_round_plan") or [],
            "candidate_blueprints": optimizer_feedback.get("candidate_blueprints") or [],
            "promote_patterns": optimizer_feedback.get("promote_patterns") or [],
            "reject_patterns": optimizer_feedback.get("reject_patterns") or [],
            "guardrails": optimizer_feedback.get("guardrails") or [],
        }

    @staticmethod
    def _base_factor_refs(expression: str) -> list[str]:
        return re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)

    @staticmethod
    def _factor_match_keys(factor: dict[str, Any]) -> set[str]:
        keys = {
            factor.get("name"),
            factor.get("expression"),
            factor.get("original_expression"),
        }
        repair = factor.get("calculation_repair") or {}
        keys.update(
            {
                repair.get("original_expression"),
                repair.get("repaired_expression"),
            }
        )
        return {str(item) for item in keys if item}

    @staticmethod
    def _unique_texts(items: list[Any]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not item:
                continue
            text = str(item)
            if text in seen:
                continue
            seen.add(text)
            unique.append(text)
        return unique

    @staticmethod
    def _unattributed_source() -> dict[str, Any]:
        return {
            "iteration": None,
            "source_type": "unattributed",
            "strategy": None,
            "parent_factors": [],
            "feedback_basis": None,
            "source_note": None,
        }

    @staticmethod
    def _without_research_source(factor: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in factor.items() if key != "research_source"}

    @staticmethod
    def _init_research_memory() -> dict[str, Any]:
        return {
            "active_hypotheses": [],
            "promoted_patterns": [],
            "rejected_patterns": [],
            "watchlist_factors": [],
            "refinement_blocked": [],
            "factor_families": {
                "family_counts": {},
                "factor_family_map": {},
                "family_concentration": {},
                "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
            },
            "family_exploration_blocked": [],
        }

    @staticmethod
    def _init_round_memory(iteration: int) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "ideation": {"candidates": []},
            "calculation": {"success_count": 0, "failure_count": 0, "results": []},
            "evaluation": {"ranked_factors": [], "diagnostics": {}},
            "optimizer_advice": {},
            "memory_delta": {},
        }

    def _hook_ideation(self, round_memory: dict[str, Any], factors: list[dict[str, Any]]) -> None:
        round_memory["ideation"] = {
            "candidates": self._compact_candidates(factors),
        }

    def _hook_calculation(self, round_memory: dict[str, Any], calculation: dict[str, Any]) -> None:
        round_memory["calculation"] = self._compact_calculation(calculation)

    def _hook_evaluation(self, round_memory: dict[str, Any], evaluation: dict[str, Any]) -> None:
        round_memory["evaluation"] = {
            "factor_count": evaluation.get("factor_count"),
            "effective_count": evaluation.get("effective_count"),
            "excellent_count": evaluation.get("excellent_count"),
            "diagnostics": evaluation.get("diagnostics") or {},
            "llm_analysis": evaluation.get("llm_analysis") or {},
            "optimizer_feedback_hook": evaluation.get("optimizer_feedback_hook") or {},
            "ranked_factors": self._compact_ranked(evaluation.get("ranked_factors") or []),
        }

    @staticmethod
    def _hook_optimizer(round_memory: dict[str, Any], optimizer_feedback: dict[str, Any]) -> None:
        round_memory["optimizer_advice"] = {
            "diagnostics": optimizer_feedback.get("diagnostics") or [],
            "next_round_plan": optimizer_feedback.get("next_round_plan") or [],
            "guardrails": optimizer_feedback.get("guardrails") or [],
            "refinement_blocked": SingleRunFactorMiningPipeline._merge_refinement_blocked(
                optimizer_feedback.get("refinement_blocked") or [],
                round_memory,
            ),
            "family_exploration_blocked": SingleRunFactorMiningPipeline._merge_family_exploration_blocked(
                optimizer_feedback.get("family_exploration_blocked") or [],
                round_memory,
            ),
            "reject_patterns": optimizer_feedback.get("reject_patterns") or [],
            "promote_patterns": optimizer_feedback.get("promote_patterns") or [],
        }

    @staticmethod
    def _merge_refinement_blocked(
        optimizer_blocked: list[dict[str, Any]],
        round_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        blocked = list(optimizer_blocked)
        if round_memory.get("iteration") == 1:
            for item in ((round_memory.get("evaluation") or {}).get("ranked_factors") or []):
                if not SingleRunFactorMiningPipeline._is_poor_first_round_factor(item):
                    continue
                blocked.append(
                    {
                        "name": item.get("name"),
                        "reason": "first_round_poor_evaluation",
                        "metric_evidence": {
                            "score": item.get("score"),
                            "quality": item.get("quality"),
                            "mean_ic": item.get("mean_ic"),
                            "icir": item.get("icir"),
                            "ic_win_rate": item.get("ic_win_rate"),
                            "long_short_return": item.get("long_short_return"),
                        },
                        "blocked_actions": ["mutation", "crossover", "refinement"],
                    }
                )
        return SingleRunFactorMiningPipeline._limit_unique(blocked, 24)

    @staticmethod
    def _is_poor_first_round_factor(item: dict[str, Any]) -> bool:
        if item.get("quality") == "weak":
            return True
        score = item.get("score")
        if score is not None and score < 30:
            return True
        return not item.get("is_effective")

    def _build_memory_delta(self, round_memory: dict[str, Any]) -> dict[str, Any]:
        optimizer_advice = round_memory.get("optimizer_advice") or {}
        return {
            "active_hypotheses": optimizer_advice.get("next_round_plan") or [],
            "promoted_patterns": optimizer_advice.get("promote_patterns") or [],
            "rejected_patterns": optimizer_advice.get("reject_patterns") or [],
            "refinement_blocked": optimizer_advice.get("refinement_blocked") or [],
            "factor_family_updates": self._build_factor_family_updates(round_memory),
            "family_exploration_blocked": optimizer_advice.get("family_exploration_blocked") or [],
            "watchlist_factors": self._build_watchlist(round_memory),
        }

    def _apply_memory_delta(self, research_memory: dict[str, Any], memory_delta: dict[str, Any]) -> None:
        active_hypotheses = memory_delta.get("active_hypotheses") or []
        if active_hypotheses:
            research_memory["active_hypotheses"] = self._limit_unique(active_hypotheses, 8)
        research_memory["promoted_patterns"] = self._limit_unique(
            [*research_memory.get("promoted_patterns", []), *memory_delta.get("promoted_patterns", [])],
            16,
        )
        research_memory["rejected_patterns"] = self._limit_unique(
            [*research_memory.get("rejected_patterns", []), *memory_delta.get("rejected_patterns", [])],
            16,
        )
        research_memory["refinement_blocked"] = self._limit_unique(
            [*research_memory.get("refinement_blocked", []), *memory_delta.get("refinement_blocked", [])],
            24,
        )
        self._apply_factor_family_updates(research_memory, memory_delta.get("factor_family_updates") or [])
        research_memory["family_exploration_blocked"] = self._limit_unique(
            [
                *research_memory.get("family_exploration_blocked", []),
                *memory_delta.get("family_exploration_blocked", []),
                *self._detect_family_concentration_blocks_from_memory(research_memory),
            ],
            16,
        )
        research_memory["watchlist_factors"] = self._limit_unique(
            [*research_memory.get("watchlist_factors", []), *memory_delta.get("watchlist_factors", [])],
            12,
        )

    @staticmethod
    def _build_factor_family_updates(round_memory: dict[str, Any]) -> list[dict[str, Any]]:
        updates = []
        for item in ((round_memory.get("ideation") or {}).get("candidates") or []):
            name = item.get("name")
            family_tag = item.get("family_tag") or (item.get("research_identity") or {}).get("family_tag")
            if not name:
                continue
            updates.append(
                {
                    "name": name,
                    "family_tag": family_tag,
                    "family_rationale": item.get("family_rationale")
                    or (item.get("research_identity") or {}).get("family_rationale"),
                    "iteration": round_memory.get("iteration"),
                    "category": item.get("category"),
                    "expression": item.get("expression"),
                }
            )
        return updates

    @staticmethod
    def _apply_factor_family_updates(research_memory: dict[str, Any], updates: list[dict[str, Any]]) -> None:
        families = research_memory.setdefault(
            "factor_families",
            {
                "family_counts": {},
                "factor_family_map": {},
                "family_concentration": {},
                "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
            },
        )
        factor_family_map = families.setdefault("factor_family_map", {})
        for item in updates:
            name = item.get("name")
            family_tag = item.get("family_tag")
            if not family_tag:
                continue
            if not name or name in factor_family_map:
                continue
            factor_family_map[name] = {
                "family_tag": family_tag,
                "iteration": item.get("iteration"),
                "category": item.get("category"),
                "expression": item.get("expression"),
            }
        counts: dict[str, int] = {}
        for item in factor_family_map.values():
            family_tag = item.get("family_tag")
            if not family_tag:
                continue
            counts[family_tag] = counts.get(family_tag, 0) + 1
        total = sum(counts.values())
        families["family_counts"] = counts
        families["factor_count"] = total
        families["threshold"] = SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD
        families["family_concentration"] = {
            key: round(value / total, 6) for key, value in counts.items()
        } if total else {}

    @staticmethod
    def _merge_family_exploration_blocked(
        optimizer_blocked: list[dict[str, Any]],
        round_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        blocked = list(optimizer_blocked)
        blocked.extend(SingleRunFactorMiningPipeline._detect_family_concentration_blocks(round_memory))
        return SingleRunFactorMiningPipeline._limit_unique(blocked, 16)

    @staticmethod
    def _detect_family_concentration_blocks(round_memory: dict[str, Any]) -> list[dict[str, Any]]:
        updates = SingleRunFactorMiningPipeline._build_factor_family_updates(round_memory)
        counts: dict[str, int] = {}
        for item in updates:
            family_tag = item.get("family_tag")
            if not family_tag:
                continue
            counts[family_tag] = counts.get(family_tag, 0) + 1
        total = sum(counts.values())
        if total < SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FACTORS:
            return []
        blocked = []
        for family_tag, count in counts.items():
            concentration = count / total if total else 0
            if (
                count >= SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FAMILY_COUNT
                and concentration > SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD
            ):
                blocked.append(
                    {
                        "family_tag": family_tag,
                        "concentration": round(concentration, 6),
                        "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
                        "reason": "family_concentration_exceeded",
                        "blocked_actions": ["new_hypothesis", "mutation", "crossover", "refinement"],
                    }
                )
        return blocked

    @staticmethod
    def _detect_family_concentration_blocks_from_memory(research_memory: dict[str, Any]) -> list[dict[str, Any]]:
        families = research_memory.get("factor_families") or {}
        counts = families.get("family_counts") or {}
        total = sum(counts.values())
        if total < SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FACTORS:
            return []
        blocked = []
        for family_tag, count in counts.items():
            concentration = count / total if total else 0
            if (
                count >= SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FAMILY_COUNT
                and concentration > SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD
            ):
                blocked.append(
                    {
                        "family_tag": family_tag,
                        "concentration": round(concentration, 6),
                        "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
                        "reason": "family_concentration_exceeded",
                        "blocked_actions": ["new_hypothesis", "mutation", "crossover", "refinement"],
                    }
                )
        return blocked

    @staticmethod
    def _build_watchlist(round_memory: dict[str, Any]) -> list[dict[str, Any]]:
        ranked = (round_memory.get("evaluation") or {}).get("ranked_factors") or []
        watchlist = [
            item
            for item in ranked
            if item.get("score") is not None and not item.get("is_effective")
        ]
        watchlist = sorted(watchlist, key=lambda item: item.get("score") or 0, reverse=True)
        return [
            {
                "name": item.get("name"),
                "expression": item.get("expression"),
                "category": item.get("category"),
                "score": item.get("score"),
                "quality": item.get("quality"),
                "direction": item.get("direction"),
            }
            for item in watchlist[:5]
        ]

    @staticmethod
    def _summarize_optimizer_feedback(optimizer_feedback: dict[str, Any]) -> dict[str, Any]:
        return {
            "diagnostics_count": len(optimizer_feedback.get("diagnostics") or []),
            "next_round_plan_count": len(optimizer_feedback.get("next_round_plan") or []),
            "candidate_blueprints_count": len(optimizer_feedback.get("candidate_blueprints") or []),
            "guardrails_count": len(optimizer_feedback.get("guardrails") or []),
            "refinement_blocked_count": len(optimizer_feedback.get("refinement_blocked") or []),
            "family_exploration_blocked_count": len(optimizer_feedback.get("family_exploration_blocked") or []),
            "reject_patterns_count": len(optimizer_feedback.get("reject_patterns") or []),
            "promote_patterns_count": len(optimizer_feedback.get("promote_patterns") or []),
        }

    @staticmethod
    def _optimizer_tool_calls(
        optimizer_result: dict[str, Any],
        iteration: int,
        stage: str,
    ) -> list[dict[str, Any]]:
        calls = []
        for item in optimizer_result.get("tool_trace") or []:
            calls.append(
                {
                    "time": optimizer_result.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                    "agent": "FactorOptimizer",
                    "called": f"FactorToolbox.{item.get('tool')}",
                    "input": {
                        "iteration": iteration,
                        "stage": stage,
                        **(item.get("arguments") or {}),
                    },
                    "result": item.get("result"),
                }
            )
        return calls

    @staticmethod
    def _limit_unique(items: list[Any], limit: int) -> list[Any]:
        unique: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[-limit:]

    def _build_research_context(
        self,
        iteration: int,
        factors: list[dict[str, Any]],
        calculation: dict[str, Any],
        evaluation: dict[str, Any],
        history: list[dict[str, Any]],
        history_performance: list[dict[str, Any]],
        research_memory: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "target_factor_count": self.factors_per_round,
            "research_memory": research_memory,
            "candidate_factors": self._compact_candidates(factors),
            "calculation_summary": self._compact_calculation(calculation),
            "evaluation_summary": {
                "factor_count": evaluation.get("factor_count"),
                "effective_count": evaluation.get("effective_count"),
                "excellent_count": evaluation.get("excellent_count"),
                "diagnostics": evaluation.get("diagnostics"),
                "llm_analysis": evaluation.get("llm_analysis"),
                "optimizer_feedback_hook": evaluation.get("optimizer_feedback_hook"),
                "ranked_factors": self._compact_ranked(evaluation.get("ranked_factors") or []),
            },
            "factor_family_concentration": self._project_factor_family_concentration(research_memory, factors),
            "base_factor_pool": self._compact_base_factors(),
            "history": {
                "recent_factor_names": [item.get("name") for item in history[-12:]],
                "recent_performance": self._compact_ranked(history_performance[-12:]),
            },
        }

    @staticmethod
    def _compact_candidates(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def identity(item: dict[str, Any]) -> dict[str, Any]:
            return item.get("research_identity") or {}

        return [
            {
                "name": item.get("name"),
                "expression": item.get("expression"),
                "category": item.get("category"),
                "family_tag": item.get("family_tag") or identity(item).get("family_tag"),
                "family_rationale": item.get("family_rationale") or identity(item).get("family_rationale"),
                "logic": item.get("logic"),
                "research_identity": {
                    "family_tag": identity(item).get("family_tag") or item.get("family_tag"),
                    "family_rationale": identity(item).get("family_rationale")
                    or item.get("family_rationale"),
                    "strategy": identity(item).get("strategy"),
                    "parent_factors": identity(item).get("parent_factors"),
                    "hypothesis": identity(item).get("hypothesis"),
                    "expected_direction": identity(item).get("expected_direction"),
                    "risk_note": identity(item).get("risk_note"),
                    "optimizer_instruction": identity(item).get("optimizer_instruction"),
                },
                "tool_validation": item.get("tool_validation"),
                "quick_evaluation": item.get("quick_evaluation"),
            }
            for item in factors
        ]

    @staticmethod
    def _compact_calculation(calculation: dict[str, Any]) -> dict[str, Any]:
        return {
            "success_count": calculation.get("success_count"),
            "failure_count": calculation.get("failure_count"),
            "results": [
                {
                    "name": item.get("name"),
                    "expression": item.get("expression"),
                    "category": item.get("category"),
                    "success": item.get("success"),
                    "latest_date": item.get("latest_date"),
                    "coverage": (item.get("stats") or {}).get("latest_coverage"),
                    "mean": (item.get("stats") or {}).get("mean"),
                    "std": (item.get("stats") or {}).get("std"),
                    "error": item.get("error"),
                    "performance": item.get("performance"),
                }
                for item in calculation.get("results", [])
            ],
        }

    @staticmethod
    def _compact_ranked(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []
        for item in ranked[:12]:
            metrics = item.get("metrics") or {}
            compact.append(
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "family_tag": item.get("family_tag"),
                    "family_rationale": item.get("family_rationale"),
                    "expression": item.get("expression"),
                    "score": item.get("score"),
                    "quality": item.get("quality"),
                    "direction": item.get("direction"),
                    "is_effective": item.get("is_effective"),
                    "mean_ic": metrics.get("mean_ic"),
                    "icir": metrics.get("icir"),
                    "ic_win_rate": metrics.get("ic_win_rate"),
                    "long_short_return": metrics.get("long_short_return"),
                    "directional_monotonicity": (item.get("stratified_backtest") or {}).get("directional_monotonicity"),
                    "multi_period_metrics": item.get("multi_period_metrics"),
                }
            )
        return compact

    @staticmethod
    def _project_factor_family_concentration(
        research_memory: dict[str, Any],
        factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        families = research_memory.get("factor_families") or {}
        existing_map = dict(families.get("factor_family_map") or {})
        for item in factors:
            name = item.get("name")
            if not name or name in existing_map:
                continue
            identity = item.get("research_identity") or {}
            existing_map[name] = {
                "family_tag": item.get("family_tag") or identity.get("family_tag"),
                "category": item.get("category"),
                "expression": item.get("expression"),
            }
        counts: dict[str, int] = {}
        for item in existing_map.values():
            family_tag = item.get("family_tag")
            if not family_tag:
                continue
            counts[family_tag] = counts.get(family_tag, 0) + 1
        total = sum(counts.values())
        concentration = {
            key: round(value / total, 6) for key, value in counts.items()
        } if total else {}
        blocked_candidates = []
        if total >= SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FACTORS:
            for family_tag, count in counts.items():
                ratio = count / total if total else 0
                if (
                    count >= SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_MIN_FAMILY_COUNT
                    and ratio > SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD
                ):
                    blocked_candidates.append(
                        {
                            "family_tag": family_tag,
                            "concentration": round(ratio, 6),
                            "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
                            "reason": "family_concentration_exceeded",
                            "blocked_actions": ["new_hypothesis", "mutation", "crossover", "refinement"],
                        }
                    )
        return {
            "factor_count": total,
            "family_counts": counts,
            "family_concentration": concentration,
            "threshold": SingleRunFactorMiningPipeline.FAMILY_CONCENTRATION_THRESHOLD,
            "blocked_candidates": blocked_candidates,
        }

    def _compact_base_factors(self) -> dict[str, Any]:
        base_path = self.checkpoint_dir / "base_factors.json"
        if not base_path.exists():
            return {"factor_count": 0, "factors": []}
        payload = json.loads(base_path.read_text(encoding="utf-8"))
        definitions = payload.get("factor_definitions") or []
        return {
            "factor_count": len(definitions),
            "categories": payload.get("factor_categories"),
            "factors": [
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "formula": item.get("formula"),
                    "direction": item.get("direction"),
                    "description": item.get("description"),
                }
                for item in definitions
            ],
        }

    def validate_optimizer(self, target_count: int = 3) -> dict[str, Any]:
        evaluation_path = self.checkpoint_dir / "evaluation_report.md"
        ranked_path = self.checkpoint_dir / "ranked_factors.json"
        base_path = self.checkpoint_dir / "base_factors.json"
        report = evaluation_path.read_text(encoding="utf-8")
        ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
        base = json.loads(base_path.read_text(encoding="utf-8"))
        result = self.optimizer.optimize_from_report(
            evaluation_report=report,
            ranked_factors=ranked,
            base_factors=base,
            target_count=target_count,
        )
        marker = "\n## Optimizer Agent Validation\n"
        base_report = report.split(marker, 1)[0].rstrip()
        self._write_text(
            "evaluation_report.md",
            base_report + self.optimizer.build_report_section(result),
        )
        return result

    def _build_base_factor_diff(self, ranked_payload: dict[str, Any]) -> str:
        base_path = self.checkpoint_dir / "base_factors.json"
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        base_definitions = base_payload.get("factor_definitions", [])
        base_names = {item.get("name") for item in base_definitions}
        ranked_factors = ranked_payload.get("ranked_factors", [])
        proposed = [
            item
            for item in ranked_factors
            if item.get("name") not in base_names and item.get("is_effective")
        ]
        watchlist = [
            item
            for item in ranked_factors
            if item.get("name") not in base_names and not item.get("is_effective")
        ][:10]

        lines = [
            "# Base Factors Diff",
            "",
            f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Source base factor count: `{len(base_definitions)}`",
            f"- Ranked candidate count: `{len(ranked_factors)}`",
            f"- Proposed additions: `{len(proposed)}`",
            "",
            "## Proposed Additions",
            "",
        ]

        if proposed:
            for item in proposed:
                metrics = item.get("metrics", {})
                lines.extend(
                    [
                        f"+ {item.get('name')}",
                        f"+   category: {item.get('category', '-')}",
                        f"+   expression: {item.get('expression', '-')}",
                        f"+   score: {item.get('score', '-')}",
                        f"+   quality: {item.get('quality', '-')}",
                        f"+   direction: {item.get('direction', '-')}",
                        f"+   mean_ic: {metrics.get('mean_ic', '-')}",
                        f"+   icir: {metrics.get('icir', '-')}",
                        f"+   ic_win_rate: {metrics.get('ic_win_rate', '-')}",
                        f"+   long_short_return: {metrics.get('long_short_return', '-')}",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "No ranked candidate met the effective-factor threshold, so no base-factor addition is proposed.",
                    "",
                ]
            )

        lines.extend(
            [
                "## Watchlist",
                "",
                "The following high-ranked candidates were calculated and evaluated but are not proposed for the base pool yet.",
                "",
            ]
        )
        if watchlist:
            for item in watchlist:
                metrics = item.get("metrics", {})
                lines.append(
                    "- {name} | score={score} | quality={quality} | mean_ic={mean_ic} | icir={icir}".format(
                        name=item.get("name", "-"),
                        score=item.get("score", "-"),
                        quality=item.get("quality", "-"),
                        mean_ic=metrics.get("mean_ic", "-"),
                        icir=metrics.get("icir", "-"),
                    )
                )
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def _prepare_result_dir(self, run_id: str) -> tuple[str, Path]:
        self._ensure_checkpoint_workspace()
        results_root = Path("results")
        result_dir = results_root / run_id
        actual_run_id = run_id
        suffix = 1
        while result_dir.exists():
            actual_run_id = f"{run_id}_{suffix:02d}"
            result_dir = results_root / actual_run_id
            suffix += 1
        result_dir.mkdir(parents=True, exist_ok=False)
        for source in self.checkpoint_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, result_dir / source.name)
        return actual_run_id, result_dir

    def _ensure_checkpoint_workspace(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.CHECKPOINT_JSON_FILES:
            path = self.checkpoint_dir / filename
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")
        for filename in self.CHECKPOINT_TEXT_FILES:
            path = self.checkpoint_dir / filename
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.result_dir is not None:
            (self.result_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _write_text(self, filename: str, payload: str) -> None:
        path = self.checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        if self.result_dir is not None:
            (self.result_dir / filename).write_text(payload, encoding="utf-8")

    def _write_cal_visualizations(self, payload: dict[str, Any]) -> None:
        if self.result_dir is None:
            return
        manifest = payload.get("manifest") or {}
        files = payload.get("files") or {}
        (self.result_dir / "cal_visualizations.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for filename, content in files.items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid visualization path: {filename}")
            path = self.result_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _to_posix(path: Path) -> str:
        return path.as_posix()
