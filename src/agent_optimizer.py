from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_tools import FactorToolbox
from .llm_client import OpenAICompatibleLLMClient, ToolSpec


class FactorOptimizerAgent:
    name = "FactorOptimizer"

    def __init__(
        self,
        toolbox: FactorToolbox,
        llm_client: OpenAICompatibleLLMClient | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.llm_client = llm_client or OpenAICompatibleLLMClient()

    def optimize_from_report(
        self,
        evaluation_report: str,
        ranked_factors: dict[str, Any] | None = None,
        base_factors: dict[str, Any] | None = None,
        target_count: int = 3,
    ) -> dict[str, Any]:
        system_prompt = self.system_prompt()
        user_prompt = self.user_prompt(
            evaluation_report=evaluation_report,
            ranked_factors=ranked_factors,
            base_factors=base_factors,
            target_count=target_count,
        )
        content, trace = self._run_tool_optimization(system_prompt, user_prompt, target_count)
        parsed = self._parse_json(content)
        return {
            "schema_version": "1.0",
            "agent": self.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_count": target_count,
            "generation_mode": "tool_call",
            "tool_trace": trace,
            "raw_response": content,
            "optimization": parsed,
        }

    def optimize_from_evaluation(
        self,
        evaluation: dict[str, Any],
        history: list[dict[str, Any]],
        target_count: int = 3,
    ) -> dict[str, Any]:
        report = self._evaluation_to_brief(evaluation, history)
        return self.optimize_from_report(
            evaluation_report=report,
            ranked_factors=evaluation,
            base_factors=None,
            target_count=target_count,
        )

    def optimize_from_context(
        self,
        research_context: dict[str, Any],
        target_count: int = 3,
    ) -> dict[str, Any]:
        system_prompt = self.system_prompt()
        user_prompt = self.context_prompt(research_context, target_count)
        content, trace = self._run_tool_optimization(system_prompt, user_prompt, target_count)
        parsed = self._parse_json(content)
        return {
            "schema_version": "1.0",
            "agent": self.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "target_count": target_count,
            "generation_mode": "tool_call",
            "tool_trace": trace,
            "raw_response": content,
            "optimization": parsed,
        }

    def _run_tool_optimization(
        self,
        system_prompt: str,
        user_prompt: str,
        target_count: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        messages, trace = self.llm_client.chat_with_tools_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self._tools(),
            max_iterations=max(12, target_count * 6),
        )
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return message.get("content") or "", trace
        return "", trace

    def _tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="query_base_factor_pool",
                description="查询基础因子池，确认可用基础因子名称、类别、公式、方向和经济含义。",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["momentum", "reversal", "volatility", "volume_price", "trend", "liquidity"],
                        },
                        "limit": {"type": "integer", "default": 30},
                    },
                    "required": [],
                },
                handler=self.toolbox.query_base_factor_pool,
            ),
            ToolSpec(
                name="describe_data_fields",
                description="查询当前数据面板可用字段、覆盖率和可用于因子表达式的字段。",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=self.toolbox.describe_data_fields,
            ),
            ToolSpec(
                name="validate_and_compute_factor",
                description="验证 Optimizer 提出的候选表达式是否能被算子引擎计算，并返回覆盖率和最新截面统计。",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "expression": {"type": "string"},
                        "category": {"type": "string", "enum": ["momentum", "reversal", "volatility", "volume_price", "trend", "liquidity"]},
                        "logic": {"type": "string"},
                        "family_tag": {
                            "type": "string",
                            "enum": [
                                "price_momentum",
                                "price_reversal",
                                "volatility_risk",
                                "volume_price_confirmation",
                                "liquidity_pressure",
                                "trend_strength",
                                "flow_attention",
                                "cross_family_composite",
                                "other",
                            ],
                        },
                        "family_rationale": {"type": "string"},
                        "strategy": {"type": "string", "enum": ["mutation", "crossover", "refinement", "inversion", "new_hypothesis"]},
                        "parent_factors": {"type": "array", "items": {"type": "string"}},
                        "hypothesis": {"type": "string"},
                        "expected_direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "risk_note": {"type": "string"},
                        "optimizer_instruction": {"type": "string"},
                    },
                    "required": ["name", "expression", "category", "logic", "family_tag", "family_rationale"],
                },
                handler=self.toolbox.validate_and_compute_factor_dict,
            ),
            ToolSpec(
                name="evaluate_factor_quick",
                description="快速评估候选优化表达式，返回多周期 IC/ICIR、分层回测、单调性、胜率和多空收益。",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "horizon": {"type": "integer", "default": 5},
                        "recent_days": {"type": "integer", "default": 252},
                        "quantile_groups": {"type": "integer", "default": 5},
                    },
                    "required": ["expression"],
                },
                handler=self.toolbox.evaluate_factor_quick,
            ),
            ToolSpec(
                name="analyze_factor_orthogonality",
                description="检查 Optimizer 提出的候选表达式与基础因子池、父因子或历史参考表达式的相关性。",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "reference_expressions": {"type": "array", "items": {"type": "string"}},
                        "recent_days": {"type": "integer", "default": 252},
                        "max_base_factors": {"type": "integer", "default": 20},
                    },
                    "required": ["expression"],
                },
                handler=self.toolbox.analyze_factor_orthogonality,
            ),
        ]

    def build_report_section(self, result: dict[str, Any]) -> str:
        optimization = result.get("optimization") or {}
        next_round = optimization.get("next_round_plan") or []
        diagnostics = optimization.get("diagnostics") or []
        candidates = optimization.get("candidate_blueprints") or []
        guardrails = optimization.get("guardrails") or []
        blocked = optimization.get("refinement_blocked") or []
        family_blocked = optimization.get("family_exploration_blocked") or []

        lines = [
            "",
            "## Optimizer 智能体验证",
            "",
            f"- Agent: `{self.name}`",
            f"- 生成时间: `{result.get('created_at', '')}`",
            f"- 下一轮目标因子数: `{result.get('target_count', '')}`",
            "- 运行模式: `api_validation_only`",
            "- 是否执行 pipeline: `false`",
            "",
            "### 诊断",
            "",
        ]
        lines.extend(self._bullet_lines(diagnostics))
        lines.extend(["", "### 下一轮计划", ""])
        lines.extend(self._bullet_lines(next_round))
        lines.extend(["", "### 候选蓝图", ""])
        if candidates:
            for item in candidates:
                lines.extend(
                    [
                        f"- 名称: `{item.get('name', '-')}`",
                        f"  - 策略: `{item.get('strategy', '-')}`",
                        f"  - 父因子: `{', '.join(item.get('parent_factors') or []) or '-'}`",
                        f"  - 假设: {item.get('hypothesis', '-')}",
                        f"  - 表达式意图: `{item.get('expression_intent', '-')}`",
                        f"  - 预期方向: `{item.get('expected_direction', '-')}`",
                        f"  - 风险提示: {item.get('risk_note', '-')}",
                    ]
                )
        else:
            lines.append("- 未返回候选蓝图。")
        lines.extend(["", "### 约束", ""])
        lines.extend(self._bullet_lines(guardrails))
        lines.extend(["", "### 禁止继续改进的因子", ""])
        lines.extend(self._blocked_lines(blocked))
        lines.extend(["", "### 禁止继续探索的因子家族", ""])
        lines.extend(self._family_blocked_lines(family_blocked))
        lines.extend(["", "### 拒绝模式", ""])
        lines.extend(self._bullet_lines(optimization.get("reject_patterns")))
        lines.extend(["", "### 推广模式", ""])
        lines.extend(self._bullet_lines(optimization.get("promote_patterns")))
        lines.extend(["", "### Optimizer 原始响应", "", "```json"])
        lines.append(json.dumps(optimization, ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
        return "\n".join(lines)

    def build_final_report_section(self, result: dict[str, Any]) -> str:
        optimization = result.get("optimization") or {}
        next_round = optimization.get("next_round_plan") or []
        diagnostics = optimization.get("diagnostics") or []
        candidates = optimization.get("candidate_blueprints") or []
        guardrails = optimization.get("guardrails") or []
        blocked = optimization.get("refinement_blocked") or []
        family_blocked = optimization.get("family_exploration_blocked") or []

        lines = [
            "",
            "## 最终研究建议",
            "",
            f"- Agent: `{self.name}`",
            f"- 生成时间: `{result.get('created_at', '')}`",
            "- 运行模式: `final_research_advice`",
            "- 是否继续执行 pipeline: `false`",
            "",
            "### 最终诊断",
            "",
        ]
        lines.extend(self._bullet_lines(diagnostics))
        lines.extend(["", "### 建议的下一步研究", ""])
        lines.extend(self._bullet_lines(next_round))
        lines.extend(["", "### 后续运行候选蓝图", ""])
        if candidates:
            for item in candidates:
                lines.extend(
                    [
                        f"- 名称: `{item.get('name', '-')}`",
                        f"  - 策略: `{item.get('strategy', '-')}`",
                        f"  - 父因子: `{', '.join(item.get('parent_factors') or []) or '-'}`",
                        f"  - 假设: {item.get('hypothesis', '-')}",
                        f"  - 表达式意图: `{item.get('expression_intent', '-')}`",
                        f"  - 预期方向: `{item.get('expected_direction', '-')}`",
                        f"  - 风险提示: {item.get('risk_note', '-')}",
                    ]
                )
        else:
            lines.append("- 未返回候选蓝图。")
        lines.extend(["", "### 约束", ""])
        lines.extend(self._bullet_lines(guardrails))
        lines.extend(["", "### 禁止继续改进的因子", ""])
        lines.extend(self._blocked_lines(blocked))
        lines.extend(["", "### 禁止继续探索的因子家族", ""])
        lines.extend(self._family_blocked_lines(family_blocked))
        lines.extend(["", "### 拒绝模式", ""])
        lines.extend(self._bullet_lines(optimization.get("reject_patterns")))
        lines.extend(["", "### 推广模式", ""])
        lines.extend(self._bullet_lines(optimization.get("promote_patterns")))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def system_prompt() -> str:
        return (
            "你是 FactorOptimizer，一个资深 ETF 量化因子研究优化智能体。"
            "你的任务不是随机枚举公式。请阅读评估证据，诊断因子有效或失败的原因，"
            "并设计下一步受控研究动作。只能返回严格 JSON。"
            "你必须使用工具增强优化过程：先查询必要的基础因子或数据字段，"
            "再对准备写入 candidate_blueprints 的表达式意图调用 validate_and_compute_factor；"
            "对可计算表达式调用 evaluate_factor_quick；对有父因子或历史参考的表达式调用 analyze_factor_orthogonality。"
            "除非评估证据或工具返回支持，否则不要声称某个因子已经被证明有效。"
            "请优先基于 IC、ICIR、IC 胜率、多空收益、覆盖率、方向和经济逻辑，"
            "提出可解释的 mutation、crossover、refinement 或 inversion。"
            "第一轮评估效果太差的因子必须进入 refinement_blocked，后续不得围绕这些因子继续改进。"
            "必须根据研究上下文中的 factor_family_concentration 判断因子家族集中度；"
            "当任一家族 concentration > threshold 时，必须在 family_exploration_blocked 中明确禁止继续探索该家族。"
            "最终 JSON 的 candidate_blueprints 中必须体现工具验证后的结论，"
            "risk_note 或 optimizer_instruction 里要说明表达式验证、快速评估或正交性检查的关键结果。"
        )

    @staticmethod
    def user_prompt(
        evaluation_report: str,
        ranked_factors: dict[str, Any] | None,
        base_factors: dict[str, Any] | None,
        target_count: int,
    ) -> str:
        ranked_summary = FactorOptimizerAgent._compact_ranked(ranked_factors or {})
        base_summary = FactorOptimizerAgent._compact_base(base_factors or {})
        return (
            f"请设计下一步受控优化动作，并给出恰好 {target_count} 个候选蓝图。\n\n"
            "在最终输出前，必须用工具验证你要提出的候选表达式："
            "validate_and_compute_factor 检查可计算性，evaluate_factor_quick 快速评估，"
            "analyze_factor_orthogonality 检查与父因子或基础因子的相关性。"
            "无法通过工具验证的蓝图不得作为已验证候选输出，只能写入 reject_patterns 或 guardrails。\n\n"
            "请按以下 schema 返回 JSON：\n"
            "{\n"
            '  "diagnostics": ["..."],\n'
            '  "next_round_plan": ["..."],\n'
            '  "candidate_blueprints": [\n'
            "    {\n"
            '      "name": "...",\n'
            '      "strategy": "mutation|crossover|refinement|inversion",\n'
            '      "parent_factors": ["..."],\n'
            '      "hypothesis": "...",\n'
            '      "expression_intent": "...",\n'
            '      "expected_direction": "positive|negative|neutral",\n'
            '      "risk_note": "..."\n'
            "    }\n"
            "  ],\n"
            '  "guardrails": ["..."],\n'
            '  "refinement_blocked": [\n'
            "    {\n"
            '      "name": "...",\n'
            '      "reason": "...",\n'
            '      "metric_evidence": {"score": 0, "quality": "...", "mean_ic": 0, "icir": 0, "ic_win_rate": 0, "long_short_return": 0},\n'
            '      "blocked_actions": ["mutation", "crossover", "refinement"]\n'
            "    }\n"
            "  ],\n"
            '  "family_exploration_blocked": [\n'
            "    {\n"
            '      "family_tag": "...",\n'
            '      "concentration": 0,\n'
            '      "threshold": 0.35,\n'
            '      "reason": "...",\n'
            '      "blocked_actions": ["new_hypothesis", "mutation", "crossover", "refinement"]\n'
            "    }\n"
            "  ],\n"
            '  "reject_patterns": ["..."],\n'
            '  "promote_patterns": ["..."]\n'
            "}\n\n"
            "如果第一轮因子 quality=weak，或 IC/ICIR/胜率/多空收益没有形成可解释信号，"
            "必须把该因子写入 refinement_blocked；不得再建议围绕其做 mutation、crossover 或 refinement。\n\n"
            "如果家族集中度 concentration > threshold，必须把该家族写入 family_exploration_blocked，"
            "并禁止下一轮继续探索该家族。\n\n"
            "基础因子池摘要：\n"
            f"{base_summary}\n\n"
            "已排序因子指标摘要：\n"
            f"{ranked_summary}\n\n"
            "评估报告 markdown：\n"
            f"{evaluation_report}\n"
        )

    @staticmethod
    def context_prompt(research_context: dict[str, Any], target_count: int) -> str:
        return (
            f"请设计下一步受控优化动作，并给出恰好 {target_count} 个候选蓝图。\n\n"
            "请使用完整研究上下文。上下文包含当前候选因子、计算成功与失败情况、"
            "覆盖率和数值统计、评估指标、评估器诊断、Evaluator LLM 分析、"
            "Evaluator 给 Optimizer 的结构化反馈钩子、基础因子池摘要、"
            "以及历史因子表现。\n\n"
            "必须优先读取 evaluation_summary.optimizer_feedback_hook："
            "refinement_candidates 用于受控精炼，inversion_candidates 用于取反验证，"
            "mutation_candidates 用于变异探索，block_candidates 不得作为下一轮 parent_factors。"
            "同时结合 multi_period_metrics、stratified_backtest 和 directional_monotonicity 判断信号稳定性。\n\n"
            "在最终输出前，必须用工具验证你要提出的候选表达式："
            "validate_and_compute_factor 检查可计算性，evaluate_factor_quick 快速评估，"
            "analyze_factor_orthogonality 检查与父因子、历史候选或基础因子的相关性。"
            "无法通过工具验证的蓝图不得作为已验证候选输出，只能写入 reject_patterns 或 guardrails。\n\n"
            "请按以下 schema 返回 JSON：\n"
            "{\n"
            '  "diagnostics": ["..."],\n'
            '  "next_round_plan": ["..."],\n'
            '  "candidate_blueprints": [\n'
            "    {\n"
            '      "name": "...",\n'
            '      "strategy": "mutation|crossover|refinement|inversion|new_hypothesis",\n'
            '      "parent_factors": ["..."],\n'
            '      "hypothesis": "...",\n'
            '      "expression_intent": "...",\n'
            '      "expected_direction": "positive|negative|neutral",\n'
            '      "risk_note": "...",\n'
            '      "optimizer_instruction": "..."\n'
            "    }\n"
            "  ],\n"
            '  "guardrails": ["..."],\n'
            '  "refinement_blocked": [\n'
            "    {\n"
            '      "name": "...",\n'
            '      "reason": "...",\n'
            '      "metric_evidence": {"score": 0, "quality": "...", "mean_ic": 0, "icir": 0, "ic_win_rate": 0, "long_short_return": 0},\n'
            '      "blocked_actions": ["mutation", "crossover", "refinement"]\n'
            "    }\n"
            "  ],\n"
            '  "family_exploration_blocked": [\n'
            "    {\n"
            '      "family_tag": "...",\n'
            '      "concentration": 0,\n'
            '      "threshold": 0.35,\n'
            '      "reason": "...",\n'
            '      "blocked_actions": ["new_hypothesis", "mutation", "crossover", "refinement"]\n'
            "    }\n"
            "  ],\n"
            '  "reject_patterns": ["..."],\n'
            '  "promote_patterns": ["..."]\n'
            "}\n\n"
            "如果研究上下文显示第一轮因子 quality=weak，或 IC/ICIR/胜率/多空收益没有形成可解释信号，"
            "必须把该因子写入 refinement_blocked；后续蓝图不得把这些因子列为 parent_factors，"
            "也不得围绕其做 mutation、crossover 或 refinement。\n\n"
            "如果 factor_family_concentration 中任一家族 concentration > threshold，"
            "必须把该家族写入 family_exploration_blocked；后续蓝图不得生成该 family_tag，"
            "不得把该家族因子作为 parent_factors，也不得围绕该家族做 new_hypothesis、mutation、crossover 或 refinement。\n\n"
            "研究上下文 JSON：\n"
            f"{json.dumps(research_context, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _compact_ranked(payload: dict[str, Any], limit: int = 8) -> str:
        rows = []
        for item in (payload.get("ranked_factors") or [])[:limit]:
            metrics = item.get("metrics") or {}
            rows.append(
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "expression": item.get("expression"),
                    "score": item.get("score"),
                    "quality": item.get("quality"),
                    "direction": item.get("direction"),
                    "mean_ic": metrics.get("mean_ic"),
                    "icir": metrics.get("icir"),
                    "ic_win_rate": metrics.get("ic_win_rate"),
                    "long_short_return": metrics.get("long_short_return"),
                    "directional_monotonicity": (item.get("stratified_backtest") or {}).get("directional_monotonicity"),
                    "multi_period_metrics": item.get("multi_period_metrics"),
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _compact_base(payload: dict[str, Any], limit: int = 12) -> str:
        rows = []
        for item in (payload.get("factor_definitions") or [])[:limit]:
            rows.append(
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "formula": item.get("formula"),
                    "direction": item.get("direction"),
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _evaluation_to_brief(evaluation: dict[str, Any], history: list[dict[str, Any]]) -> str:
        ranked = evaluation.get("ranked_factors") or []
        history_names = [item.get("name") for item in history[-12:]]
        lines = [
            "# Optimizer Loop Brief",
            "",
            f"- Evaluated factors: `{evaluation.get('factor_count', 0)}`",
            f"- Effective factors: `{evaluation.get('effective_count', 0)}`",
            f"- Excellent factors: `{evaluation.get('excellent_count', 0)}`",
            f"- Recent historical factor names: `{', '.join(name for name in history_names if name) or 'none'}`",
            "",
            "## Ranked Evidence",
            "",
        ]
        for item in ranked[:8]:
            metrics = item.get("metrics") or {}
            lines.append(
                "- {name} | score={score} | quality={quality} | direction={direction} | mean_ic={mean_ic} | icir={icir} | win_rate={win_rate} | long_short={long_short}".format(
                    name=item.get("name", "-"),
                    score=item.get("score", "-"),
                    quality=item.get("quality", "-"),
                    direction=item.get("direction", "-"),
                    mean_ic=metrics.get("mean_ic", "-"),
                    icir=metrics.get("icir", "-"),
                    win_rate=metrics.get("ic_win_rate", "-"),
                    long_short=metrics.get("long_short_return", "-"),
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        parsed = FactorOptimizerAgent._load_json_object(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("Optimizer returned JSON, but it is not an object schema")
        parsed.setdefault("diagnostics", [])
        parsed.setdefault("next_round_plan", [])
        parsed.setdefault("candidate_blueprints", [])
        parsed.setdefault("guardrails", [])
        parsed.setdefault("refinement_blocked", [])
        parsed.setdefault("family_exploration_blocked", [])
        parsed.setdefault("reject_patterns", [])
        parsed.setdefault("promote_patterns", [])
        return parsed

    @staticmethod
    def _load_json_object(content: str) -> Any:
        candidates = [content.strip()]
        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks)
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(content[start : end + 1].strip())
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
        raise RuntimeError("Optimizer returned non-parseable JSON content: " + "; ".join(errors[-3:]))

    @staticmethod
    def _bullet_lines(items: Any) -> list[str]:
        if not items:
            return ["- 无。"]
        if isinstance(items, str):
            return [f"- {items}"]
        return [f"- {item}" for item in items]

    @staticmethod
    def _blocked_lines(items: Any) -> list[str]:
        if not items:
            return ["- 无。"]
        if isinstance(items, str):
            return [f"- {items}"]
        lines = []
        for item in items:
            if not isinstance(item, dict):
                lines.append(f"- {item}")
                continue
            actions = ", ".join(item.get("blocked_actions") or []) or "-"
            lines.append(
                "- {name} | 原因: {reason} | 禁止动作: {actions}".format(
                    name=item.get("name", "-"),
                    reason=item.get("reason", "-"),
                    actions=actions,
                )
            )
        return lines

    @staticmethod
    def _family_blocked_lines(items: Any) -> list[str]:
        if not items:
            return ["- 无。"]
        if isinstance(items, str):
            return [f"- {items}"]
        lines = []
        for item in items:
            if not isinstance(item, dict):
                lines.append(f"- {item}")
                continue
            actions = ", ".join(item.get("blocked_actions") or []) or "-"
            lines.append(
                "- {family} | concentration={concentration} | threshold={threshold} | 原因: {reason} | 禁止动作: {actions}".format(
                    family=item.get("family_tag", "-"),
                    concentration=item.get("concentration", "-"),
                    threshold=item.get("threshold", "-"),
                    reason=item.get("reason", "-"),
                    actions=actions,
                )
            )
        return lines


def run_optimizer_validation(
    checkpoint_dir: str | Path = "checkpoints",
    target_count: int = 3,
) -> dict[str, Any]:
    from .pipeline import SingleRunFactorMiningPipeline

    return SingleRunFactorMiningPipeline(checkpoint_dir=checkpoint_dir).validate_optimizer(
        target_count=target_count
    )
