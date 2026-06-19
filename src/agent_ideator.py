from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_tools import FactorToolbox
from .factor_family import validate_family_tag
from .llm_client import OpenAICompatibleLLMClient, ToolSpec
from .skill_loader import SkillLoader


class FactorIdeatorAgent:
    name = "FactorIdeator"

    def __init__(self, toolbox: FactorToolbox, llm_client: OpenAICompatibleLLMClient | None = None) -> None:
        self.toolbox = toolbox
        self.llm_client = llm_client or OpenAICompatibleLLMClient()
        self.skill_loader = SkillLoader(Path("prompts") / "skills")

    def discover(
        self,
        iteration: int = 1,
        factors_per_round: int = 3,
        history: list[dict[str, Any]] | None = None,
        feedback: dict[str, Any] | None = None,
        optimizer_feedback: dict[str, Any] | None = None,
        research_memory: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        factors: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        tools = [
            ToolSpec(
                name="list_skills",
                description="List read-only project research skills available to the Ideator.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self.skill_loader.list_skills,
            ),
            ToolSpec(
                name="load_skill",
                description="Load one allowed SKILL.md by skill name. This cannot read arbitrary files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
                handler=self.skill_loader.load_skill,
            ),
            ToolSpec(
                name="query_base_factor_pool",
                description="查询基础因子池，了解已有基础因子的名称、类别、公式、方向和经济含义。",
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
                description="查询当前数据面板可用字段、字段类型、覆盖率和可用于因子表达式的字段。",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self.toolbox.describe_data_fields,
            ),
            ToolSpec(
                name="analyze_factor_orthogonality",
                description="计算候选因子与基础因子池及参考表达式之间的相关性，用于判断候选因子是否具备正交性。",
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
            ToolSpec(
                name="validate_and_compute_factor",
                description="验证候选因子表达式，并计算最新截面的统计信息。",
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
                description="使用近期 Spearman IC 相对未来收益快速评估因子表现。",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "horizon": {"type": "integer", "default": 5},
                        "recent_days": {"type": "integer", "default": 252},
                    },
                    "required": ["expression"],
                },
                handler=self.toolbox.evaluate_factor_quick,
            ),
        ]
        _, trace = self.llm_client.chat_with_tools_loop(
            system_prompt=(
                "You are running with a read-only Skill Loader. "
                "Before proposing factors, call list_skills and then load_skill for "
                "rdagent-hypothesis-generation and rdagent-hypothesis-to-factor-task. "
                "Skills are research-method guidance only; they cannot replace factor validation, "
                "orthogonality checks, or quick evaluation. "
                "你是 FactorIdeator，一个 ETF 量化因子挖掘智能体。"
                "你必须只通过工具调用工作。请提出具有经济含义的候选因子，"
                "挖掘前可调用 query_base_factor_pool 查询基础因子池，调用 describe_data_fields 查询可用数据字段。"
                "对每个候选因子先调用 analyze_factor_orthogonality 检查与基础因子池的相关性，"
                "再调用 validate_and_compute_factor 验证表达式，并对有效候选调用 evaluate_factor_quick。"
                "只能使用这些算子语法：add, sub, mul, div, abs, sign, log, neg, sqrt, "
                "max, min, power, ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_delta, ts_delay, ts_return, ts_corr。"
                "可用字段包括：open, high, low, close, vol, amount, pct_chg, flow, flow_ratio。"
                "允许引用基础因子，例如 $momentum_20d。"
                "每个候选因子必须给出一个主家族标签 family_tag：price_momentum, price_reversal, volatility_risk, "
                "volume_price_confirmation, liquidity_pressure, trend_strength, flow_attention, cross_family_composite, other。"
                "family_tag 是研究语义判断，不是表达式关键词分类；必须同时给出 family_rationale 解释该标签的研究机制依据。"
            ),
            user_prompt=self._build_prompt(
                iteration,
                factors_per_round,
                history or [],
                feedback,
                optimizer_feedback,
                research_memory,
            ),
            tools=tools,
            max_iterations=max(14, factors_per_round * 6 + 6),
        )

        evaluations: dict[str, dict[str, Any]] = {}
        validated: list[dict[str, Any]] = []
        for item in trace:
            now = datetime.now().isoformat(timespec="seconds")
            calls.append(
                {
                    "time": now,
                    "agent": self.name,
                    "called": self._tool_call_name(item["tool"]),
                    "input": item["arguments"],
                    "result": self._recorded_tool_result(item["tool"], item["result"]),
                }
            )
            if item["tool"] == "validate_and_compute_factor" and item["result"].get("valid"):
                validated.append({**item["result"], "research_identity": self._research_identity(item["arguments"])})
            if item["tool"] == "evaluate_factor_quick":
                evaluations[item["arguments"]["expression"]] = item["result"]

        for item in validated:
            quick = evaluations.get(item["expression"])
            if quick is None:
                quick = self.toolbox.evaluate_factor_quick(item["expression"])
                calls.append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "agent": self.name,
                        "called": "FactorToolbox.evaluate_factor_quick",
                        "input": {"expression": item["expression"], "horizon": 5, "recent_days": 252},
                        "result": quick,
                    }
                )
            factors.append(
                {
                    "name": item["name"],
                    "expression": item["expression"],
                    "category": item["category"],
                    "family_tag": item["research_identity"]["family_tag"],
                    "family_rationale": item["research_identity"]["family_rationale"],
                    "logic": item["logic"],
                    "research_identity": item["research_identity"],
                    "tool_validation": item["stats"],
                    "quick_evaluation": quick,
                }
            )

        if not factors:
            raise RuntimeError("LLM did not produce any valid tool-validated factors")
        factors = sorted(
            factors,
            key=lambda item: (item.get("quick_evaluation") or {}).get("score") or 0,
            reverse=True,
        )[:factors_per_round]
        return factors, calls

    @staticmethod
    def _build_prompt(
        iteration: int,
        factors_per_round: int,
        history: list[dict[str, Any]],
        feedback: dict[str, Any] | None,
        optimizer_feedback: dict[str, Any] | None,
        research_memory: dict[str, Any] | None,
    ) -> str:
        memory_block = FactorIdeatorAgent._format_research_memory(research_memory)
        if iteration == 1:
            return (
                f"第 1 轮挖掘请生成恰好 {factors_per_round} 个新的 ETF 量价因子。"
                f"当前研究记忆：{memory_block}。"
                "候选因子至少覆盖 momentum 和 volume_price，并至少包含一个引用基础因子池的表达式。"
                "候选因子应分散到不同 family_tag，避免同一家族过度集中。"
                "不要只返回普通文本；必须使用工具调用验证并评估候选因子。"
            )

        prior_names = ", ".join(item.get("name", "") for item in history[-10:])
        optimizer_block = FactorIdeatorAgent._format_optimizer_feedback(optimizer_feedback)
        fallback_top_feedback = ""
        if feedback:
            ranked = feedback.get("ranked_factors", [])[:3]
            fallback_top_feedback = "; ".join(
                f"{item.get('name')} score={item.get('score')} quality={item.get('quality')} direction={item.get('direction')}"
                for item in ranked
            )
        return (
            f"第 {iteration} 轮挖掘请生成恰好 {factors_per_round} 个改进后的 ETF 因子。"
            f"需要避免重复的历史因子名称：{prior_names or '无'}。"
            f"上一轮优秀因子的备用反馈：{fallback_top_feedback or '无'}。"
            f"Optimizer 控制反馈：{optimizer_block}。"
            f"当前研究记忆：{memory_block}。"
            "请把 Optimizer 反馈视为下一轮研究简报。"
            "使用研究记忆延续活跃假设、强化有希望的方向、避开已拒绝方向；只有存在明确改动时才重新考虑观察名单因子。"
            "严格遵守 refinement_blocked 硬约束：被阻断因子不得作为 parent_factors，不得围绕其做 mutation、crossover 或 refinement。"
            "严格遵守 family_exploration_blocked 硬约束：被阻断家族不得生成新因子，不得作为父因子来源，不得围绕其做 new_hypothesis、mutation、crossover 或 refinement。"
            "把 Optimizer 给出的候选蓝图翻译为可执行表达式，且只能使用允许的算子语法。"
            "对每个最终候选因子，调用 validate_and_compute_factor 时必须传入研究身份字段：family_tag, family_rationale, strategy, parent_factors, hypothesis, expected_direction, risk_note, optimizer_instruction。"
            "只有当 Optimizer 诊断支持时，才使用 mutation、crossover、refinement 或 inversion。"
            "仍然只能通过工具调用验证和评估候选因子。"
            "把本轮要求的因子数量作为研究目标。"
        )

    @staticmethod
    def _tool_call_name(tool_name: str) -> str:
        if tool_name in {"list_skills", "load_skill"}:
            return f"SkillLoader.{tool_name}"
        return f"FactorToolbox.{tool_name}"

    @staticmethod
    def _recorded_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "load_skill":
            return result
        return {
            "valid": result.get("valid"),
            "name": result.get("name"),
            "description": result.get("description"),
            "source_path": result.get("source_path"),
            "content_length": len(result.get("content") or ""),
        }

    @staticmethod
    def _skill_usage_instruction() -> str:
        return (
            "Skill workflow: first call list_skills; then load_skill for "
            "rdagent-hypothesis-generation and rdagent-hypothesis-to-factor-task. "
            "Use rdagent-hypothesis-generation to state a testable research hypothesis before factor design. "
            "Use rdagent-hypothesis-to-factor-task to express candidates as structured factor tasks and to screen "
            "viability, relevance, duplication, and implementability before validate_and_compute_factor. "
            "Do not output plain text only; final accepted factors must still call analyze_factor_orthogonality, "
            "validate_and_compute_factor, and evaluate_factor_quick. "
        )

    @staticmethod
    def _format_research_memory(research_memory: dict[str, Any] | None) -> str:
        if not research_memory:
            return "无"
        return (
            "活跃假设={active}; 推广模式={promoted}; "
            "拒绝模式={rejected}; 阻断改进因子={blocked}; "
            "因子家族状态={families}; 阻断探索家族={family_blocked}; 观察名单因子={watchlist}"
        ).format(
            active=research_memory.get("active_hypotheses") or [],
            promoted=research_memory.get("promoted_patterns") or [],
            rejected=research_memory.get("rejected_patterns") or [],
            blocked=research_memory.get("refinement_blocked") or [],
            families=research_memory.get("factor_families") or {},
            family_blocked=research_memory.get("family_exploration_blocked") or [],
            watchlist=research_memory.get("watchlist_factors") or [],
        )

    @staticmethod
    def _format_optimizer_feedback(optimizer_feedback: dict[str, Any] | None) -> str:
        if not optimizer_feedback:
            return "无"
        diagnostics = "; ".join(optimizer_feedback.get("diagnostics") or [])
        plan = "; ".join(optimizer_feedback.get("next_round_plan") or [])
        blueprints = []
        for item in optimizer_feedback.get("candidate_blueprints") or []:
            blueprints.append(
                "{name} strategy={strategy} parents={parents} hypothesis={hypothesis} intent={intent} direction={direction} risk={risk}".format(
                    name=item.get("name", "-"),
                    strategy=item.get("strategy", "-"),
                    parents=",".join(item.get("parent_factors") or []),
                    hypothesis=item.get("hypothesis", "-"),
                    intent=item.get("expression_intent", "-"),
                    direction=item.get("expected_direction", "-"),
                    risk=item.get("risk_note", "-"),
                )
            )
        guardrails = "; ".join(optimizer_feedback.get("guardrails") or [])
        blocked = "; ".join(
            "{name} reason={reason} blocked_actions={actions}".format(
                name=item.get("name", "-"),
                reason=item.get("reason", "-"),
                actions=",".join(item.get("blocked_actions") or []),
            )
            for item in optimizer_feedback.get("refinement_blocked") or []
            if isinstance(item, dict)
        )
        family_blocked = "; ".join(
            "{family} concentration={concentration} threshold={threshold} reason={reason} blocked_actions={actions}".format(
                family=item.get("family_tag", "-"),
                concentration=item.get("concentration", "-"),
                threshold=item.get("threshold", "-"),
                reason=item.get("reason", "-"),
                actions=",".join(item.get("blocked_actions") or []),
            )
            for item in optimizer_feedback.get("family_exploration_blocked") or []
            if isinstance(item, dict)
        )
        return (
            f"诊断=[{diagnostics or '无'}]; "
            f"下一轮计划=[{plan or '无'}]; "
            f"候选蓝图=[{' | '.join(blueprints) or '无'}]; "
            f"约束=[{guardrails or '无'}]; "
            f"阻断改进=[{blocked or '无'}]; "
            f"阻断家族探索=[{family_blocked or '无'}]"
        )

    @staticmethod
    def _research_identity(arguments: dict[str, Any]) -> dict[str, Any]:
        family_tag = validate_family_tag(arguments.get("family_tag"))
        return {
            "strategy": arguments.get("strategy") or "new_hypothesis",
            "family_tag": family_tag,
            "family_rationale": arguments.get("family_rationale") or "",
            "parent_factors": arguments.get("parent_factors") or [],
            "hypothesis": arguments.get("hypothesis") or arguments.get("logic") or "",
            "expected_direction": arguments.get("expected_direction") or "neutral",
            "risk_note": arguments.get("risk_note") or "",
            "optimizer_instruction": arguments.get("optimizer_instruction") or "",
        }
