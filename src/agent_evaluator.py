from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .llm_client import OpenAICompatibleLLMClient


class FactorEvaluatorAgent:
    name = "FactorEvaluator"

    def __init__(self, llm_client: OpenAICompatibleLLMClient | None = None) -> None:
        self.llm_client = llm_client

    def evaluate(self, calculation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        evaluated: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for item in calculation["results"]:
            if not item.get("success"):
                continue
            perf = item.get("performance") or {}
            evaluated.append(
                {
                    "name": item["name"],
                    "category": item["category"],
                    "family_tag": item.get("family_tag"),
                    "family_rationale": item.get("family_rationale"),
                    "expression": item["expression"],
                    "score": perf.get("score"),
                    "quality": perf.get("quality"),
                    "direction": perf.get("direction"),
                    "is_effective": perf.get("is_effective"),
                    "is_excellent": perf.get("is_excellent"),
                    "metrics": {
                        "mean_ic": perf.get("mean_ic"),
                        "icir": perf.get("icir"),
                        "ic_win_rate": perf.get("ic_win_rate"),
                        "long_short_return": perf.get("long_short_return"),
                        "ic_observations": perf.get("ic_observations"),
                        "long_short_observations": perf.get("long_short_observations"),
                    },
                    "multi_period_metrics": perf.get("multi_period_metrics") or [],
                    "stratified_backtest": perf.get("stratified_backtest") or {},
                    "score_breakdown": perf.get("score_breakdown"),
                    "thresholds": perf.get("thresholds"),
                }
            )
            calls.append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "agent": self.name,
                    "called": "FactorEvaluator.evaluate_factor_performance",
                    "input": {"name": item["name"], "expression": item["expression"]},
                    "result": {
                        "score": perf.get("score"),
                        "quality": perf.get("quality"),
                        "direction": perf.get("direction"),
                        "is_effective": perf.get("is_effective"),
                    },
                }
            )

        ranked = sorted(evaluated, key=lambda x: x.get("score") or 0, reverse=True)
        diagnostics = self._build_diagnostics(ranked)
        optimizer_feedback_hook = self._build_optimizer_feedback_hook(ranked, diagnostics)
        llm_analysis, llm_call = self._analyze_with_llm(ranked, diagnostics, optimizer_feedback_hook)
        calls.append(llm_call)
        return {
            "schema_version": "1.0",
            "agent": self.name,
            "analysis_mode": "llm_required",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_standard": {
                "mean_ic_effective": 0.02,
                "mean_ic_excellent": 0.05,
                "icir_effective": 0.1,
                "icir_excellent": 0.5,
                "directional_ic_win_rate_effective": 0.55,
                "directional_ic_win_rate_excellent": 0.65,
                "long_short_effective": 0.005,
                "long_short_excellent": 0.01,
                "score_formula": "IC 40% + ICIR 30% + 分层单调性 20% + IC 胜率 10%",
                "monotonicity_score": "long_short_score 10% + monotonic_rank_score 10%",
            },
            "factor_count": len(ranked),
            "effective_count": sum(1 for item in ranked if item.get("is_effective")),
            "excellent_count": sum(1 for item in ranked if item.get("is_excellent")),
            "diagnostics": diagnostics,
            "llm_analysis": llm_analysis,
            "optimizer_feedback_hook": optimizer_feedback_hook,
            "ranked_factors": ranked,
        }, calls

    def build_report(self, evaluation: dict[str, Any]) -> str:
        lines = [
            "# 因子评估报告",
            "",
            f"- Agent: `{self.name}`",
            f"- 生成时间: `{evaluation.get('created_at', '')}`",
            f"- 已评估因子数: `{evaluation['factor_count']}`",
            f"- 有效因子数: `{evaluation['effective_count']}`",
            f"- 优秀因子数: `{evaluation['excellent_count']}`",
            "",
            "## 评估标准",
            "",
            "- 平均 IC：有效标准 `|IC| > 0.02`，优秀标准 `|IC| > 0.05`。",
            "- ICIR：有效标准 `|ICIR| > 0.1`，优秀标准 `|ICIR| > 0.5`。",
            "- 方向性 IC 胜率：有效标准 `> 55%`，优秀标准 `> 65%`；正向因子使用 `IC > 0`，负向因子使用 `IC < 0`。",
            "- 多空收益：有效标准 `> 0.5%`，优秀标准 `> 1%`。",
            "- 综合评分 = IC 分 40% + ICIR 分 30% + 分层单调性分 20% + 胜率分 10%。",
            "- 分层单调性分 = 多空收益强度 10% + 分组收益方向单调性 10%。",
            "",
            "## 因子排序",
            "",
            "| Rank | Factor | Score | Quality | Direction | Mean IC | ICIR | IC Win Rate | Long-Short | Monotonicity |",
            "|---:|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for idx, item in enumerate(evaluation["ranked_factors"], start=1):
            metrics = item["metrics"]
            stratified = item.get("stratified_backtest") or {}
            lines.append(
                "| {rank} | {name} | {score:.4f} | {quality} | {direction} | {mean_ic} | {icir} | {win_rate:.2%} | {long_short:.2%} | {monotonicity} |".format(
                    rank=idx,
                    name=item["name"],
                    score=item.get("score") or 0,
                    quality=item.get("quality") or "-",
                    direction=item.get("direction") or "-",
                    mean_ic=metrics.get("mean_ic"),
                    icir=metrics.get("icir"),
                    win_rate=metrics.get("ic_win_rate") or 0,
                    long_short=metrics.get("long_short_return") or 0,
                    monotonicity=stratified.get("directional_monotonicity"),
                )
            )
        lines.extend(["", "## 多周期 IC/ICIR", ""])
        for item in evaluation["ranked_factors"][:10]:
            lines.append(f"### {item['name']}")
            lines.append("")
            lines.append("| Horizon | Mean IC | ICIR | IC Win Rate | Long-Short | Monotonicity |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for metric in item.get("multi_period_metrics") or []:
                stratified = metric.get("stratified_backtest") or {}
                lines.append(
                    "| {horizon} | {mean_ic} | {icir} | {win_rate:.2%} | {long_short:.2%} | {monotonicity} |".format(
                        horizon=metric.get("horizon"),
                        mean_ic=metric.get("mean_ic"),
                        icir=metric.get("icir"),
                        win_rate=metric.get("ic_win_rate") or 0,
                        long_short=metric.get("long_short_return") or 0,
                        monotonicity=stratified.get("directional_monotonicity"),
                    )
                )
            lines.append("")
        lines.extend(["## 分层回测摘要", ""])
        for item in evaluation["ranked_factors"][:10]:
            stratified = item.get("stratified_backtest") or {}
            groups = stratified.get("group_mean_returns") or []
            group_text = ", ".join(
                "Q{bucket}={value:.2%}".format(
                    bucket=group.get("bucket"),
                    value=group.get("mean_return") or 0,
                )
                for group in groups
            )
            lines.append(
                "- {name}: long_short={long_short:.2%}, monotonicity={monotonicity}, groups={groups}".format(
                    name=item["name"],
                    long_short=stratified.get("long_short_return") or 0,
                    monotonicity=stratified.get("directional_monotonicity"),
                    groups=group_text or "-",
                )
            )
        lines.extend(
            [
                "",
                "## 研究诊断",
                "",
                "### 优胜模式",
                "",
            ]
        )
        diagnostics = evaluation.get("diagnostics") or {}
        lines.extend(self._bullet_lines(diagnostics.get("winner_patterns")))
        lines.extend(["", "### 失败模式", ""])
        lines.extend(self._bullet_lines(diagnostics.get("failure_patterns")))
        lines.extend(["", "### 风险提示", ""])
        lines.extend(self._bullet_lines(diagnostics.get("risk_flags")))
        lines.extend(["", "### 建议动作", ""])
        lines.extend(self._bullet_lines(diagnostics.get("recommended_actions")))
        llm_analysis = evaluation.get("llm_analysis") or {}
        lines.extend(["", "## LLM 评估分析", ""])
        lines.extend(["### 总体判断", ""])
        lines.extend(self._bullet_lines(llm_analysis.get("summary")))
        lines.extend(["", "### 因子级诊断", ""])
        for item in llm_analysis.get("factor_diagnoses") or []:
            lines.append(
                "- {name}: {diagnosis}".format(
                    name=item.get("name", "-"),
                    diagnosis=item.get("diagnosis") or item.get("reason") or "-",
                )
            )
        lines.extend(["", "### 给 Optimizer 的反馈钩子", ""])
        hook = evaluation.get("optimizer_feedback_hook") or {}
        for key, title in [
            ("refinement_candidates", "精炼候选"),
            ("inversion_candidates", "取反候选"),
            ("mutation_candidates", "变异候选"),
            ("block_candidates", "阻断候选"),
            ("family_feedback", "家族反馈"),
        ]:
            lines.append(f"- {title}: `{len(hook.get(key) or [])}`")
        lines.extend(
            [
                "",
                "## 方向规则",
                "",
                "- 正向因子：`IC > 0`，因子值越大表示未来收益越高。",
                "- 负向因子：`IC < 0`，因子值越小表示未来收益越高；实际使用时需要明确取反规则。",
            ]
        )
        return "\n".join(lines) + "\n"

    def _analyze_with_llm(
        self,
        ranked: list[dict[str, Any]],
        diagnostics: dict[str, list[str]],
        optimizer_feedback_hook: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_payload = {
            "ranked_factors": self._compact_for_llm(ranked),
            "rule_diagnostics": diagnostics,
            "optimizer_feedback_hook": optimizer_feedback_hook,
        }
        system_prompt = (
            "你是 FactorEvaluator，一个量化因子评估智能体。"
            "你的职责是基于评估指标解释因子是否有效、弱在哪里、是否适合取反/精炼/变异/阻断。"
            "只能根据输入证据分析，不要发明新指标，不要声称 weak 因子有效。"
            "返回严格 JSON。"
        )
        user_prompt = (
            "请生成因子评估分析，schema 如下：\n"
            "{\n"
            '  "summary": ["..."],\n'
            '  "factor_diagnoses": [{"name": "...", "diagnosis": "...", "optimizer_relevance": "refine|mutate|invert|block|watch"}],\n'
            '  "cross_factor_findings": ["..."],\n'
            '  "optimizer_priorities": ["..."],\n'
            '  "risk_warnings": ["..."]\n'
            "}\n\n"
            "评估证据 JSON：\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )
        call = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "agent": self.name,
            "called": "FactorEvaluator.llm_analysis",
            "input": {"factor_count": len(ranked)},
            "result": {},
        }
        try:
            client = self.llm_client or OpenAICompatibleLLMClient()
            response = client._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[],
            )
            content = response["choices"][0]["message"].get("content") or ""
            parsed = self._parse_json(content)
            if not parsed:
                raise ValueError("LLM did not return parseable JSON")
            call["result"] = {"status": "success", "raw_response": content}
            return self._normalize_llm_analysis(parsed), call
        except Exception as exc:
            call["result"] = {"status": "failed", "error": str(exc)}
            raise RuntimeError(f"FactorEvaluator LLM analysis failed: {exc}") from exc

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        candidates = [content.strip()]
        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks)
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(content[start : end + 1].strip())
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _normalize_llm_analysis(parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": parsed.get("summary") if isinstance(parsed.get("summary"), list) else [],
            "factor_diagnoses": parsed.get("factor_diagnoses") if isinstance(parsed.get("factor_diagnoses"), list) else [],
            "cross_factor_findings": parsed.get("cross_factor_findings") if isinstance(parsed.get("cross_factor_findings"), list) else [],
            "optimizer_priorities": parsed.get("optimizer_priorities") if isinstance(parsed.get("optimizer_priorities"), list) else [],
            "risk_warnings": parsed.get("risk_warnings") if isinstance(parsed.get("risk_warnings"), list) else [],
        }

    @staticmethod
    def _compact_for_llm(ranked: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        rows = []
        for item in ranked[:limit]:
            metrics = item.get("metrics") or {}
            stratified = item.get("stratified_backtest") or {}
            rows.append(
                {
                    "name": item.get("name"),
                    "family_tag": item.get("family_tag"),
                    "category": item.get("category"),
                    "score": item.get("score"),
                    "quality": item.get("quality"),
                    "direction": item.get("direction"),
                    "is_effective": item.get("is_effective"),
                    "mean_ic": metrics.get("mean_ic"),
                    "icir": metrics.get("icir"),
                    "ic_win_rate": metrics.get("ic_win_rate"),
                    "long_short_return": metrics.get("long_short_return"),
                    "directional_monotonicity": stratified.get("directional_monotonicity"),
                    "multi_period_metrics": item.get("multi_period_metrics"),
                }
            )
        return rows

    @staticmethod
    def _build_optimizer_feedback_hook(
        ranked: list[dict[str, Any]],
        diagnostics: dict[str, list[str]],
    ) -> dict[str, Any]:
        family_counts: dict[str, int] = {}
        for item in ranked:
            family = item.get("family_tag") or "unknown"
            family_counts[family] = family_counts.get(family, 0) + 1
        total = sum(family_counts.values()) or 1
        family_feedback = [
            {
                "family_tag": family,
                "count": count,
                "concentration": round(count / total, 6),
            }
            for family, count in sorted(family_counts.items(), key=lambda row: row[1], reverse=True)
        ]
        return {
            "schema_version": "1.0",
            "purpose": "structured_evaluator_feedback_for_optimizer",
            "refinement_candidates": [
                FactorEvaluatorAgent._feedback_item(item, "refinement_candidate")
                for item in ranked
                if FactorEvaluatorAgent._is_refinement_candidate(item)
            ][:6],
            "inversion_candidates": [
                FactorEvaluatorAgent._feedback_item(item, "inversion_candidate")
                for item in ranked
                if FactorEvaluatorAgent._is_inversion_candidate(item)
            ][:6],
            "mutation_candidates": [
                FactorEvaluatorAgent._feedback_item(item, "mutation_candidate")
                for item in ranked
                if FactorEvaluatorAgent._is_mutation_candidate(item)
            ][:6],
            "block_candidates": [
                FactorEvaluatorAgent._feedback_item(item, "block_candidate")
                for item in ranked
                if FactorEvaluatorAgent._is_block_candidate(item)
            ][:10],
            "family_feedback": family_feedback,
            "diagnostic_basis": diagnostics,
        }

    @staticmethod
    def _feedback_item(item: dict[str, Any], action_hint: str) -> dict[str, Any]:
        metrics = item.get("metrics") or {}
        stratified = item.get("stratified_backtest") or {}
        return {
            "name": item.get("name"),
            "family_tag": item.get("family_tag"),
            "category": item.get("category"),
            "action_hint": action_hint,
            "reason": FactorEvaluatorAgent._feedback_reason(item),
            "metric_evidence": {
                "score": item.get("score"),
                "quality": item.get("quality"),
                "direction": item.get("direction"),
                "mean_ic": metrics.get("mean_ic"),
                "icir": metrics.get("icir"),
                "ic_win_rate": metrics.get("ic_win_rate"),
                "long_short_return": metrics.get("long_short_return"),
                "directional_monotonicity": stratified.get("directional_monotonicity"),
            },
        }

    @staticmethod
    def _feedback_reason(item: dict[str, Any]) -> str:
        metrics = item.get("metrics") or {}
        stratified = item.get("stratified_backtest") or {}
        return (
            "score={score}, quality={quality}, direction={direction}, "
            "mean_ic={mean_ic}, icir={icir}, win_rate={win_rate}, "
            "long_short={long_short}, monotonicity={monotonicity}"
        ).format(
            score=item.get("score"),
            quality=item.get("quality"),
            direction=item.get("direction"),
            mean_ic=metrics.get("mean_ic"),
            icir=metrics.get("icir"),
            win_rate=metrics.get("ic_win_rate"),
            long_short=metrics.get("long_short_return"),
            monotonicity=stratified.get("directional_monotonicity"),
        )

    @staticmethod
    def _is_refinement_candidate(item: dict[str, Any]) -> bool:
        metrics = item.get("metrics") or {}
        return bool(
            not item.get("is_effective")
            and abs(metrics.get("mean_ic") or 0) >= 0.03
            and abs(metrics.get("icir") or 0) >= 0.1
            and (metrics.get("ic_win_rate") or 0) >= 0.55
            and abs(metrics.get("long_short_return") or 0) < 0.005
        )

    @staticmethod
    def _is_inversion_candidate(item: dict[str, Any]) -> bool:
        metrics = item.get("metrics") or {}
        return bool(
            item.get("direction") == "negative"
            and abs(metrics.get("mean_ic") or 0) >= 0.02
            and abs(metrics.get("icir") or 0) >= 0.1
            and (metrics.get("ic_win_rate") or 0) >= 0.55
        )

    @staticmethod
    def _is_mutation_candidate(item: dict[str, Any]) -> bool:
        metrics = item.get("metrics") or {}
        return bool(
            not item.get("is_effective")
            and 30 <= (item.get("score") or 0) < 65
            and abs(metrics.get("mean_ic") or 0) >= 0.02
        )

    @staticmethod
    def _is_block_candidate(item: dict[str, Any]) -> bool:
        metrics = item.get("metrics") or {}
        return bool(
            not item.get("is_effective")
            and (
                abs(metrics.get("mean_ic") or 0) < 0.02
                or abs(metrics.get("icir") or 0) < 0.1
                or abs(metrics.get("long_short_return") or 0) < 0.0025
            )
        )

    @staticmethod
    def _build_diagnostics(ranked: list[dict[str, Any]]) -> dict[str, list[str]]:
        effective = [item for item in ranked if item.get("is_effective")]
        weak = [item for item in ranked if not item.get("is_effective")]
        negative = [item for item in ranked if item.get("direction") == "negative"]
        low_win_rate = [
            item
            for item in ranked
            if (item.get("metrics") or {}).get("ic_win_rate") is not None
            and (item.get("metrics") or {}).get("ic_win_rate") < 0.5
        ]

        winner_patterns = []
        if effective:
            categories = sorted({item.get("category", "unknown") for item in effective})
            winner_patterns.append(
                "有效因子集中在以下类别：{categories}。".format(
                    categories=", ".join(categories)
                )
            )
            top = effective[0]
            metrics = top.get("metrics") or {}
            winner_patterns.append(
                "排名最高的有效因子 {name} 的 mean_ic={mean_ic}, icir={icir}, win_rate={win_rate}, long_short={long_short}。".format(
                    name=top.get("name"),
                    mean_ic=metrics.get("mean_ic"),
                    icir=metrics.get("icir"),
                    win_rate=metrics.get("ic_win_rate"),
                    long_short=metrics.get("long_short_return"),
                )
            )
        else:
            winner_patterns.append("本轮没有因子达到有效阈值。")

        failure_patterns = []
        if weak:
            weak_categories = sorted({item.get("category", "unknown") for item in weak})
            failure_patterns.append(
                "偏弱因子出现在以下类别：{categories}。".format(
                    categories=", ".join(weak_categories)
                )
            )
        if low_win_rate:
            failure_patterns.append(
                "{count} 个因子的 IC 胜率低于 50%，说明方向可能不稳定或存在较强市场状态依赖。".format(
                    count=len(low_win_rate)
                )
            )

        risk_flags = []
        if negative:
            risk_flags.append(
                "{count} 个因子是负向信号，需要明确取反方式或使用规则。".format(
                    count=len(negative)
                )
            )
        if ranked and len(effective) == 0:
            risk_flags.append("所有候选因子均未达到有效阈值，避免将新因子提升为基础因子。")
        if ranked and len(effective) < max(1, len(ranked) // 4):
            risk_flags.append("有效因子覆盖面较窄，需要避免过度拟合单一表达式族。")

        recommended_actions = []
        if effective:
            recommended_actions.append("对有效因子进行受控变异和精炼。")
        if negative:
            recommended_actions.append("只对幅度充分且稳定的负 IC 因子测试取反方案。")
        recommended_actions.append("下一轮构想应使用候选蓝图保留研究假设、父因子、方向和风险提示。")

        return {
            "winner_patterns": winner_patterns,
            "failure_patterns": failure_patterns,
            "risk_flags": risk_flags,
            "recommended_actions": recommended_actions,
        }

    @staticmethod
    def _bullet_lines(items: Any) -> list[str]:
        if not items:
            return ["- 无。"]
        return [f"- {item}" for item in items]
