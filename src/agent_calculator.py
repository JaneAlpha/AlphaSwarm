from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .agent_tools import FactorToolbox
from .llm_client import OpenAICompatibleLLMClient


class FactorCalculatorAgent:
    name = "FactorCalculator"

    def __init__(
        self,
        toolbox: FactorToolbox,
        llm_client: OpenAICompatibleLLMClient | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.llm_client = llm_client or OpenAICompatibleLLMClient()

    def calculate(self, factors: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for factor in factors:
            result = self.toolbox.validate_and_compute_factor(factor["expression"])
            calls.append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "agent": self.name,
                    "called": "FactorToolbox.validate_and_compute_factor",
                    "input": {"name": factor["name"], "expression": factor["expression"]},
                    "result": {"valid": result.valid, "stats": result.stats, "error": result.error},
                }
            )
            repair: dict[str, Any] | None = None
            if not result.valid:
                repair = self._repair_and_compute(factor, result)
                calls.extend(repair["calls"])
                if repair["result"].valid:
                    result = repair["result"]
            results.append(
                {
                    "name": factor["name"],
                    "expression": repair["expression"] if repair and repair["result"].valid else factor["expression"],
                    "original_expression": factor["expression"] if repair and repair["result"].valid else None,
                    "category": factor["category"],
                    "family_tag": factor.get("family_tag"),
                    "family_rationale": factor.get("family_rationale"),
                    "research_identity": factor.get("research_identity"),
                    "success": result.valid,
                    "latest_date": result.latest_date,
                    "stats": result.stats,
                    "performance": self.toolbox.evaluate_factor_quick(result.expression) if result.valid else None,
                    "values": result.values if result.valid else [],
                    "error": result.error,
                    "repair": repair["summary"] if repair else None,
                }
            )

        return {
            "schema_version": "1.0",
            "agent": self.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "success_count": sum(1 for item in results if item["success"]),
            "failure_count": sum(1 for item in results if not item["success"]),
            "results": results,
        }, calls

    def build_report(self, calculation: dict[str, Any]) -> str:
        lines = [
            "# 因子计算报告",
            "",
            f"- Agent: `{self.name}`",
            f"- 生成时间: `{calculation.get('created_at', '')}`",
            f"- 成功数量: `{calculation['success_count']}`",
            f"- 失败数量: `{calculation['failure_count']}`",
            "",
            "## 计算摘要",
            "",
            "| Factor | Status | Latest Date | Values | Coverage | Mean | Std | Error |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in calculation["results"]:
            stats = item.get("stats") or {}
            status = "success" if item["success"] else "failed"
            lines.append(
                "| {name} | {status} | {date} | {values} | {coverage:.2%} | {mean} | {std} | {error} |".format(
                    name=item["name"],
                    status=status,
                    date=item.get("latest_date") or "-",
                    values=len(item.get("values") or []),
                    coverage=stats.get("latest_coverage") or 0,
                    mean=stats.get("mean"),
                    std=stats.get("std"),
                    error=item.get("error") or "",
                )
            )
        repaired = [item for item in calculation["results"] if item.get("repair")]
        if repaired:
            lines.extend(["", "## 已修复表达式", ""])
            for item in repaired:
                repair = item.get("repair") or {}
                lines.extend(
                    [
                        f"- {item['name']}",
                        f"  - 原始表达式: `{item.get('original_expression') or '-'}`",
                        f"  - 修复后表达式: `{item.get('expression') or '-'}`",
                        f"  - 原因: {repair.get('reason') or '-'}",
                    ]
                )
        lines.extend(
            [
                "",
                "## 说明",
                "",
                "- 每个因子表达式都会由算子引擎解析。",
                "- 基础因子引用会在计算前展开。",
                "- 最新截面值会保存在 `factor_cal_results.json`。",
            ]
        )
        return "\n".join(lines) + "\n"

    def build_visualizations(self, calculation: dict[str, Any]) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        visualizations: list[dict[str, Any]] = []
        used_slugs: set[str] = set()
        for index, item in enumerate(calculation.get("results") or [], start=1):
            performance = item.get("performance") or {}
            if not item.get("success") or not performance:
                continue
            slug = self._factor_slug(item.get("name") or f"factor_{index}", index, used_slugs)
            filename = f"cal_visualizations/{slug}.json"
            payload = self._build_factor_visualization(item, performance)
            files[filename] = payload
            visualizations.append(
                {
                    "name": item.get("name"),
                    "category": item.get("category"),
                    "family_tag": item.get("family_tag"),
                    "expression": item.get("expression"),
                    "score": performance.get("score"),
                    "quality": performance.get("quality"),
                    "direction": performance.get("direction"),
                    "file": filename,
                }
            )
        return {
            "manifest": {
                "schema_version": "1.0",
                "agent": self.name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": "factor_cal_results.json",
                "factor_count": len(visualizations),
                "visualizations": visualizations,
            },
            "files": files,
        }

    @staticmethod
    def _build_factor_visualization(item: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
        multi_period = [
            FactorCalculatorAgent._period_visual(metric)
            for metric in performance.get("multi_period_metrics") or []
        ]
        stratified = FactorCalculatorAgent._stratified_visual(performance.get("stratified_backtest") or {})
        return {
            "schema_version": "1.0",
            "agent": FactorCalculatorAgent.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "factor": {
                "name": item.get("name"),
                "category": item.get("category"),
                "family_tag": item.get("family_tag"),
                "expression": item.get("expression"),
                "score": performance.get("score"),
                "quality": performance.get("quality"),
                "direction": performance.get("direction"),
            },
            "evaluation_parameters": FactorCalculatorAgent._evaluation_parameter_visual(performance),
            "multi_period_metrics": multi_period,
            "stratified_backtest": stratified,
        }

    @staticmethod
    def _evaluation_parameter_visual(performance: dict[str, Any]) -> dict[str, Any]:
        multi_period = performance.get("multi_period_metrics") or []
        horizons = [item.get("horizon") for item in multi_period if item.get("horizon") is not None]
        return {
            "primary_horizon": performance.get("horizon"),
            "recent_days": performance.get("recent_days"),
            "quantile_groups": performance.get("quantile_groups"),
            "multi_horizons": horizons,
            "ic_method": "daily cross-sectional Spearman correlation between factor value and future return",
            "future_return": "close.shift(-horizon) / close - 1",
            "icir_method": "mean_ic / std(ic)",
            "stratified_method": "rank factor values by date, split into quantile groups, then compare grouped future returns",
            "long_short_method": "highest quantile mean return - lowest quantile mean return",
            "directional_monotonicity_method": "Spearman correlation between quantile rank and group return; negative signals reverse the sign",
            "thresholds": performance.get("thresholds") or {},
        }

    @staticmethod
    def _period_visual(metric: dict[str, Any]) -> dict[str, Any]:
        stratified = metric.get("stratified_backtest") or {}
        return {
            "horizon": metric.get("horizon"),
            "mean_ic": metric.get("mean_ic"),
            "icir": metric.get("icir"),
            "ic_win_rate": metric.get("ic_win_rate"),
            "long_short_return": metric.get("long_short_return"),
            "direction": metric.get("direction"),
            "ic_observations": metric.get("ic_observations"),
            "long_short_observations": metric.get("long_short_observations"),
            "directional_monotonicity": stratified.get("directional_monotonicity"),
        }

    @staticmethod
    def _stratified_visual(stratified: dict[str, Any]) -> dict[str, Any]:
        groups = [
            {
                "bucket": item.get("bucket"),
                "label": f"Q{item.get('bucket')}",
                "mean_return": item.get("mean_return"),
            }
            for item in stratified.get("group_mean_returns") or []
        ]
        return {
            "group_count": stratified.get("group_count"),
            "groups": groups,
            "long_short_return": stratified.get("long_short_return"),
            "monotonicity_corr": stratified.get("monotonicity_corr"),
            "directional_monotonicity": stratified.get("directional_monotonicity"),
        }

    @staticmethod
    def _factor_slug(name: str, index: int, used_slugs: set[str]) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
        slug = "_".join(part for part in slug.split("_") if part)
        if not slug:
            slug = f"factor_{index}"
        slug = slug[:80]
        candidate = slug
        suffix = 2
        while candidate in used_slugs:
            candidate = f"{slug}_{suffix}"
            suffix += 1
        used_slugs.add(candidate)
        return candidate

    def _repair_and_compute(self, factor: dict[str, Any], failed_result: Any) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        error = failed_result.error or ""
        repaired_expression = self._repair_expression(factor, error, calls)
        if not repaired_expression:
            return {
                "expression": factor["expression"],
                "result": failed_result,
                "calls": calls,
                "summary": {
                    "attempted": True,
                    "success": False,
                    "reason": "LLM 未返回可用的修复表达式。",
                },
            }

        result = self.toolbox.validate_and_compute_factor(repaired_expression)
        calls.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "agent": self.name,
                "called": "FactorToolbox.validate_and_compute_factor",
                "input": {
                    "name": factor["name"],
                    "expression": repaired_expression,
                    "repair_of": factor["expression"],
                },
                "result": {"valid": result.valid, "stats": result.stats, "error": result.error},
            }
        )
        return {
            "expression": repaired_expression,
            "result": result,
            "calls": calls,
            "summary": {
                "attempted": True,
                "success": result.valid,
                "reason": "LLM 在计算失败后修复了表达式。",
                "error_before_repair": error,
                "error_after_repair": result.error,
            },
        }

    def _repair_expression(
        self,
        factor: dict[str, Any],
        error: str,
        calls: list[dict[str, Any]],
    ) -> str | None:
        system_prompt = (
            "你是 FactorCalculator，一个 ETF 因子表达式修复智能体。"
            "你只能修复语法、未支持算子、无效字段、窗口参数或基础因子引用问题。"
            "只能返回严格 JSON，不要编造业务结论。"
        )
        user_prompt = (
            "请修复这个计算失败的因子表达式，使其可以被算子引擎计算。\n\n"
            "允许字段：open, high, low, close, vol, amount, pct_chg, flow, flow_ratio。\n"
            "允许算子：add, sub, mul, div, abs, sign, log, neg, sqrt, max, min, power, "
            "ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_delta, ts_delay, ts_return, ts_corr.\n"
            "允许引用基础因子，例如 $momentum_20d。\n\n"
            "返回 JSON schema: {\"expression\": \"...\", \"reason\": \"...\"}\n\n"
            f"因子名称：{factor.get('name')}\n"
            f"类别：{factor.get('category')}\n"
            f"逻辑：{factor.get('logic')}\n"
            f"失败表达式：{factor.get('expression')}\n"
            f"错误：{error}\n"
        )
        try:
            response = self.llm_client._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[],
            )
            content = response["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            calls.append(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "agent": self.name,
                    "called": "FactorCalculator.repair_expression",
                    "input": {"name": factor.get("name"), "expression": factor.get("expression"), "error": error},
                    "result": {"error": str(exc)},
                }
            )
            return None
        calls.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "agent": self.name,
                "called": "FactorCalculator.repair_expression",
                "input": {"name": factor.get("name"), "expression": factor.get("expression"), "error": error},
                "result": {"raw_response": content},
            }
        )
        parsed = self._parse_repair_response(content)
        return parsed.get("expression")

    @staticmethod
    def _parse_repair_response(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
