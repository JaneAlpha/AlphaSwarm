from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .dataloader import ETFDataLoader
from .expression_utils import BASE_FACTOR_EXPRESSION_ALIASES, expand_base_factor_references, parse_expression
from .factor_family import validate_family_tag
from .operators import OperatorEngine


@dataclass(frozen=True)
class ToolResult:
    valid: bool
    expression: str
    latest_date: str | None = None
    values: list[dict[str, Any]] | None = None
    stats: dict[str, Any] | None = None
    error: str | None = None


class FactorToolbox:
    def __init__(self, loader: ETFDataLoader | None = None) -> None:
        self.loader = loader or ETFDataLoader()
        self.panel = self.loader.load_daily_panel()
        self.engine = OperatorEngine(self.panel)

    def validate_and_compute_factor(self, expression: str) -> ToolResult:
        try:
            expanded_expression = expand_base_factor_references(expression)
            node = parse_expression(expanded_expression)
            values = self.engine.evaluate(node)
            if not isinstance(values, pd.Series):
                raise ValueError("Expression must produce a time series")
            if len(values) != len(self.panel):
                raise ValueError("Expression result length mismatch")

            frame = self.panel[["date", "symbol"]].copy()
            frame["value"] = values.replace([np.inf, -np.inf], np.nan)
            latest = frame.loc[frame.groupby("symbol")["date"].idxmax()].sort_values("symbol")
            valid_latest = latest["value"].dropna()
            stats = {
                "rows": int(len(frame)),
                "latest_symbols": int(len(latest)),
                "latest_non_null": int(valid_latest.shape[0]),
                "latest_coverage": round(float(latest["value"].notna().mean()), 6),
                "mean": self._clean_float(valid_latest.mean()),
                "std": self._clean_float(valid_latest.std()),
                "min": self._clean_float(valid_latest.min()),
                "max": self._clean_float(valid_latest.max()),
            }
            return ToolResult(
                valid=True,
                expression=expression,
                latest_date=str(latest["date"].max().date()),
                values=self._records(latest),
                stats=stats,
            )
        except Exception as exc:
            return ToolResult(valid=False, expression=expression, error=str(exc))

    def validate_and_compute_factor_dict(
        self,
        name: str = "",
        expression: str = "",
        category: str = "",
        logic: str = "",
        family_tag: str = "",
        family_rationale: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        try:
            resolved_family_tag = validate_family_tag(family_tag)
        except ValueError as exc:
            return {
                "name": name,
                "expression": expression,
                "category": category,
                "logic": logic,
                "family_tag": family_tag,
                "family_rationale": family_rationale,
                "valid": False,
                "latest_date": None,
                "stats": None,
                "error": str(exc),
            }
        if not str(family_rationale or "").strip():
            return {
                "name": name,
                "expression": expression,
                "category": category,
                "logic": logic,
                "family_tag": resolved_family_tag,
                "family_rationale": family_rationale,
                "valid": False,
                "latest_date": None,
                "stats": None,
                "error": "family_rationale must be provided by the Ideator",
            }
        if not expression:
            return {
                "name": name,
                "expression": expression,
                "category": category,
                "logic": logic,
                "family_tag": resolved_family_tag,
                "family_rationale": family_rationale,
                "valid": False,
                "latest_date": None,
                "stats": None,
                "error": "Missing required factor expression",
            }
        result = self.validate_and_compute_factor(expression)
        return {
            "name": name,
            "expression": expression,
            "category": category,
            "logic": logic,
            "family_tag": resolved_family_tag,
            "family_rationale": family_rationale,
            "valid": result.valid,
            "latest_date": result.latest_date,
            "stats": result.stats,
            "error": result.error,
        }

    def query_base_factor_pool(
        self,
        category: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        path = Path("checkpoints/base_factors.json")
        if not path.exists():
            return {"valid": False, "error": "base_factors.json not found", "factors": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        definitions = payload.get("factor_definitions") or []
        if category:
            definitions = [item for item in definitions if item.get("category") == category]
        rows = [
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "formula": item.get("formula"),
                "direction": item.get("direction"),
                "description": item.get("description"),
            }
            for item in definitions[: max(1, min(int(limit or 30), 100))]
        ]
        categories: dict[str, int] = {}
        for item in payload.get("factor_definitions") or []:
            key = item.get("category") or "unknown"
            categories[key] = categories.get(key, 0) + 1
        return {
            "valid": True,
            "category_filter": category,
            "factor_count": len(payload.get("factor_definitions") or []),
            "returned_count": len(rows),
            "categories": categories,
            "factors": rows,
        }

    def describe_data_fields(self) -> dict[str, Any]:
        snapshot = self.loader.snapshot(self.panel)
        fields = []
        for column in self.panel.columns:
            series = self.panel[column]
            item: dict[str, Any] = {
                "name": column,
                "dtype": str(series.dtype),
                "non_null_ratio": self._clean_float(series.notna().mean()),
            }
            if pd.api.types.is_numeric_dtype(series):
                item.update(
                    {
                        "mean": self._clean_float(series.mean()),
                        "std": self._clean_float(series.std()),
                        "min": self._clean_float(series.min()),
                        "max": self._clean_float(series.max()),
                    }
                )
            fields.append(item)
        return {
            "valid": True,
            "source": snapshot.source,
            "rows": snapshot.rows,
            "symbols": snapshot.symbols,
            "start_date": snapshot.start_date,
            "end_date": snapshot.end_date,
            "fields": fields,
            "expression_fields": [
                field
                for field in ["open", "high", "low", "close", "vol", "amount", "pct_chg", "flow", "flow_ratio"]
                if field in self.panel.columns
            ],
        }

    def analyze_factor_orthogonality(
        self,
        expression: str,
        reference_expressions: list[str] | None = None,
        recent_days: int = 252,
        max_base_factors: int = 20,
    ) -> dict[str, Any]:
        candidate = self._evaluate_expression_series(expression)
        if candidate is None:
            result = self.validate_and_compute_factor(expression)
            return {"valid": False, "expression": expression, "error": result.error}

        refs = self._base_factor_references(max_base_factors)
        for idx, ref_expression in enumerate(reference_expressions or [], start=1):
            refs.append(
                {
                    "name": f"reference_{idx}",
                    "category": "provided",
                    "expression": ref_expression,
                }
            )

        recent_dates = self.panel["date"].drop_duplicates().sort_values().tail(max(20, int(recent_days or 252)))
        mask = self.panel["date"].isin(recent_dates)
        candidate_recent = candidate[mask]
        correlations = []
        for ref in refs:
            ref_series = self._evaluate_expression_series(ref["expression"])
            if ref_series is None:
                correlations.append(
                    {
                        "name": ref["name"],
                        "category": ref.get("category"),
                        "expression": ref["expression"],
                        "valid": False,
                        "error": "reference expression could not be computed",
                    }
                )
                continue
            corr = self._series_corr(candidate_recent, ref_series[mask])
            correlations.append(
                {
                    "name": ref["name"],
                    "category": ref.get("category"),
                    "expression": ref["expression"],
                    "valid": True,
                    "correlation": corr,
                    "abs_correlation": abs(corr) if corr is not None else None,
                }
            )

        valid_corrs = [item for item in correlations if item.get("abs_correlation") is not None]
        max_abs = max((item["abs_correlation"] for item in valid_corrs), default=None)
        mean_abs = self._clean_float(np.mean([item["abs_correlation"] for item in valid_corrs])) if valid_corrs else None
        high_corr = [item for item in valid_corrs if item["abs_correlation"] >= 0.7]
        return {
            "valid": True,
            "expression": expression,
            "recent_days": recent_days,
            "reference_count": len(refs),
            "max_abs_correlation": self._clean_float(max_abs),
            "mean_abs_correlation": mean_abs,
            "orthogonal_enough": bool(max_abs is None or max_abs < 0.7),
            "high_correlation_count": len(high_corr),
            "top_correlations": sorted(
                valid_corrs,
                key=lambda item: item.get("abs_correlation") or 0,
                reverse=True,
            )[:10],
            "correlations": correlations,
        }

    def evaluate_factor_quick(
        self,
        expression: str,
        horizon: int = 5,
        recent_days: int = 252,
        quantile_groups: int = 5,
        multi_horizons: list[int] | None = None,
    ) -> dict[str, Any]:
        computed = self.validate_and_compute_factor(expression)
        if not computed.valid or computed.values is None:
            return {"valid": False, "error": computed.error}

        node = parse_expression(expand_base_factor_references(expression))
        factor = self.engine.evaluate(node)
        frame = self.panel[["date", "symbol", "close"]].copy()
        frame["factor"] = factor.replace([np.inf, -np.inf], np.nan)
        horizons = self._evaluation_horizons(horizon, multi_horizons)
        horizon_metrics = [
            self._evaluate_horizon_metrics(frame, item, recent_days, quantile_groups)
            for item in horizons
        ]
        primary = next((item for item in horizon_metrics if item["horizon"] == horizon), horizon_metrics[0])
        mean_ic = primary["mean_ic"]
        icir = primary["icir"]
        win_rate = primary["ic_win_rate"]
        long_short_return = primary["long_short_return"]
        direction = primary["direction"]
        monotonicity = primary.get("stratified_backtest", {}).get("directional_monotonicity")
        long_short_score = min(abs(long_short_return or 0) / 0.01 * 10, 10)
        monotonic_rank_score = max(0, min(monotonicity or 0, 1)) * 10
        ic_values = []
        ic_score = min(abs(mean_ic or 0) / 0.05 * 40, 40)
        icir_score = min(abs(icir or 0) / 0.5 * 30, 30)
        monotonicity_score = long_short_score + monotonic_rank_score
        win_rate_score = max(0, min(((win_rate or 0) - 0.5) / 0.2 * 10, 10))
        score = self._clean_float(ic_score + icir_score + monotonicity_score + win_rate_score)
        is_effective = bool(
            mean_ic is not None
            and icir is not None
            and win_rate is not None
            and long_short_return is not None
            and abs(mean_ic) > 0.02
            and abs(icir) > 0.1
            and win_rate > 0.55
            and abs(long_short_return) > 0.005
        )
        is_excellent = bool(
            mean_ic is not None
            and icir is not None
            and win_rate is not None
            and long_short_return is not None
            and abs(mean_ic) > 0.05
            and abs(icir) > 0.5
            and win_rate > 0.65
            and abs(long_short_return) > 0.01
        )
        quality = "excellent" if is_excellent else "effective" if is_effective else "weak"
        return {
            "valid": True,
            "horizon": horizon,
            "recent_days": recent_days,
            "quantile_groups": quantile_groups,
            "mean_ic": mean_ic,
            "icir": icir,
            "ic_win_rate": win_rate,
            "long_short_return": long_short_return,
            "ic_observations": primary["ic_observations"],
            "long_short_observations": primary["long_short_observations"],
            "direction": direction,
            "multi_period_metrics": horizon_metrics,
            "stratified_backtest": primary["stratified_backtest"],
            "score": score,
            "score_breakdown": {
                "ic_score": self._clean_float(ic_score),
                "icir_score": self._clean_float(icir_score),
                "monotonicity_score": self._clean_float(monotonicity_score),
                "long_short_score": self._clean_float(long_short_score),
                "monotonic_rank_score": self._clean_float(monotonic_rank_score),
                "win_rate_score": self._clean_float(win_rate_score),
            },
            "thresholds": {
                "effective": {
                    "abs_mean_ic": 0.02,
                    "abs_icir": 0.1,
                    "directional_ic_win_rate": 0.55,
                    "abs_long_short_return": 0.005,
                },
                "excellent": {
                    "abs_mean_ic": 0.05,
                    "abs_icir": 0.5,
                    "directional_ic_win_rate": 0.65,
                    "abs_long_short_return": 0.01,
                },
            },
            "is_effective": is_effective,
            "is_excellent": is_excellent,
            "quality": quality,
        }

    @staticmethod
    def _evaluation_horizons(primary_horizon: int, multi_horizons: list[int] | None) -> list[int]:
        values = multi_horizons or [1, primary_horizon, 10, 20]
        horizons: list[int] = []
        for value in values:
            try:
                horizon = int(value)
            except (TypeError, ValueError):
                continue
            if horizon <= 0 or horizon in horizons:
                continue
            horizons.append(horizon)
        if primary_horizon not in horizons:
            horizons.insert(0, primary_horizon)
        return horizons

    def _evaluate_horizon_metrics(
        self,
        base_frame: pd.DataFrame,
        horizon: int,
        recent_days: int,
        quantile_groups: int,
    ) -> dict[str, Any]:
        frame = base_frame.copy()
        frame["future_return"] = frame.groupby("symbol")["close"].shift(-horizon) / frame["close"] - 1
        recent_dates = frame["date"].drop_duplicates().sort_values().tail(recent_days)
        frame = frame[frame["date"].isin(recent_dates)]
        ic_values = []
        long_short_values = []
        bucket_returns: dict[int, list[float]] = {}
        for _, group in frame.dropna(subset=["factor", "future_return"]).groupby("date"):
            if group["factor"].nunique() <= 1 or group["future_return"].nunique() <= 1:
                continue
            ic_values.append(group["factor"].corr(group["future_return"], method="spearman"))
            ranked = group.copy()
            bucket_count = min(quantile_groups, len(ranked))
            if bucket_count < 2:
                continue
            ranked["bucket"] = pd.qcut(
                ranked["factor"].rank(method="first"),
                q=bucket_count,
                labels=False,
                duplicates="drop",
            )
            if ranked["bucket"].nunique() < 2:
                continue
            grouped_returns = ranked.groupby("bucket")["future_return"].mean()
            for bucket, value in grouped_returns.items():
                bucket_returns.setdefault(int(bucket), []).append(float(value))
            low = grouped_returns.loc[grouped_returns.index.min()]
            high = grouped_returns.loc[grouped_returns.index.max()]
            long_short_values.append(high - low)
        ic = pd.Series(ic_values, dtype="float64").dropna()
        long_short = pd.Series(long_short_values, dtype="float64").dropna()
        mean_ic = self._clean_float(ic.mean())
        icir = self._clean_float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() else None
        long_short_return = self._clean_float(long_short.mean())
        direction = "positive" if mean_ic is not None and mean_ic > 0 else "negative" if mean_ic is not None and mean_ic < 0 else "neutral"
        if not len(ic):
            win_rate = None
        elif direction == "negative":
            win_rate = self._clean_float((ic < 0).mean())
        elif direction == "positive":
            win_rate = self._clean_float((ic > 0).mean())
        else:
            win_rate = self._clean_float((ic == 0).mean())
        bucket_summary = [
            {"bucket": bucket + 1, "mean_return": self._clean_float(np.mean(values))}
            for bucket, values in sorted(bucket_returns.items())
        ]
        monotonicity_corr = None
        if len(bucket_summary) >= 2:
            x = pd.Series([item["bucket"] for item in bucket_summary], dtype="float64")
            y = pd.Series([item["mean_return"] for item in bucket_summary], dtype="float64")
            monotonicity_corr = self._clean_float(x.corr(y, method="spearman"))
        if direction == "negative" and monotonicity_corr is not None:
            directional_monotonicity = self._clean_float(-monotonicity_corr)
        else:
            directional_monotonicity = monotonicity_corr
        return {
            "horizon": horizon,
            "mean_ic": mean_ic,
            "icir": icir,
            "ic_win_rate": win_rate,
            "long_short_return": long_short_return,
            "ic_observations": int(len(ic)),
            "long_short_observations": int(len(long_short)),
            "direction": direction,
            "stratified_backtest": {
                "group_count": quantile_groups,
                "group_mean_returns": bucket_summary,
                "long_short_return": long_short_return,
                "monotonicity_corr": monotonicity_corr,
                "directional_monotonicity": directional_monotonicity,
            },
        }

    @staticmethod
    def _clean_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 10)

    def _evaluate_expression_series(self, expression: str) -> pd.Series | None:
        try:
            node = parse_expression(expand_base_factor_references(expression))
            values = self.engine.evaluate(node)
            if not isinstance(values, pd.Series) or len(values) != len(self.panel):
                return None
            return values.replace([np.inf, -np.inf], np.nan)
        except Exception:
            return None

    def _base_factor_references(self, limit: int) -> list[dict[str, Any]]:
        path = Path("checkpoints/base_factors.json")
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            {
                "name": item.get("name"),
                "category": item.get("category"),
                "expression": BASE_FACTOR_EXPRESSION_ALIASES.get(item.get("name"), item.get("formula")),
            }
            for item in (payload.get("factor_definitions") or [])[: max(1, min(int(limit or 20), 100))]
            if item.get("name") and item.get("formula")
        ]

    @staticmethod
    def _series_corr(left: pd.Series, right: pd.Series) -> float | None:
        frame = pd.DataFrame({"left": left, "right": right}).dropna()
        if len(frame) < 30 or frame["left"].nunique() <= 1 or frame["right"].nunique() <= 1:
            return None
        return FactorToolbox._clean_float(frame["left"].corr(frame["right"], method="spearman"))

    @classmethod
    def _records(cls, frame: pd.DataFrame) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            out.append(
                {
                    "date": str(row.date.date()),
                    "symbol": row.symbol,
                    "value": cls._clean_float(row.value),
                }
            )
        return out
