from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_factor_registry import BASE_FACTOR_DEFINITIONS, definitions_by_category
from .dataloader import ETFDataLoader


class BaseFactorGenerator:
    def __init__(self, loader: ETFDataLoader | None = None) -> None:
        self.loader = loader or ETFDataLoader()

    def build(self) -> dict[str, Any]:
        panel = self.loader.load_daily_panel()
        factors = self._calculate_factors(panel)
        latest = self._latest_cross_section(factors)
        snapshot = self.loader.snapshot(panel)

        factor_columns = [definition.name for definition in BASE_FACTOR_DEFINITIONS]
        coverage = {
            column: {
                "non_null": int(latest[column].notna().sum()),
                "coverage_ratio": round(float(latest[column].notna().mean()), 6),
            }
            for column in factor_columns
        }

        return {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "step": "step0_base_factor_pool",
            "data_snapshot": snapshot.to_dict(),
            "factor_categories": {
                category: [item.name for item in items]
                for category, items in definitions_by_category().items()
            },
            "factor_definitions": [
                definition.to_dict() for definition in BASE_FACTOR_DEFINITIONS
            ],
            "latest_date": str(latest["date"].max().date()),
            "latest_cross_section": self._records(latest[["date", "symbol", *factor_columns]]),
            "quality": {
                "factor_count": len(factor_columns),
                "symbol_count": int(latest["symbol"].nunique()),
                "coverage": coverage,
            },
        }

    def write(self, output_path: str | Path) -> dict[str, Any]:
        result = self.build()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _calculate_factors(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        group = out.groupby("symbol", group_keys=False)

        out["momentum_20d"] = group["nav"].pct_change(20)
        out["momentum_60d"] = group["nav"].pct_change(60)
        out["reversal_5d"] = -group["nav"].pct_change(5)
        out["volatility_20d"] = group["return"].transform(
            lambda s: s.rolling(20, min_periods=15).std() * np.sqrt(252)
        )
        out["volatility_60d"] = group["return"].transform(
            lambda s: s.rolling(60, min_periods=40).std() * np.sqrt(252)
        )
        out["volume_price_corr_20d"] = group.apply(
            lambda x: x["return"].rolling(20, min_periods=15).corr(x["turnover_amount"])
        ).reset_index(level=0, drop=True)
        turnover_mean_20 = group["turnover_amount"].transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )
        turnover_std_20 = group["turnover_amount"].transform(
            lambda s: s.rolling(20, min_periods=15).std()
        )
        out["turnover_zscore_20d"] = (out["turnover_amount"] - turnover_mean_20) / turnover_std_20

        ma20 = group["nav"].transform(lambda s: s.rolling(20, min_periods=15).mean())
        ma60 = group["nav"].transform(lambda s: s.rolling(60, min_periods=40).mean())
        out["trend_strength_20_60d"] = out["nav"] / ma20 - out["nav"] / ma60

        out["liquidity_turnover_20d"] = turnover_mean_20
        safe_turnover = out["turnover_amount"].replace(0, np.nan)
        out["_amihud"] = out["return"].abs() / safe_turnover
        out["liquidity_amihud_20d"] = out.groupby("symbol")["_amihud"].transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )
        return out.drop(columns=["_amihud"])

    @staticmethod
    def _latest_cross_section(factors: pd.DataFrame) -> pd.DataFrame:
        idx = factors.groupby("symbol")["date"].idxmax()
        latest = factors.loc[idx].sort_values("symbol").reset_index(drop=True)
        return latest

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clean = frame.replace({np.nan: None, pd.NaT: None})
        records = clean.to_dict(orient="records")
        for row in records:
            if hasattr(row.get("date"), "date"):
                row["date"] = str(row["date"].date())
        return records
