from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from .schemas import DataSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


class ETFDataLoader:
    """Read-only access to this project's local ETF data files."""

    def __init__(self, data_root: str | os.PathLike[str] | None = None) -> None:
        root = data_root or os.getenv("ETF_DATA_ROOT")
        self.data_root = Path(root) if root else DEFAULT_DATA_ROOT

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def daily_panel_path(self) -> Path:
        return self.processed_dir / "etf_daily_panel.parquet"

    @property
    def stock_db_path(self) -> Path:
        return self.data_root / "stock_data.db"

    def load_daily_panel(self) -> pd.DataFrame:
        if self.daily_panel_path.exists():
            frame = pd.read_parquet(self.daily_panel_path)
            return self._clean_daily_panel(frame, source=str(self.daily_panel_path))

        if self.stock_db_path.exists():
            return self._load_daily_from_sqlite()

        raise FileNotFoundError(
            "No ETF daily data found. Expected processed/etf_daily_panel.parquet "
            f"or stock_data.db under {self.data_root}."
        )

    def snapshot(self, frame: pd.DataFrame) -> DataSnapshot:
        return DataSnapshot(
            source=frame.attrs.get("source", "unknown"),
            rows=int(len(frame)),
            symbols=int(frame["symbol"].nunique()),
            start_date=str(frame["date"].min().date()),
            end_date=str(frame["date"].max().date()),
            fields=list(frame.columns),
        )

    def _load_daily_from_sqlite(self) -> pd.DataFrame:
        with sqlite3.connect(self.stock_db_path) as conn:
            frame = pd.read_sql_query(
                """
                select
                    trade_date as date,
                    symbol,
                    close as nav,
                    volume,
                    close,
                    open,
                    high,
                    low
                from daily
                order by symbol, trade_date
                """,
                conn,
            )
        frame["turnover_amount"] = frame["close"] * frame["volume"]
        frame["return"] = frame.groupby("symbol")["nav"].pct_change()
        return self._clean_daily_panel(frame, source=str(self.stock_db_path))

    @staticmethod
    def _clean_daily_panel(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        required = {"date", "symbol", "nav"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Daily panel missing required fields: {sorted(missing)}")

        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["symbol"] = out["symbol"].astype(str)
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

        if "return" not in out.columns:
            out["return"] = out.groupby("symbol")["nav"].pct_change()

        if "turnover_amount" not in out.columns:
            out["turnover_amount"] = pd.NA

        out["close"] = out["nav"]
        out["amount"] = out["turnover_amount"]
        out["pct_chg"] = out["return"]
        if "volume" in out.columns and "vol" not in out.columns:
            out["vol"] = out["volume"]
        elif "shares" in out.columns and "vol" not in out.columns:
            out["vol"] = out["shares"]
        if "open" not in out.columns:
            out["open"] = out["nav"]
        if "high" not in out.columns:
            out["high"] = out["nav"]
        if "low" not in out.columns:
            out["low"] = out["nav"]

        out.attrs["source"] = source
        return out
