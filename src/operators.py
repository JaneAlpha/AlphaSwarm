from __future__ import annotations

import numpy as np
import pandas as pd

from .expression_utils import Call, Name, Node, Number


class OperatorEngine:
    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.group = self.panel.groupby("symbol", group_keys=False)

    def evaluate(self, node: Node) -> pd.Series | float:
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Name):
            if node.value not in self.panel.columns:
                raise ValueError(f"Unknown field: {node.value}")
            return self.panel[node.value]
        if isinstance(node, Call):
            args = [self.evaluate(arg) for arg in node.args]
            return self._call(node.name, args)
        raise TypeError(f"Unsupported node: {node!r}")

    def _call(self, name: str, args: list[pd.Series | float]) -> pd.Series | float:
        if name == "add":
            return args[0] + args[1]
        if name == "sub":
            return args[0] - args[1]
        if name == "mul":
            return args[0] * args[1]
        if name == "div":
            return args[0] / self._safe_denominator(args[1])
        if name == "neg":
            return -args[0]
        if name == "abs":
            return args[0].abs()
        if name == "sign":
            return np.sign(args[0])
        if name == "log":
            return np.log(args[0].where(args[0] > 0))
        if name == "delay":
            return self._series(args[0]).groupby(self.panel["symbol"]).shift(self._window(args[1]))
        if name == "sqrt":
            return np.sqrt(self._series(args[0]).where(self._series(args[0]) >= 0))
        if name == "max":
            return pd.concat([self._series(args[0]), self._series(args[1])], axis=1).max(axis=1)
        if name == "min":
            return pd.concat([self._series(args[0]), self._series(args[1])], axis=1).min(axis=1)
        if name == "power":
            return np.power(args[0], args[1])
        if name == "ts_return":
            window = self._window(args[1])
            return self._series(args[0]).groupby(self.panel["symbol"]).pct_change(window)
        if name == "ts_delay":
            return self._series(args[0]).groupby(self.panel["symbol"]).shift(self._window(args[1]))
        if name == "ts_delta":
            series = self._series(args[0])
            return series - series.groupby(self.panel["symbol"]).shift(self._window(args[1]))
        if name == "ts_mean":
            return self._rolling(args[0], args[1], "mean")
        if name == "ts_std":
            return self._rolling(args[0], args[1], "std")
        if name == "ts_sum":
            return self._rolling(args[0], args[1], "sum")
        if name == "ts_max":
            return self._rolling(args[0], args[1], "max")
        if name == "ts_min":
            return self._rolling(args[0], args[1], "min")
        if name == "ts_corr":
            left = self._series(args[0])
            right = self._series(args[1])
            window = self._window(args[2])
            data = pd.DataFrame(
                {"symbol": self.panel["symbol"], "left": left, "right": right},
                index=self.panel.index,
            )
            return data.groupby("symbol", group_keys=False).apply(
                lambda x: x["left"].rolling(window, min_periods=max(3, window // 2)).corr(x["right"])
            )
        raise ValueError(f"Unsupported operator: {name}")

    def _rolling(self, series: pd.Series | float, window_value: pd.Series | float, method: str) -> pd.Series:
        source = self._series(series)
        window = self._window(window_value)
        rolled = source.groupby(self.panel["symbol"]).rolling(window, min_periods=max(3, window // 2))
        result = getattr(rolled, method)().reset_index(level=0, drop=True)
        return result

    @staticmethod
    def _safe_denominator(value: pd.Series | float) -> pd.Series | float:
        if isinstance(value, pd.Series):
            return value.replace(0, np.nan)
        return np.nan if value == 0 else value

    @staticmethod
    def _series(value: pd.Series | float) -> pd.Series:
        if not isinstance(value, pd.Series):
            raise TypeError("Operator requires a series argument")
        return value

    @staticmethod
    def _window(value: pd.Series | float) -> int:
        if isinstance(value, pd.Series):
            raise TypeError("Window must be a numeric literal")
        window = int(value)
        if window <= 0:
            raise ValueError("Window must be positive")
        return window
