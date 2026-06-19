from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[,()+\-*/]")
BASE_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
BASE_FACTOR_EXPRESSION_ALIASES = {
    "momentum_20d": "ts_return(close,20)",
    "momentum_60d": "ts_return(close,60)",
    "reversal_5d": "neg(ts_return(close,5))",
    "volatility_20d": "ts_std(pct_chg,20)",
    "volatility_60d": "ts_std(pct_chg,60)",
    "volume_price_corr_20d": "ts_corr(pct_chg,amount,20)",
    "turnover_zscore_20d": "div(sub(amount,ts_mean(amount,20)),ts_std(amount,20))",
    "trend_strength_20_60d": "sub(div(close,ts_mean(close,20)),div(close,ts_mean(close,60)))",
    "liquidity_turnover_20d": "ts_mean(amount,20)",
    "liquidity_amihud_20d": "ts_mean(div(abs(pct_chg),amount),20)",
}


@dataclass(frozen=True)
class Call:
    name: str
    args: list["Node"]


@dataclass(frozen=True)
class Name:
    value: str


@dataclass(frozen=True)
class Number:
    value: float


Node = Union[Call, Name, Number]


class ExpressionParser:
    def __init__(self, expression: str) -> None:
        self.tokens = TOKEN_RE.findall(expression.replace(" ", ""))
        self.pos = 0

    def parse(self) -> Node:
        node = self._parse_node()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")
        return node

    def _parse_node(self) -> Node:
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")

        token = self._take()
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return Number(float(token))

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            raise ValueError(f"Unexpected token: {token}")

        if self._peek() != "(":
            return Name(token)

        self._take_expected("(")
        args: list[Node] = []
        if self._peek() != ")":
            while True:
                args.append(self._parse_node())
                if self._peek() == ",":
                    self._take()
                    continue
                break
        self._take_expected(")")
        return Call(token, args)

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _take_expected(self, expected: str) -> None:
        token = self._take()
        if token != expected:
            raise ValueError(f"Expected {expected}, got {token}")


def parse_expression(expression: str) -> Node:
    return ExpressionParser(expression).parse()


def expand_base_factor_references(expression: str, checkpoint_path: str | Path = "checkpoints/base_factors.json") -> str:
    path = Path(checkpoint_path)
    if not path.exists() or "$" not in expression:
        return expression

    data = json.loads(path.read_text(encoding="utf-8"))
    definitions = {
        item["name"]: item["formula"]
        for item in data.get("factor_definitions", [])
        if "name" in item and "formula" in item
    }
    definitions.update(BASE_FACTOR_EXPRESSION_ALIASES)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in definitions:
            raise ValueError(f"Unknown base factor reference: ${name}")
        return definitions[name]

    return BASE_REF_RE.sub(replace, expression)
