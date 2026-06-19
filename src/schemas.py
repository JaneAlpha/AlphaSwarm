from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DataSnapshot:
    source: str
    rows: int
    symbols: int
    start_date: str
    end_date: str
    fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    category: str
    description: str
    formula: str
    direction: str
    lookback: int
    required_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
