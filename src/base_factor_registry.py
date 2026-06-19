from __future__ import annotations

from .schemas import FactorDefinition


BASE_FACTOR_DEFINITIONS: list[FactorDefinition] = [
    FactorDefinition(
        name="momentum_20d",
        category="momentum",
        description="20 trading day cumulative NAV return.",
        formula="nav / delay(nav, 20) - 1",
        direction="higher_is_better",
        lookback=20,
        required_fields=["nav"],
    ),
    FactorDefinition(
        name="momentum_60d",
        category="momentum",
        description="60 trading day cumulative NAV return.",
        formula="nav / delay(nav, 60) - 1",
        direction="higher_is_better",
        lookback=60,
        required_fields=["nav"],
    ),
    FactorDefinition(
        name="reversal_5d",
        category="reversal",
        description="Negative 5 trading day return, capturing short-term reversal.",
        formula="-(nav / delay(nav, 5) - 1)",
        direction="higher_is_better",
        lookback=5,
        required_fields=["nav"],
    ),
    FactorDefinition(
        name="volatility_20d",
        category="volatility",
        description="Annualized volatility of daily returns over 20 trading days.",
        formula="std(return, 20) * sqrt(252)",
        direction="lower_is_better",
        lookback=20,
        required_fields=["return"],
    ),
    FactorDefinition(
        name="volatility_60d",
        category="volatility",
        description="Annualized volatility of daily returns over 60 trading days.",
        formula="std(return, 60) * sqrt(252)",
        direction="lower_is_better",
        lookback=60,
        required_fields=["return"],
    ),
    FactorDefinition(
        name="volume_price_corr_20d",
        category="volume_price",
        description="20 day correlation between returns and turnover amount.",
        formula="corr(return, turnover_amount, 20)",
        direction="context_dependent",
        lookback=20,
        required_fields=["return", "turnover_amount"],
    ),
    FactorDefinition(
        name="turnover_zscore_20d",
        category="volume_price",
        description="Latest turnover amount standardized by its 20 day history.",
        formula="(turnover_amount - mean(turnover_amount, 20)) / std(turnover_amount, 20)",
        direction="context_dependent",
        lookback=20,
        required_fields=["turnover_amount"],
    ),
    FactorDefinition(
        name="trend_strength_20_60d",
        category="trend",
        description="Difference between 20 day and 60 day moving-average position.",
        formula="nav / mean(nav, 20) - nav / mean(nav, 60)",
        direction="higher_is_better",
        lookback=60,
        required_fields=["nav"],
    ),
    FactorDefinition(
        name="liquidity_turnover_20d",
        category="liquidity",
        description="Average turnover amount over 20 trading days.",
        formula="mean(turnover_amount, 20)",
        direction="higher_is_better",
        lookback=20,
        required_fields=["turnover_amount"],
    ),
    FactorDefinition(
        name="liquidity_amihud_20d",
        category="liquidity",
        description="Amihud-style illiquidity: mean absolute return divided by turnover.",
        formula="mean(abs(return) / turnover_amount, 20)",
        direction="lower_is_better",
        lookback=20,
        required_fields=["return", "turnover_amount"],
    ),
]


def definitions_by_category() -> dict[str, list[FactorDefinition]]:
    grouped: dict[str, list[FactorDefinition]] = {}
    for definition in BASE_FACTOR_DEFINITIONS:
        grouped.setdefault(definition.category, []).append(definition)
    return grouped
