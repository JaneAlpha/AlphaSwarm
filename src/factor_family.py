from __future__ import annotations

from typing import Any


FAMILY_TAGS = {
    "price_momentum",
    "price_reversal",
    "volatility_risk",
    "volume_price_confirmation",
    "liquidity_pressure",
    "trend_strength",
    "flow_attention",
    "cross_family_composite",
    "other",
}


def validate_family_tag(value: Any) -> str:
    family_tag = str(value or "").strip()
    if family_tag not in FAMILY_TAGS:
        raise ValueError(
            "family_tag must be provided by the Ideator and must be one of: "
            + ", ".join(sorted(FAMILY_TAGS))
        )
    return family_tag
