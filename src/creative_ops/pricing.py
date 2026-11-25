"""Model prices, hand-kept (ADR 0006). Rates go stale: re-check against the source."""

from decimal import Decimal

# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Rates given in issue #112; not re-checked against the source page (date checked: unchecked).
TIER_LIMIT = 100_000  # prompt tokens; above it the higher rates apply to the whole call

# model id -> ((input, output) USD per million tokens up to TIER_LIMIT, same above it)
_RATES: dict[str, tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]] = {
    "claude-haiku-5-5": (
        (Decimal("0.10"), Decimal("0.50")),
        (Decimal("0.50"), Decimal("2.50")),
    ),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    """The cost of one model call, or None for a model with no known rates."""
    rates = _RATES.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates[0] if input_tokens <= TIER_LIMIT else rates[1]
    return (input_rate * input_tokens + output_rate * output_tokens) / 1_000_000
