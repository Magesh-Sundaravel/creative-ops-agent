from decimal import Decimal

import pytest

from creative_ops.pricing import cost_usd

HAIKU = "claude-haiku-5-5"


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected"),
    [
        (0, 0, "0"),
        (100_000, 1_000, "0.0105"),  # 100,000 prompt tokens is still the low tier
        (100_001, 1_000, "0.0525005"),  # above: 0.50/2.50 per million
        (1_000, 2_000, "0.0011"),
    ],
)
def test_haiku_cost_at_each_tier(input_tokens, output_tokens, expected):
    assert cost_usd(HAIKU, input_tokens, output_tokens) == Decimal(expected)


def test_unknown_model_has_no_cost():
    assert cost_usd("some-other-model", 10, 10) is None
