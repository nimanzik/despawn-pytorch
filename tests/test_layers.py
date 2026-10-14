from __future__ import annotations

import pytest
import torch

from despawn_pytorch.layers import HardThreshold


class TestHardThreshold:
    @pytest.mark.parametrize("init_value", [0, 1])
    def test_integer_matches_float_threshold(self, init_value: int) -> None:
        integer = HardThreshold(init_value=init_value)
        floating = HardThreshold(init_value=float(init_value))
        signal = torch.linspace(-2, 2, 9).reshape(1, 1, -1, 1)

        torch.testing.assert_close(integer(signal), floating(signal))
        integer(signal).sum().backward()
        floating(signal).sum().backward()
        for actual, expected in zip(
            integer.parameters(), floating.parameters(), strict=True
        ):
            assert actual.dtype == torch.get_default_dtype()
            assert actual.requires_grad
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad)

    @pytest.mark.parametrize("init_value", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_threshold(self, init_value: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            HardThreshold(init_value=init_value)

    @pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            HardThreshold(alpha=alpha)

    def test_uses_non_negative_thresholds(self) -> None:
        constrained = HardThreshold(init_value=0.5)
        negative = HardThreshold(init_value=0.5)
        with torch.no_grad():
            negative.positive_threshold.fill_(-0.5)
            negative.negative_threshold.fill_(-0.5)

        signal = torch.linspace(-2, 2, 9).reshape(1, 1, -1, 1)

        torch.testing.assert_close(negative(signal), constrained(signal))
