from __future__ import annotations

import pytest
import torch

from despawn_pytorch.layers import HardThreshold


class TestHardThreshold:
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
