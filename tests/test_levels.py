from __future__ import annotations

import pytest

from despawn_pytorch import get_num_levels


class TestGetNumLevels:
    @pytest.mark.parametrize(
        ("signal_length", "expected"),
        [(2, 1), (3, 1), (4, 2), (7, 2), (8, 3), (2000, 10)],
    )
    def test_level_selection(self, signal_length: int, expected: int) -> None:
        """Check that level selection uses floor of the base two logarithm."""
        assert get_num_levels(signal_length) == expected

    @pytest.mark.parametrize("signal_length", [1, 0, -1, 2.5, True])
    def test_invalid_length(self, signal_length: int) -> None:
        """Check that invalid signal lengths raise a clear error."""
        with pytest.raises(ValueError, match="signal_length"):
            get_num_levels(signal_length)
