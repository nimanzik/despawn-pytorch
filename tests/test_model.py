from __future__ import annotations

from typing import Literal

import pytest
import torch

from despawn_pytorch.model import Despawn

type ActiveConstraintLike = Literal["cqf", "per_layer", "per_filter", "free"]
type LegacyConstraintLike = Literal["CQF", "PerLayer", "PerFilter", "Free"]


class TestDespawn:
    @pytest.mark.parametrize(
        ("constraint", "expected_kernel_params"),
        [("cqf", 1), ("per_layer", 3), ("per_filter", 6), ("free", 12)],
    )
    def test_kernel_constraint_parameter_sharing(
        self, constraint: ActiveConstraintLike, expected_kernel_params: int
    ) -> None:
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            threshold_init=0.25,
        )

        assert len(model.kernel_store) == expected_kernel_params

        x = torch.randn(2, 1, 15, 1)
        recon, coeff_loss = model(x)
        recon2, approx, details = model.decompose(x)

        assert recon.shape == x.shape
        assert coeff_loss.shape == (2, 1, 1, 1)
        assert recon2.shape == x.shape
        torch.testing.assert_close(recon2, recon)
        assert approx.shape == (2, 1, 2, 1)
        assert [detail.shape for detail in details] == [
            torch.Size([2, 1, 2, 1]),
            torch.Size([2, 1, 4, 1]),
            torch.Size([2, 1, 8, 1]),
        ]

    @pytest.mark.parametrize("constraint", ["CQF", "PerLayer", "PerFilter", "Free"])
    def test_rejects_legacy_constraint_names(
        self, constraint: LegacyConstraintLike
    ) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint=constraint)  # ty: ignore[invalid-argument-type]

    def test_rejects_unknown_constraint(self) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint="unknown")  # ty: ignore[invalid-argument-type]

    def test_loss_coeff_none_returns_zero(self) -> None:
        model = Despawn(loss_coeff=None)
        _, coeff_loss = model(torch.randn(2, 1, 8, 1))

        assert coeff_loss.shape == (1, 1, 1, 1)
        assert torch.equal(coeff_loss, torch.zeros_like(coeff_loss))
