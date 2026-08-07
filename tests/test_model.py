from __future__ import annotations

from typing import Literal

import numpy as np
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

        x = torch.randn(2, 15)
        recon, coeff_loss = model(x)
        recon2, approx, details = model.decompose(x)

        assert recon.shape == x.shape
        assert coeff_loss.shape == (2,)
        assert recon2.shape == x.shape
        torch.testing.assert_close(recon2, recon)
        assert approx.shape == (2, 2)
        assert [detail.shape for detail in details] == [
            torch.Size([2, 2]),
            torch.Size([2, 4]),
            torch.Size([2, 8]),
        ]

    @pytest.mark.parametrize("shape", [(15,), (2, 15), (2, 3, 15)])
    def test_time_last_shapes(self, shape: tuple[int, ...]) -> None:
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1], n_levels=3, threshold_init=0.25
        )
        x = torch.randn(shape)

        reconstruction, coefficient_loss = model(x)
        _, approximation, details = model.decompose(x)

        assert reconstruction.shape == x.shape
        assert coefficient_loss.shape == x.shape[:-1]
        assert approximation.shape == (*x.shape[:-1], 2)
        assert [detail.shape for detail in details] == [
            torch.Size((*x.shape[:-1], 2)),
            torch.Size((*x.shape[:-1], 4)),
            torch.Size((*x.shape[:-1], 8)),
        ]

    def test_leading_dimensions_are_independent_signals(self) -> None:
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1], n_levels=3, threshold_init=0.25
        )
        x = torch.randn(2, 3, 15)

        reconstruction, coefficient_loss = model(x)
        flat_reconstruction, flat_coefficient_loss = model(x.reshape(-1, 15))

        torch.testing.assert_close(reconstruction, flat_reconstruction.reshape(x.shape))
        torch.testing.assert_close(
            coefficient_loss, flat_coefficient_loss.reshape(x.shape[:-1])
        )

    @pytest.mark.parametrize(
        "x", [torch.tensor(1.0), torch.empty(2, 0), torch.empty(0, 8)]
    )
    def test_rejects_inputs_without_signals(self, x: torch.Tensor) -> None:
        model = Despawn()

        with pytest.raises(ValueError):
            model(x)

    @pytest.mark.parametrize("constraint", ["CQF", "PerLayer", "PerFilter", "Free"])
    def test_rejects_legacy_constraint_names(
        self, constraint: LegacyConstraintLike
    ) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint=constraint)  # ty: ignore[invalid-argument-type]

    def test_rejects_unknown_constraint(self) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint="unknown")  # ty: ignore[invalid-argument-type]

    def test_numpy_integer_kernel_size(self) -> None:
        model = Despawn(kernel_init=np.int64(8))

        assert model.kernel_store[0].shape == (1, 1, 8, 1)

    def test_rejects_empty_kernel(self) -> None:
        with pytest.raises(ValueError, match="at least one coefficient"):
            Despawn(kernel_init=[])

    @pytest.mark.parametrize("kernel_size", [0, -1])
    def test_rejects_invalid_kernel_size(self, kernel_size: int) -> None:
        with pytest.raises(ValueError, match="kernel size must be a positive integer"):
            Despawn(kernel_init=kernel_size)

    def test_loss_coeff_none_returns_one_zero_per_signal(self) -> None:
        model = Despawn(loss_coeff=None)
        _, coeff_loss = model(torch.randn(2, 3, 8))

        assert coeff_loss.shape == (2, 3)
        assert torch.equal(coeff_loss, torch.zeros_like(coeff_loss))
