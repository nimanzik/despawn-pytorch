from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

from despawn_pytorch import Despawn, DespawnLoss

type ActiveConstraintLike = Literal["cqf", "per_layer", "per_filter", "free"]
type LegacyConstraintLike = Literal["CQF", "PerLayer", "PerFilter", "Free"]


class TestDespawn:
    @pytest.mark.parametrize("threshold_init", [0, 1])
    def test_integer_matches_float_threshold(self, threshold_init: int) -> None:
        integer = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1], n_levels=3, threshold_init=threshold_init
        )
        floating = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            n_levels=3,
            threshold_init=float(threshold_init),
        )
        signal = torch.linspace(-2, 2, 30).reshape(2, 15)

        reconstruction, penalty = integer(signal)
        expected_reconstruction, expected_penalty = floating(signal)

        torch.testing.assert_close(reconstruction, expected_reconstruction)
        torch.testing.assert_close(penalty, expected_penalty)

    @pytest.mark.parametrize("constraint", ["cqf", "per_layer", "per_filter", "free"])
    def test_assign_zero_kernels(self, constraint: ActiveConstraintLike) -> None:
        # Loading zero kernels with assign=True zeroes all transform outputs.
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            threshold_init=0.25,
        )
        checkpoint = {
            name: torch.zeros_like(value) if name.startswith("kernel_store.") else value
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(checkpoint, assign=True)
        signal = torch.linspace(-2, 2, 30).reshape(2, 15)

        reconstruction, penalty = model(signal)
        decomposed_reconstruction, approximation, details = model.decompose(signal)

        for output in (
            reconstruction,
            penalty,
            decomposed_reconstruction,
            approximation,
            *details,
        ):
            torch.testing.assert_close(output, torch.zeros_like(output))

    @pytest.mark.parametrize("constraint", ["cqf", "per_layer", "per_filter", "free"])
    def test_assign_kernel_training(self, constraint: ActiveConstraintLike) -> None:
        # Loaded kernels drive reconstruction and training with assign=True.
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            threshold_init=0.25,
            threshold_learnable=False,
        )
        reference = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            threshold_init=0.25,
            threshold_learnable=False,
        )
        checkpoint = {
            name: value * (0.8 + 0.02 * int(name.rsplit(".", 1)[1]))
            if name.startswith("kernel_store.")
            else value.clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(checkpoint, assign=True)
        reference.load_state_dict(checkpoint)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        signal = torch.linspace(-2, 2, 30).reshape(2, 15)
        reconstruction, penalty = model(signal)
        expected_reconstruction, expected_penalty = reference(signal)
        torch.testing.assert_close(reconstruction, expected_reconstruction)
        torch.testing.assert_close(penalty, expected_penalty)

        before = [parameter.detach().clone() for parameter in model.parameters()]
        DespawnLoss()(signal, reconstruction, penalty).backward()
        for parameter in model.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
                assert torch.count_nonzero(parameter.grad) > 0

        optimizer.step()
        for parameter, previous in zip(model.parameters(), before, strict=True):
            assert torch.equal(parameter, previous) == (not parameter.requires_grad)
        updated_reconstruction, _ = model(signal)
        assert not torch.equal(updated_reconstruction, reconstruction)

    def test_changing_initializer_does_not_change_reconstruction(self) -> None:
        initializer = np.array([0.2, -0.5, 0.7, 0.1], dtype=np.float32)
        model = Despawn(kernel_init=initializer, n_levels=3, threshold_init=0.25)
        signal = torch.linspace(-2, 2, 30).reshape(2, 15)

        with torch.no_grad():
            reconstruction_before, _ = model(signal)
            initializer += 1.0
            reconstruction_after, _ = model(signal)

        torch.testing.assert_close(reconstruction_after, reconstruction_before)

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

    @pytest.mark.parametrize(
        ("constraint", "expected_kernel_params"),
        [("cqf", 1), ("per_layer", 3), ("per_filter", 6), ("free", 12)],
    )
    def test_array_and_list_initializers_train_identically(
        self, constraint: ActiveConstraintLike, expected_kernel_params: int
    ) -> None:
        initializer = np.array([0.2, -0.5, 0.7, 0.1], dtype=np.float32)
        initializer_before = initializer.copy()
        models = [
            Despawn(
                kernel_init=kernel,
                kernels_constraint=constraint,
                n_levels=3,
                threshold_init=0.25,
            )
            for kernel in (initializer, initializer.tolist())
        ]
        signal = torch.linspace(-2, 2, 30).reshape(2, 15)

        for model in models:
            before = [parameter.detach().clone() for parameter in model.parameters()]
            assert len(before) == expected_kernel_params + 8
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            reconstruction, penalty = model(signal)
            DespawnLoss()(signal, reconstruction, penalty).backward()

            for parameter in model.parameters():
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()

            optimizer.step()
            for parameter, previous in zip(model.parameters(), before, strict=True):
                assert not torch.equal(parameter, previous)

        np.testing.assert_array_equal(initializer, initializer_before)
        for (name, parameter), (reference_name, reference) in zip(
            models[0].named_parameters(), models[1].named_parameters(), strict=True
        ):
            assert name == reference_name
            torch.testing.assert_close(parameter, reference)

    @pytest.mark.parametrize("constraint", ["cqf", "per_layer", "per_filter", "free"])
    def test_updating_one_kernel_preserves_other_kernels_and_initializer(
        self, constraint: ActiveConstraintLike
    ) -> None:
        initializer = np.array([0.2, -0.5, 0.7, 0.1], dtype=np.float32)
        initializer_before = initializer.copy()
        model = Despawn(
            kernel_init=initializer, kernels_constraint=constraint, n_levels=3
        )
        kernels = [
            parameter
            for _, parameter in model.named_parameters()
            if parameter.numel() == initializer.size
        ]
        before = [kernel.detach().clone() for kernel in kernels]

        with torch.no_grad():
            kernels[0].add_(1.0)

        assert not torch.equal(kernels[0], before[0])
        for kernel, previous in zip(kernels[1:], before[1:], strict=True):
            torch.testing.assert_close(kernel, previous)
        np.testing.assert_array_equal(initializer, initializer_before)

    @pytest.mark.parametrize("constraint", ["cqf", "per_layer", "per_filter", "free"])
    @pytest.mark.parametrize(
        ("kernel_learnable", "threshold_learnable"),
        [(False, True), (True, False), (False, False)],
    )
    def test_training_preserves_frozen_parameters(
        self,
        constraint: ActiveConstraintLike,
        kernel_learnable: bool,
        threshold_learnable: bool,
    ) -> None:
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            kernel_learnable=kernel_learnable,
            threshold_init=0.25,
            threshold_learnable=threshold_learnable,
        )
        signal = torch.linspace(-2, 2, 30).reshape(2, 15).requires_grad_()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reconstruction, penalty = model(signal)
        DespawnLoss()(signal, reconstruction, penalty).backward()

        for parameter in model.parameters():
            expected_learnable = (
                kernel_learnable if parameter.numel() == 4 else threshold_learnable
            )
            assert parameter.requires_grad == expected_learnable
            if expected_learnable:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            else:
                assert parameter.grad is None

        optimizer.step()
        for parameter, previous in zip(model.parameters(), before, strict=True):
            assert torch.equal(parameter, previous) == (not parameter.requires_grad)

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
