from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from despawn_pytorch import Despawn, DespawnLoss


class TestDespawnLoss:
    def test_manual_calculation(self) -> None:
        """Check that the combined loss matches its manual calculation."""
        target = torch.tensor([1.0, -2.0, 3.0])
        reconstruction = torch.tensor([0.5, -1.0, 2.0])
        coefficient_penalty = torch.tensor([0.25, 0.75])
        sparsity_weight = 2.0
        criterion = DespawnLoss(sparsity_weight=sparsity_weight)

        loss = criterion(target, reconstruction, coefficient_penalty)
        expected = F.l1_loss(reconstruction, target) + sparsity_weight * torch.mean(
            coefficient_penalty
        )

        torch.testing.assert_close(loss, expected)
        assert loss.ndim == 0

    def test_sparsity_weight(self) -> None:
        """Check that the sparsity weight scales only the coefficient term."""
        target = torch.tensor([1.0, 2.0])
        reconstruction = torch.tensor([0.0, 4.0])
        coefficient_penalty = torch.tensor([0.5])

        unweighted = DespawnLoss(sparsity_weight=1.0)(
            target, reconstruction, coefficient_penalty
        )
        weighted = DespawnLoss(sparsity_weight=3.0)(
            target, reconstruction, coefficient_penalty
        )

        expected_difference = 2.0 * coefficient_penalty.mean()
        torch.testing.assert_close(weighted - unweighted, expected_difference)

    def test_zero_weight(self) -> None:
        """Check that zero sparsity weight returns only reconstruction loss."""
        target = torch.tensor([1.0, 2.0])
        reconstruction = torch.tensor([0.0, 4.0])
        coefficient_penalty = torch.tensor([100.0])

        loss = DespawnLoss(sparsity_weight=0.0)(
            target, reconstruction, coefficient_penalty
        )

        torch.testing.assert_close(loss, F.l1_loss(reconstruction, target))

    def test_no_coefficient_loss(self) -> None:
        """Check that no coefficient loss uses only reconstruction loss."""
        signal = torch.randn(2, 1, 8, 1)
        model = Despawn(loss_coeff=None)
        reconstruction, coefficient_penalty = model(signal)

        loss = DespawnLoss()(signal, reconstruction, coefficient_penalty)

        torch.testing.assert_close(loss, F.l1_loss(reconstruction, signal))

    def test_gradients(self) -> None:
        """Check that gradients reach both loss inputs."""
        target = torch.tensor([1.0, 2.0])
        reconstruction = torch.tensor([0.0, 4.0], requires_grad=True)
        coefficient_penalty = torch.tensor([0.5], requires_grad=True)

        loss = DespawnLoss(sparsity_weight=2.0)(
            target, reconstruction, coefficient_penalty
        )
        loss.backward()

        assert reconstruction.grad is not None
        assert coefficient_penalty.grad is not None
        torch.testing.assert_close(coefficient_penalty.grad, torch.tensor([2.0]))

    @pytest.mark.parametrize(
        "sparsity_weight", [-1.0, float("inf"), float("-inf"), float("nan"), True]
    )
    def test_invalid_weight(self, sparsity_weight: float) -> None:
        """Check that invalid sparsity weights raise a clear error."""
        with pytest.raises(ValueError, match="sparsity_weight"):
            DespawnLoss(sparsity_weight=sparsity_weight)
