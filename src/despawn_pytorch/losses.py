from __future__ import annotations

import math
from numbers import Real

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DespawnLoss"]


class DespawnLoss(nn.Module):
    """Combined reconstruction and coefficient sparsity loss for DeSpaWN.

    The reconstruction term is the mean absolute error between the target and
    reconstructed signals. The sparsity term is the mean coefficient penalty
    returned by :class:`despawn_pytorch.Despawn`, scaled by
    ``sparsity_weight``.

    Parameters
    ----------
    sparsity_weight : float, default=1.0
        Weight applied to the coefficient sparsity term. It must be finite and
        nonnegative.
    """

    def __init__(self, sparsity_weight: float = 1.0) -> None:
        super().__init__()
        if (
            isinstance(sparsity_weight, bool)
            or not isinstance(sparsity_weight, Real)
            or not math.isfinite(sparsity_weight)
            or sparsity_weight < 0
        ):
            raise ValueError(
                "sparsity_weight must be a finite, nonnegative real number, "
                f"got {sparsity_weight}."
            )
        self.sparsity_weight = float(sparsity_weight)

    def forward(
        self, target: Tensor, reconstruction: Tensor, coefficient_penalty: Tensor
    ) -> Tensor:
        """Calculate the scalar training loss."""
        reconstruction_loss = F.l1_loss(reconstruction, target)
        sparsity_loss = self.sparsity_weight * coefficient_penalty.mean()
        return reconstruction_loss + sparsity_loss
