from __future__ import annotations

from math import isfinite

import torch
import torch.nn as nn
from torch import Tensor

from .ops import apply_conv2d, apply_conv_transpose2d

__all__ = [
    "HardThreshold",
    "HighPassTrans",
    "HighPassWave",
    "LowPassTrans",
    "LowPassWave",
]


class LowPassWave(nn.Module):
    """Low-pass analysis filter for signal decomposition.

    Applies a `'SAME'` padding style convolution with stride (2, 1) to the
    input signal.
    """

    def forward(self, signal: Tensor, kernel: Tensor) -> Tensor:
        return apply_conv2d(signal, kernel, strides=(2, 1))


class HighPassWave(nn.Module):
    """High-pass analysis filter for signal decomposition.

    Applies a `'SAME'` padding style convolution with stride (2, 1) to the
    input signal.

    Derives the filter kernel following the Conjugate Quadrature Filter (CQF)
    relationship.
    """

    def forward(self, signal: Tensor, kernel: Tensor) -> Tensor:
        idxs = torch.arange(kernel.shape[2], device=kernel.device, dtype=kernel.dtype)
        signs = ((-1) ** idxs).reshape(1, 1, -1, 1)
        hp_kernel = torch.flip(kernel, dims=[2]) * signs
        return apply_conv2d(signal, hp_kernel, strides=(2, 1))


class LowPassTrans(nn.Module):
    """Low-pass synthesis filter for signal reconstruction.

    Applies a `'SAME'` padding style transposed convolution with stride (2, 1)
    to the input signal.

    Derives the filter kernel following the Conjugate Quadrature Filter (CQF)
    relationship (implicitly via the adjoint property of the transposed
    convolution).
    """

    def forward(
        self,
        signal: Tensor,
        kernel: Tensor,
        output_shape: tuple[int, int, int, int] | list[int] | torch.Size,
    ) -> Tensor:
        return apply_conv_transpose2d(signal, kernel, output_shape, strides=(2, 1))


class HighPassTrans(nn.Module):
    """High-pass synthesis filter for signal reconstruction.

    Applies a `'SAME'` padding style transposed convolution with stride (2, 1)
    to the input signal.

    Derives the filter kernel following the Conjugate Quadrature Filter (CQF)
    relationship (relies on the adjoint property of the transposed
    convolution).
    """

    def forward(
        self,
        signal: Tensor,
        kernel: Tensor,
        output_shape: tuple[int, int, int, int] | list[int] | torch.Size,
    ) -> Tensor:
        idxs = torch.arange(kernel.shape[2], device=kernel.device, dtype=kernel.dtype)
        signs = ((-1) ** (idxs + 1)).reshape(1, 1, -1, 1)
        hp_kernel = torch.flip(kernel, dims=[2]) * signs
        return apply_conv_transpose2d(signal, hp_kernel, output_shape, strides=(2, 1))


class HardThreshold(nn.Module):
    """Asymmetric hard-thresholding layer.

    Approximates an asymmetric hard-thresholding, where the values within the
    dead band, ``(-b, +b)`` are suppressed towards zero and values outside are
    kept.

    Positive and negative thresholds can be learned independently as scalar
    parameters.
    """

    def __init__(
        self, init_value: float | int = 1.0, learnable: bool = True, alpha: float = 10.0
    ) -> None:
        super().__init__()
        if not isfinite(init_value) or init_value < 0:
            raise ValueError(
                f"init_value must be finite and non-negative, got {init_value}"
            )
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be finite and positive, got {alpha}")

        self.positive_threshold = nn.Parameter(
            torch.full((1, 1, 1, 1), init_value), requires_grad=learnable
        )
        self.negative_threshold = nn.Parameter(
            torch.full((1, 1, 1, 1), init_value), requires_grad=learnable
        )
        self.alpha = float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        mask = torch.sigmoid(
            self.alpha * (x - self.positive_threshold.abs())
        ) + torch.sigmoid(-self.alpha * (x + self.negative_threshold.abs()))
        return x * mask
