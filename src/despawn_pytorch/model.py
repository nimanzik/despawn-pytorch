from __future__ import annotations

from enum import StrEnum
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from .layers import (
    HardThreshold,
    HighPassTrans,
    HighPassWave,
    LowPassTrans,
    LowPassWave,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Despawn", "get_num_levels"]


class _SupportsArray(Protocol):
    def __array__(self) -> Any: ...


class _KernelsConstraint(StrEnum):
    CQF = "cqf"
    PER_LAYER = "per_layer"
    PER_FILTER = "per_filter"
    FREE = "free"


def _create_kernel(
    kernel_init: int | Sequence[float] | _SupportsArray = 8, learnable: bool = True
) -> nn.Parameter:
    """Create a convolution kernel parameter.

    Parameters
    ----------
    kernel_init : int or array-like, default=8
        If this is an integer, it is used as the kernel length. The kernel
        values are drawn from a normal distribution with mean 0 and standard
        deviation 0.05. If this is not an integer, it is converted to a tensor
        and flattened before it is reshaped for convolution.
    learnable : bool, default=True
        If True, the returned parameter is updated during training. If False,
        the returned parameter is fixed.

    Returns
    -------
    kernel : torch.nn.Parameter
        A parameter with shape ``(1, 1, kernel_size, 1)``. This shape is used
        by the wavelet convolution layers.
    """
    dtype = torch.get_default_dtype()
    if isinstance(kernel_init, int):
        kernel = torch.empty(kernel_init, dtype=dtype).normal_(mean=0, std=0.05)
    else:
        kernel = torch.as_tensor(kernel_init, dtype=dtype).flatten()
    return nn.Parameter(kernel.reshape(1, 1, -1, 1), requires_grad=learnable)


def get_num_levels(signal_length: int) -> int:
    """Return the number of decomposition levels for a signal length.

    Calculate the result as ``floor(log2(signal_length))``.

    Parameters
    ----------
    signal_length : int
        Number of samples in the signal.

    Returns
    -------
    n : int
        Number of decomposition levels.
    """
    if (
        isinstance(signal_length, bool)
        or not isinstance(signal_length, Integral)
        or signal_length < 2
    ):
        raise ValueError(
            f"signal_length must be an integer of at least 2, got {signal_length}."
        )
    return int(signal_length).bit_length() - 1


class Despawn(nn.Module):
    """Deep Sparse Wavelet Network (DeSpaWN).

    An encoder-decoder network whose analysis and synthesis filters are
    learnable wavelets. Hard-thresholding layers sparsify the wavelet
    coefficients at each decomposition level, coupling reconstruction
    quality with a sparsity regulariser.
    """

    def __init__(
        self,
        *,
        kernel_init: int | Sequence[float] | _SupportsArray = 8,
        kernel_learnable: bool = True,
        kernels_constraint: Literal["cqf", "per_layer", "per_filter", "free"] = "cqf",
        n_levels: int = 1,
        loss_coeff: Literal["l1", None] = "l1",
        threshold_init: float = 1.0,
        threshold_learnable: bool = True,
    ):
        super().__init__()

        if not isinstance(n_levels, int) or n_levels < 1:
            raise ValueError(f"level must be a positive integer, got {n_levels}.")
        self.n_levels = n_levels

        if loss_coeff not in {"l1", None}:
            raise ValueError(f"loss_coeff must be 'l1' or None, got {loss_coeff}.")
        self.loss_coeff = loss_coeff

        # ----- Kernel parameters ------
        try:
            constraint = _KernelsConstraint(kernels_constraint)
        except ValueError as e:
            valid = ", ".join(member.value for member in _KernelsConstraint)
            raise ValueError(
                f"kernels_constraint must be one of {valid}, got {kernels_constraint}"
            ) from e

        def get_kernel_list(n: int) -> list[nn.Parameter]:
            """Create a list of `n` learnable convolution kernel parameters."""
            return [_create_kernel(kernel_init, kernel_learnable) for _ in range(n)]

        if constraint == _KernelsConstraint.CQF:
            # Share one kernel across all levels and filter banks.
            kern = _create_kernel(kernel_init, kernel_learnable)
            self._kG = [kern] * n_levels
            self._kH = [kern] * n_levels
            self._kGT = [kern] * n_levels
            self._kHT = [kern] * n_levels
            self.kernel_store = nn.ParameterList([kern])

        elif constraint == _KernelsConstraint.PER_LAYER:
            # Share one kernel across all filter banks at each level.
            kerns = get_kernel_list(n_levels)
            self._kG = kerns
            self._kH = kerns
            self._kGT = kerns
            self._kHT = kerns
            self.kernel_store = nn.ParameterList(kerns)

        elif constraint == _KernelsConstraint.PER_FILTER:
            # Use separate analysis kernels, with synthesis tied to analysis.
            kerns_G = get_kernel_list(n_levels)
            kerns_H = get_kernel_list(n_levels)
            self._kG = kerns_G
            self._kH = kerns_H
            self._kGT = kerns_G  # synthesis G  = analysis G
            self._kHT = kerns_H  # synthesis H  = analysis H
            self.kernel_store = nn.ParameterList(kerns_G + kerns_H)

        elif constraint == _KernelsConstraint.FREE:
            # Use independent kernels for every filter bank at every level.
            kerns_G = get_kernel_list(n_levels)
            kerns_H = get_kernel_list(n_levels)
            kerns_GT = get_kernel_list(n_levels)
            kerns_HT = get_kernel_list(n_levels)
            self._kG = kerns_G
            self._kH = kerns_H
            self._kGT = kerns_GT
            self._kHT = kerns_HT
            self.kernel_store = nn.ParameterList(
                kerns_G + kerns_H + kerns_GT + kerns_HT
            )

        # Stateless filters are reused across levels.
        self.lp_wave = LowPassWave()
        self.hp_wave = HighPassWave()
        self.lp_trans = LowPassTrans()
        self.hp_trans = HighPassTrans()

        # Threshold detail coefficients at each level and the final
        # approximation.
        self.ht_details = nn.ModuleList(
            [
                HardThreshold(init_value=threshold_init, learnable=threshold_learnable)
                for _ in range(n_levels)
            ]
        )
        self.ht_approx = HardThreshold(
            init_value=threshold_init, learnable=threshold_learnable
        )

    def _transform(self, x: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Decompose, threshold, and reconstruct an input tensor."""
        g = x

        hl = []  # detail coefficients, finest to coarsest
        inSizel = []  # shapes before downsampling, needed for reconstruction

        # ----- Decomposition -----
        for decomp_level in range(self.n_levels):
            inSizel.append(g.shape)  # save shape before downsampling

            kG = self._kG[decomp_level]  # low-pass analysis kernel tensor
            kH = self._kH[decomp_level]  # high-pass analysis kernel tensor

            # Detail coefficients: high-pass filtered + hard-thresholded
            h = self.ht_details[decomp_level](self.hp_wave(g, kH))
            hl.append(h)

            # Approximation: low-pass filtered (downsampled)
            g = self.lp_wave(g, kG)

        # Hard-threshold the final approximation
        g = self.ht_approx(g)
        gint = g  # save for coefficient output / L1 loss

        # ----- Reconstruction -----
        for recon_level in range(self.n_levels - 1, -1, -1):
            kGT = self._kGT[recon_level]
            kHT = self._kHT[recon_level]

            h = self.hp_trans(hl[recon_level], kHT, inSizel[recon_level])
            g = self.lp_trans(g, kGT, inSizel[recon_level])
            g = g + h

        return g, gint, hl

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the reconstruction and coefficient sparsity loss."""
        reconstruction, approximation, details = self._transform(x)

        if self.loss_coeff is None:
            coeff_loss = torch.zeros(1, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            all_coeffs = torch.cat([approximation] + details, dim=2)
            coeff_loss = torch.mean(torch.abs(all_coeffs), dim=2, keepdim=True)

        return reconstruction, coeff_loss

    def decompose(self, x: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Return the reconstruction and thresholded wavelet coefficients.

        Detail coefficients are ordered from the coarsest level to the finest.
        """
        reconstruction, approximation, details = self._transform(x)
        return reconstruction, approximation, details[::-1]
