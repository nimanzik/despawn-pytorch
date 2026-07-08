from enum import StrEnum
from typing import Literal, overload

import torch
import torch.nn as nn
from torch import Tensor

from .layers import (
    HardThresholdAssym,
    HighPassTrans,
    HighPassWave,
    LowPassTrans,
    LowPassWave,
)


class KernelsConstraint(StrEnum):
    CQF = "cqf"
    PER_LAYER = "per_layer"
    PER_FILTER = "per_filter"
    FREE = "free"


def _create_kernel(kernel_init=8, learnable=True):
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


class DeSpaWN(nn.Module):
    """Deep Sparse Wavelet Network (DeSpaWN).

    An encoder-decoder network whose analysis and synthesis filters are
    learnable wavelets. Hard-thresholding layers sparsify the wavelet
    coefficients at each decomposition level, coupling reconstruction
    quality with a sparsity regulariser.
    """

    def __init__(
        self,
        kernel_init=8,
        kern_trainable=True,
        level=1,
        loss_coeff="l1",
        kernels_constraint="cqf",
        init_ht=1.0,
        train_ht=True,
    ):
        super().__init__()

        self.level = level
        self.loss_coeff = loss_coeff

        # ------------------------------------------------------------------
        # Kernel parameters
        # ------------------------------------------------------------------
        # Four kernel lists of length `level` are maintained:
        #   _kG  — low-pass  analysis  (forward, decomposition)
        #   _kH  — high-pass analysis  (derived from _kG via QMF)
        #   _kGT — low-pass  synthesis (reconstruction)
        #   _kHT — high-pass synthesis (derived from _kGT via QMF)
        #
        # Depending on kernels_constraint, lists may share the same direct
        # nn.Parameter objects, exactly mirroring the TF code.
        # All *unique* parameters are stored in self.kern_store so PyTorch
        # registers and optimises them correctly.

        try:
            kernels_constraint = KernelsConstraint(kernels_constraint)
        except ValueError as e:
            valid = ", ".join(constraint.value for constraint in KernelsConstraint)
            raise ValueError(
                f"kernels_constraint must be one of {valid}, got {kernels_constraint}"
            ) from e

        def get_kernel_list(n: int) -> list[nn.Parameter]:
            """Create a list of `n` learnable convolution kernel parameters."""
            return [_create_kernel(kernel_init, kern_trainable) for _ in range(n)]

        if kernels_constraint == KernelsConstraint.CQF:
            # One kernel parameter shared by every level and every filter bank
            kern = _create_kernel(kernel_init, kern_trainable)
            self._kG = [kern] * level
            self._kH = [kern] * level
            self._kGT = [kern] * level
            self._kHT = [kern] * level
            self.kern_store = nn.ParameterList([kern])

        elif kernels_constraint == KernelsConstraint.PER_LAYER:
            # One kernel per level, shared across all four filter banks
            kerns = get_kernel_list(level)
            self._kG = kerns
            self._kH = kerns
            self._kGT = kerns
            self._kHT = kerns
            self.kern_store = nn.ParameterList(kerns)

        elif kernels_constraint == KernelsConstraint.PER_FILTER:
            # Separate G and H kernels per level; synthesis tied to analysis
            kerns_G = get_kernel_list(level)
            kerns_H = get_kernel_list(level)
            self._kG = kerns_G
            self._kH = kerns_H
            self._kGT = kerns_G  # synthesis G  = analysis G
            self._kHT = kerns_H  # synthesis H  = analysis H
            self.kern_store = nn.ParameterList(kerns_G + kerns_H)

        elif kernels_constraint == KernelsConstraint.FREE:
            # All four filter banks are fully independent per level
            kerns_G = get_kernel_list(level)
            kerns_H = get_kernel_list(level)
            kerns_GT = get_kernel_list(level)
            kerns_HT = get_kernel_list(level)
            self._kG = kerns_G
            self._kH = kerns_H
            self._kGT = kerns_GT
            self._kHT = kerns_HT
            self.kern_store = nn.ParameterList(kerns_G + kerns_H + kerns_GT + kerns_HT)

        # ------------------------------------------------------------------
        # Stateless filter layers  (no parameters; one instance reused)
        # ------------------------------------------------------------------
        self.lp_wave = LowPassWave()
        self.hp_wave = HighPassWave()
        self.lp_trans = LowPassTrans()
        self.hp_trans = HighPassTrans()

        # ------------------------------------------------------------------
        # Hard-thresholding layers
        # One per decomposition level (for detail coefficients) + one for
        # the final approximation — exactly as in the TF code.
        # ------------------------------------------------------------------
        self.ht_details = nn.ModuleList(
            [
                HardThresholdAssym(init_value=init_ht, learnable=train_ht)
                for _ in range(level)
            ]
        )
        self.ht_approx = HardThresholdAssym(init_value=init_ht, learnable=train_ht)

    # ----------------------------------------------------------------------

    @overload
    def forward(
        self, x: Tensor, return_coeffs: Literal[False] = False
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def forward(
        self, x: Tensor, return_coeffs: Literal[True]
    ) -> tuple[Tensor, Tensor, list[Tensor]]: ...

    @overload
    def forward(
        self, x: Tensor, return_coeffs: bool
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[Tensor]]: ...

    def forward(
        self, x: Tensor, return_coeffs: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[Tensor]]:
        """
        Forward pass: decomposition → hard-thresholding → reconstruction.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 1, T, 1)
            Input signal in PyTorch NCHW layout.
        return_coeffs : bool, optional
            False (default) — returns (reconstructed, loss_term).
                Equivalent to TF model1.  Use for training.
            True  — returns (reconstructed, approx_coeffs, detail_list).
                Equivalent to TF model2.  Use for inspection / plotting.

        Returns
        -------
        g : torch.Tensor, shape (N, 1, T, 1)
            Reconstructed signal.
        When return_coeffs=False:
            v_loss : torch.Tensor, shape (N, 1, 1, 1)
                Sparsity loss term (mean |coeff|) or zeros if loss_coeff=None.
                Scaled and added to the reconstruction loss in the training
                loop.
        When return_coeffs=True:
            gint     : torch.Tensor — approximation coefficients (deepest
            level).
            hl_rev   : list of torch.Tensor — detail coefficients ordered
                       coarsest-to-finest, matching TF model2's hl[::-1]
                       output.
        """
        g = x

        hl = []  # detail coefficients, appended finest → coarsest order
        inSizel = []  # pre-downsampling shapes, needed for reconstruction

        # ------------------------------------------------------------------
        # Decomposition
        # ------------------------------------------------------------------
        for lev in range(self.level):
            inSizel.append(g.shape)  # save shape before downsampling

            kG = self._kG[lev]  # low-pass  analysis kernel tensor
            kH = self._kH[lev]  # high-pass analysis kernel tensor

            # Detail coefficients: high-pass filtered + hard-thresholded
            h = self.ht_details[lev](self.hp_wave(g, kH))
            hl.append(h)

            # Approximation: low-pass filtered (downsampled)
            g = self.lp_wave(g, kG)

        # Hard-threshold the final approximation
        g = self.ht_approx(g)
        gint = g  # save for coefficient output / L1 loss

        # ------------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------------
        for lev in range(self.level - 1, -1, -1):
            kGT = self._kGT[lev]
            kHT = self._kHT[lev]

            h = self.hp_trans(hl[lev], kHT, inSizel[lev])
            g = self.lp_trans(g, kGT, inSizel[lev])
            g = g + h

        # ------------------------------------------------------------------
        # Sparsity loss term
        # ------------------------------------------------------------------
        if not self.loss_coeff:
            v_loss = torch.zeros(1, 1, 1, 1, device=x.device, dtype=x.dtype)

        elif self.loss_coeff == "l1":
            # Concatenate all coefficients along the time axis (dim=2), then
            # compute mean absolute value — mirrors TF's:
            #   reduce_mean(abs(concat([gint] + hl, axis=1)), axis=1)
            # where axis=1 in NHWC == dim=2 in NCHW.
            all_coeffs = torch.cat([gint] + hl, dim=2)
            v_loss = torch.mean(torch.abs(all_coeffs), dim=2, keepdim=True)

        else:
            raise ValueError(
                f"Unknown loss_coeff '{self.loss_coeff}'. Choose 'l1' or None."
            )

        if return_coeffs:
            # detail list coarsest → finest (matches TF model2)
            return (g, gint, hl[::-1])
        return g, v_loss
