# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring
------         of High-Frequency Time Series (DeSpaWN) — PyTorch port

Description:
--------------
Model builder for the DeSpaWN network in PyTorch.
Ported from the original TensorFlow 2.1 implementation by Dr. Gabriel Michau.

Please cite the corresponding paper:
          Michau, G., Frusque, G., & Fink, O. (2022).
          Fully learnable deep wavelet transform for unsupervised monitoring of
          high-frequency time series.
          Proceedings of the National Academy of Sciences, 119(8).

Original author: Dr. Gabriel Michau,
                 Chair of Intelligent Maintenance Systems, ETH Zürich

Porting notes
-------------
* createDeSpaWN() is replaced by the DeSpaWN nn.Module class.
  The two-model pattern of the TF version (model1 for training, model2 for
  inspection) is collapsed into a single forward() with a return_coeffs flag.
* inputSize is accepted for API parity but has no effect: PyTorch shapes are
  always dynamic.
* Input/output tensors use PyTorch's NCHW layout (N, 1, T, 1).
  The single reshape from TF's NHWC lives in the demo script.
* Kernel sharing across levels/filter-banks is reproduced exactly:
  CQF        — one kernel shared across all levels and all four filter banks
  PerLayer   — one kernel per level, shared across all four filter banks
  PerFilter  — separate G/H kernels per level; synthesis tied to analysis
  Free       — all four filter banks independent per level
  All unique Kernel modules are collected in self.kern_store so PyTorch
  registers (and optimises) their parameters correctly.
"""

import torch
import torch.nn as nn
from lib import despawnLayers_torch as lay


class DeSpaWN(nn.Module):
    """
    Deep Sparse Wavelet Network (DeSpaWN).

    An encoder-decoder network whose analysis and synthesis filters are
    learnable wavelets.  Hard-thresholding layers sparsify the wavelet
    coefficients at each decomposition level, coupling reconstruction
    quality with a sparsity regulariser.

    Parameters
    ----------
    inputSize : int or None, optional
        Accepted for API parity with the TF version; has no effect.
        The default is None.
    kernelInit : int or array-like, optional
        Kernel initialisation.  If int: size of random-normal kernel.
        If array-like: used directly as initial kernel values (e.g. db-4).
        The default is 8.
    kernTrainable : bool, optional
        Whether wavelet kernels are trainable.  Set to False to compare
        against a fixed (classical) wavelet decomposition.
        The default is True.
    level : int, optional
        Number of decomposition levels.  Ideally floor(log2(T)).
        The default is 1.
    lossCoeff : str or None, optional
        Sparsity loss to compute on the wavelet coefficients.
        'l1'  — mean absolute value of all coefficients (returned as a
                scalar tensor that the training loop scales and adds to the
                reconstruction loss).
        None  — no sparsity loss (zeros tensor returned).
        The default is 'l1'.
    kernelsConstraint : str, optional
        Which DeSpaWN variant to build (see paper Section 4.4 Ablation Study):
        'CQF'       — single shared kernel for all levels and filter banks
        'PerLayer'  — one kernel per level, shared across filter banks
        'PerFilter' — separate G and H kernels per level; synthesis tied
        'Free'      — all four filter banks independent per level
        The default is 'CQF'.
    initHT : float, optional
        Initial value of hard-thresholding parameters thrP and thrN.
        The default is 1.0.
    trainHT : bool, optional
        Whether threshold parameters are trainable.
        The default is True.
    """

    def __init__(
        self,
        inputSize=None,         # kept for API parity; unused in PyTorch
        kernelInit=8,
        kernTrainable=True,
        level=1,
        lossCoeff='l1',
        kernelsConstraint='CQF',
        initHT=1.0,
        trainHT=True,
    ):
        super().__init__()

        self.level     = level
        self.lossCoeff = lossCoeff

        # ------------------------------------------------------------------
        # Kernel modules
        # ------------------------------------------------------------------
        # Four kernel lists of length `level` are maintained:
        #   _kG  — low-pass  analysis  (forward, decomposition)
        #   _kH  — high-pass analysis  (derived from _kG via QMF)
        #   _kGT — low-pass  synthesis (reconstruction)
        #   _kHT — high-pass synthesis (derived from _kGT via QMF)
        #
        # Depending on kernelsConstraint, lists may share the same Kernel
        # objects (shared parameters), exactly mirroring the TF code.
        # All *unique* Kernel instances are stored in self.kern_store so
        # PyTorch registers and optimises their parameters correctly.

        def make(n):
            return [lay.Kernel(kernelInit, kernTrainable) for _ in range(n)]

        if kernelsConstraint == 'CQF':
            # One kernel instance shared by every level and every filter bank
            kern = lay.Kernel(kernelInit, kernTrainable)
            self._kG  = [kern] * level
            self._kH  = [kern] * level
            self._kGT = [kern] * level
            self._kHT = [kern] * level
            self.kern_store = nn.ModuleList([kern])

        elif kernelsConstraint == 'PerLayer':
            # One kernel per level, shared across all four filter banks
            kerns = make(level)
            self._kG  = kerns
            self._kH  = kerns
            self._kGT = kerns
            self._kHT = kerns
            self.kern_store = nn.ModuleList(kerns)

        elif kernelsConstraint == 'PerFilter':
            # Separate G and H kernels per level; synthesis tied to analysis
            kerns_G = make(level)
            kerns_H = make(level)
            self._kG  = kerns_G
            self._kH  = kerns_H
            self._kGT = kerns_G        # synthesis G  = analysis G
            self._kHT = kerns_H        # synthesis H  = analysis H
            self.kern_store = nn.ModuleList(kerns_G + kerns_H)

        elif kernelsConstraint == 'Free':
            # All four filter banks are fully independent per level
            kerns_G  = make(level)
            kerns_H  = make(level)
            kerns_GT = make(level)
            kerns_HT = make(level)
            self._kG  = kerns_G
            self._kH  = kerns_H
            self._kGT = kerns_GT
            self._kHT = kerns_HT
            self.kern_store = nn.ModuleList(
                kerns_G + kerns_H + kerns_GT + kerns_HT
            )

        else:
            raise ValueError(
                f"Unknown kernelsConstraint '{kernelsConstraint}'. "
                "Choose from: 'CQF', 'PerLayer', 'PerFilter', 'Free'."
            )

        # ------------------------------------------------------------------
        # Stateless filter layers  (no parameters; one instance reused)
        # ------------------------------------------------------------------
        self.lp_wave  = lay.LowPassWave()
        self.hp_wave  = lay.HighPassWave()
        self.lp_trans = lay.LowPassTrans()
        self.hp_trans = lay.HighPassTrans()

        # ------------------------------------------------------------------
        # Hard-thresholding layers
        # One per decomposition level (for detail coefficients) + one for
        # the final approximation — exactly as in the TF code.
        # ------------------------------------------------------------------
        self.ht_details = nn.ModuleList([
            lay.HardThresholdAssym(init=initHT, trainBias=trainHT)
            for _ in range(level)
        ])
        self.ht_approx = lay.HardThresholdAssym(init=initHT, trainBias=trainHT)

    # ----------------------------------------------------------------------

    def forward(self, x, return_coeffs=False):
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
                Sparsity loss term (mean |coeff|) or zeros if lossCoeff=None.
                Scaled and added to the reconstruction loss in the training loop.
        When return_coeffs=True:
            gint     : torch.Tensor — approximation coefficients (deepest level).
            hl_rev   : list of torch.Tensor — detail coefficients ordered
                       finest-to-coarsest (reversed vs. decomposition order,
                       matching TF model2's hl[::-1] output).
        """
        g = x

        hl      = []    # detail coefficients, appended finest → coarsest order
        inSizel = []    # pre-downsampling shapes, needed for reconstruction

        # ------------------------------------------------------------------
        # Decomposition
        # ------------------------------------------------------------------
        for lev in range(self.level):
            inSizel.append(g.shape)             # save shape before downsampling

            kG = self._kG[lev]()                # low-pass  analysis kernel tensor
            kH = self._kH[lev]()                # high-pass analysis kernel tensor

            # Detail coefficients: high-pass filtered + hard-thresholded
            h = self.ht_details[lev](self.hp_wave(g, kH))
            hl.append(h)

            # Approximation: low-pass filtered (downsampled)
            g = self.lp_wave(g, kG)

        # Hard-threshold the final approximation
        g    = self.ht_approx(g)
        gint = g                                # save for coefficient output / L1 loss

        # ------------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------------
        for lev in range(self.level - 1, -1, -1):
            kGT = self._kGT[lev]()
            kHT = self._kHT[lev]()

            h = self.hp_trans(hl[lev], kHT, inSizel[lev])
            g = self.lp_trans(g,       kGT, inSizel[lev])
            g = g + h

        # ------------------------------------------------------------------
        # Sparsity loss term
        # ------------------------------------------------------------------
        if not self.lossCoeff:
            v_loss = torch.zeros(1, 1, 1, 1, device=x.device, dtype=x.dtype)

        elif self.lossCoeff == 'l1':
            # Concatenate all coefficients along the time axis (dim=2), then
            # compute mean absolute value — mirrors TF's:
            #   reduce_mean(abs(concat([gint] + hl, axis=1)), axis=1)
            # where axis=1 in NHWC == dim=2 in NCHW.
            all_coeffs = torch.cat([gint] + hl, dim=2)
            v_loss = torch.mean(torch.abs(all_coeffs), dim=2, keepdim=True)

        else:
            raise ValueError(
                f"Unknown lossCoeff '{self.lossCoeff}'. "
                "Choose 'l1' or None."
            )

        if return_coeffs:
            return g, gint, hl[::-1]    # detail list finest → coarsest (matches TF model2)
        return g, v_loss


# ---------------------------------------------------------------------------
# Factory function  —  mirrors the original createDeSpaWN() call signature
# ---------------------------------------------------------------------------

def createDeSpaWN(
    inputSize=None,
    kernelInit=8,
    kernTrainable=True,
    level=1,
    lossCoeff='l1',
    kernelsConstraint='CQF',
    initHT=1.0,
    trainHT=True,
):
    """
    Factory function that instantiates a DeSpaWN model.

    Drop-in replacement for the TF createDeSpaWN(), returning a single
    DeSpaWN nn.Module instead of two Keras models.  Call forward() with
    return_coeffs=False for training (≡ TF model1) and return_coeffs=True
    for inspection (≡ TF model2).

    Parameters
    ----------
    See DeSpaWN.__init__ for full parameter documentation.

    Returns
    -------
    model : DeSpaWN
    """
    return DeSpaWN(
        inputSize=inputSize,
        kernelInit=kernelInit,
        kernTrainable=kernTrainable,
        level=level,
        lossCoeff=lossCoeff,
        kernelsConstraint=kernelsConstraint,
        initHT=initHT,
        trainHT=trainHT,
    )
