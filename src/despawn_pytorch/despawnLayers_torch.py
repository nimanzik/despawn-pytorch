# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring
------         of High-Frequency Time Series (DeSpaWN) — PyTorch port

Description:
--------------
Custom nn.Module layers used to build the DeSpaWN network in PyTorch.
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
* TF uses NHWC layout (batch, time, 1, 1); PyTorch uses NCHW (batch, 1, time, 1).
  The single reshape lives in the demo script; every layer here assumes NCHW.
* Kernels are stored as (out_ch, in_ch, kH, kW) = (1, 1, k, 1), matching
  PyTorch's conv2d convention (TF uses kH, kW, in_ch, out_ch).
* Analysis filters use symmetric padding p = (k-1)//2, giving output
  size floor((T + 2p - k) / 2) + 1 = ceil(T/2) for even kernel sizes.
* Synthesis filters compute output_padding dynamically from the saved
  pre-downsampling shape so reconstruction size always matches exactly.
  For the standard db-4 kernel (k=8) output_padding is always 0 or 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class Kernel(nn.Module):
    """
    Learnable (or fixed) wavelet kernel parameter.

    The input argument to forward() is accepted but intentionally ignored —
    it mirrors the TF Layer.call() signature where the signal was passed only
    to trigger build().  In the PyTorch model, call kern() with no argument
    to retrieve the kernel tensor.

    Parameters
    ----------
    kernelInit : int or array-like
        If int  : kernel size; weights initialised from N(0,1).
        If array: used directly as the initial kernel values.
    trainKern : bool
        Whether the kernel parameter requires gradients.
    """

    def __init__(self, kernelInit=8, trainKern=True):
        super().__init__()
        if isinstance(kernelInit, int):
            data = torch.randn(1, 1, kernelInit, 1)
        else:
            arr = torch.tensor(kernelInit, dtype=torch.float32)
            data = arr.reshape(1, 1, -1, 1)          # (out_ch, in_ch, kH, kW)
        self.kernel = nn.Parameter(data, requires_grad=trainKern)

    def forward(self, x=None):                        # x is intentionally unused
        return self.kernel


# ---------------------------------------------------------------------------
# Analysis filters  (decomposition / forward wavelet transform)
# ---------------------------------------------------------------------------

class LowPassWave(nn.Module):
    """
    Low-pass analysis filter.

    Applies conv2d(signal, kernel, stride=(2,1)) with symmetric padding
    p = (k-1)//2, equivalent to TF SAME padding for even kernel sizes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, kernel):
        # x      : (N, 1, T, 1)
        # kernel : (1, 1, k, 1)
        k = kernel.shape[2]
        p = (k - 1) // 2
        return F.conv2d(x, kernel, stride=(2, 1), padding=(p, 0))


class HighPassWave(nn.Module):
    """
    High-pass analysis filter.

    Derives the high-pass kernel from the supplied low-pass kernel via
    the Quadrature Mirror Filter (QMF) relationship:

        h[n] = (-1)^n * g[K-1-n]

    i.e. flip the kernel along the time axis then multiply by the
    alternating-sign sequence [+1, -1, +1, -1, ...].
    Then applies conv2d with stride (2, 1).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, kernel):
        # kernel : (1, 1, k, 1) — low-pass kernel supplied at runtime
        k = kernel.shape[2]
        p = (k - 1) // 2
        idx    = torch.arange(k, dtype=kernel.dtype, device=kernel.device)
        signs  = ((-1.0) ** idx).reshape(1, 1, k, 1)
        hp_kernel = torch.flip(kernel, dims=[2]) * signs
        return F.conv2d(x, hp_kernel, stride=(2, 1), padding=(p, 0))


# ---------------------------------------------------------------------------
# Synthesis filters  (reconstruction / inverse wavelet transform)
# ---------------------------------------------------------------------------

class LowPassTrans(nn.Module):
    """
    Low-pass synthesis filter.

    Applies conv_transpose2d(signal, kernel, stride=(2,1)).
    output_padding is computed on the fly from the saved pre-downsampling
    shape (target_shape) so the reconstructed time dimension is exact.

    With p = (k-1)//2 and stride 2, tentative output size is:
        tentative_T = (in_T - 1) * 2 - 2*p + k
    and output_padding = target_T - tentative_T, which is always 0 or 1
    for even kernel sizes (e.g. db-4, k=8).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, kernel, target_shape):
        # x            : (N, 1, in_T, 1)   — coefficients after downsampling
        # kernel       : (1, 1, k, 1)
        # target_shape : torch.Size or tuple (N, 1, target_T, 1)
        k = kernel.shape[2]
        p = (k - 1) // 2
        target_T    = target_shape[2]
        tentative_T = (x.shape[2] - 1) * 2 - 2 * p + k
        output_padding = int(target_T - tentative_T)
        return F.conv_transpose2d(
            x, kernel,
            stride=(2, 1), padding=(p, 0),
            output_padding=(output_padding, 0)
        )


class HighPassTrans(nn.Module):
    """
    High-pass synthesis filter.

    Applies the same QMF relationship as HighPassWave to derive the
    high-pass kernel, then applies conv_transpose2d with stride (2, 1).
    output_padding is computed from target_shape, identical logic to
    LowPassTrans.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, kernel, target_shape):
        k = kernel.shape[2]
        p = (k - 1) // 2
        idx       = torch.arange(k, dtype=kernel.dtype, device=kernel.device)
        signs     = ((-1.0) ** idx).reshape(1, 1, k, 1)
        hp_kernel = torch.flip(kernel, dims=[2]) * signs
        target_T    = target_shape[2]
        tentative_T = (x.shape[2] - 1) * 2 - 2 * p + k
        output_padding = int(target_T - tentative_T)
        return F.conv_transpose2d(
            x, hp_kernel,
            stride=(2, 1), padding=(p, 0),
            output_padding=(output_padding, 0)
        )


# ---------------------------------------------------------------------------
# Hard-thresholding
# ---------------------------------------------------------------------------

class HardThresholdAssym(nn.Module):
    """
    Learnable asymmetric hard-thresholding layer.

    Multiplies the input element-wise by a smooth binary mask built from
    two sigmoid functions (one for positive values, one for negative):

        out = x * [ σ(10*(x − thrP)) + σ(−10*(x + thrN)) ]

    This approximates hard-thresholding: values inside the dead-band
    (−thrN, +thrP) are suppressed toward zero; values outside are kept.
    Both thresholds are independent learnable scalar parameters.

    Parameters
    ----------
    init : float
        Initial value for both thrP and thrN.
    trainBias : bool
        Whether the threshold parameters require gradients.
    """

    def __init__(self, init=1.0, trainBias=True):
        super().__init__()
        val = float(init)
        self.thrP = nn.Parameter(torch.full((1, 1, 1, 1), val), requires_grad=trainBias)
        self.thrN = nn.Parameter(torch.full((1, 1, 1, 1), val), requires_grad=trainBias)

    def forward(self, x):
        return x * (
            torch.sigmoid(10.0 * (x - self.thrP)) +
            torch.sigmoid(-10.0 * (x + self.thrN))
        )
