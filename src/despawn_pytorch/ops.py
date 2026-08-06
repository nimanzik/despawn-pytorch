from __future__ import annotations

import torch.nn.functional as F
from torch import Size as TorchSize
from torch import Tensor

__all__ = ["apply_conv2d", "apply_conv_transpose2d"]


def _compute_same_padding(
    input_size: int, kernel_size: int, stride: int
) -> tuple[int, int]:
    """Return padding values for a convolution with `padding='SAME'`.

    The returned tuple is ``(pad_before, pad_after)``. It pads enough so the
    output size is ``ceil(input_size / stride)``. If the total padding is odd,
    the extra value is given to the trailing side (e.g., bottom or right side).

    For more information, see [TF padding documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py#L48).

    Parameters
    ----------
    input_size : int
        Size of the input dimension.
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.

    Returns
    -------
    output : tuple[int, int]
        a 2-tuple of the padding to apply before and after the input.

    Warnings
    --------
    - This function is not generalised to `dilation` values other than 1.
    - This function is not intended to be called directly.
    """
    output_size = (input_size + stride - 1) // stride
    pad_needed = max((output_size - 1) * stride + kernel_size - input_size, 0)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before
    return pad_before, pad_after


def apply_conv2d(signal: Tensor, kernel: Tensor, strides: tuple[int, int]) -> Tensor:
    """Apply a 2-D convolution with `padding='SAME'`.

    Parameters
    ----------
    signal : torch.Tensor, shape (N, C_in, H_in, W_in)
        Input signal.
    kernel : torch.Tensor, shape (C_out, C_in, H_k, W_k)
        Convolution weights.
    strides : tuple[int, int]
        The strides of the convolution kernel along the height and width axes.

    Returns
    -------
    output : torch.Tensor, shape (N, C_out, H_out, W_out)
        Convolved output signal.
    """
    h_in, w_in = signal.shape[2], signal.shape[3]
    h_k, w_k = kernel.shape[2], kernel.shape[3]

    pad_top, pad_bottom = _compute_same_padding(h_in, h_k, stride=strides[0])
    pad_left, pad_right = _compute_same_padding(w_in, w_k, stride=strides[1])
    padded = F.pad(signal, (pad_left, pad_right, pad_top, pad_bottom))
    return F.conv2d(padded, kernel, stride=strides, padding=0)


def apply_conv_transpose2d(
    signal: Tensor,
    kernel: Tensor,
    output_shape: tuple[int, int, int, int] | list[int] | TorchSize,
    strides: tuple[int, int],
) -> Tensor:
    """Apply a 2-D transposed convolution with `padding='SAME'`.

    Parameters
    ----------
    signal : torch.Tensor, shape (N, C_in, H_in, W_in)
        Input signal.
    kernel : torch.Tensor, shape (C_in, C_out, H_k, W_k)
        Convolution weights.
    output_shape : tuple[int, int, int, int] | list[int] | torch.Size
        Desired output shape.
    strides : tuple[int, int]
        The strides of the convolution kernel along the height and width axes.

    Returns
    -------
    output : torch.Tensor, shape (N, C_out, H_out, W_out)
        Transposed convolution output.
    """
    if len(output_shape) != 4:
        raise ValueError("output_shape must contain four dimensions")

    n_out, c_out, h_out, w_out = output_shape
    if n_out != signal.shape[0]:
        raise ValueError(
            f"Requested batch size {n_out}, but input batch size is {signal.shape[0]}"
        )
    if c_out != kernel.shape[1]:
        raise ValueError(
            f"Requested {c_out} output channels, but kernel produces {kernel.shape[1]}"
        )

    expected_h_in = (h_out + strides[0] - 1) // strides[0]
    expected_w_in = (w_out + strides[1] - 1) // strides[1]
    if signal.shape[2:] != (expected_h_in, expected_w_in):
        raise ValueError(
            "Input spatial shape is inconsistent with output_shape and strides: "
            f"expected {(expected_h_in, expected_w_in)}, "
            f"got {tuple(signal.shape[2:])}"
        )

    h_k, w_k = kernel.shape[2], kernel.shape[3]
    crop_top, crop_bottom = _compute_same_padding(h_out, h_k, stride=strides[0])
    crop_left, crop_right = _compute_same_padding(w_out, w_k, stride=strides[1])

    transposed = F.conv_transpose2d(signal, kernel, stride=strides, padding=0)
    raw_h, raw_w = transposed.shape[2], transposed.shape[3]
    h_stop = raw_h - crop_bottom if crop_bottom else raw_h
    w_stop = raw_w - crop_right if crop_right else raw_w
    transposed = transposed[..., crop_top:h_stop, crop_left:w_stop]

    current_h, current_w = transposed.shape[2], transposed.shape[3]
    if current_h > h_out or current_w > w_out:
        raise RuntimeError(
            "Transpose result is larger than requested shape: "
            f"got {(current_h, current_w)}, requested {(h_out, w_out)}"
        )

    return F.pad(transposed, (0, w_out - current_w, 0, h_out - current_h))
