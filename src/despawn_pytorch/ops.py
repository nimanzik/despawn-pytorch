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


def _compute_transpose_same_cropping(kernel_size: int, stride: int) -> int:
    """Return leading-side cropping for a transposed convolution with `padding=SAME`.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.

    Returns
    -------
    output : int
        The amount of cropping to apply to the leading side of the output.

    Warnings
    --------
    - This function is not generalised to `dilation` values other than 1.
    - This function is not intended to be called directly.
    """  # noqa: W505
    return max((kernel_size - stride) // 2, 0)


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
    transposed = F.conv_transpose2d(signal, kernel, stride=strides, padding=0)

    h_k, w_k = kernel.shape[2], kernel.shape[3]
    h_out, w_out = output_shape[2], output_shape[3]

    crop_top = _compute_transpose_same_cropping(h_k, stride=strides[0])
    crop_left = _compute_transpose_same_cropping(w_k, stride=strides[1])

    h_in, w_in = transposed.shape[2], transposed.shape[3]
    crop_bottom = max(h_in - h_out - crop_top, 0)
    crop_right = max(w_in - w_out - crop_left, 0)

    transposed = transposed[
        ..., crop_top : h_in - crop_bottom, crop_left : w_in - crop_right
    ]

    pad_bottom = max(h_out - h_in, 0)
    pad_right = max(w_out - w_in, 0)
    return F.pad(transposed, (0, pad_right, 0, pad_bottom))
