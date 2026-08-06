import pytest
import torch

from despawn_pytorch.ops import apply_conv2d, apply_conv_transpose2d


@pytest.mark.parametrize("height", [3, 4, 5, 6])
@pytest.mark.parametrize("kernel_height", [2, 3, 4, 5])
def test_conv_transpose2d_is_adjoint(height: int, kernel_height: int) -> None:
    """Check the adjoint property of conv transpose2d.

    The mathematical requirement is ``<conv(x), y> = <x, conv_transpose(y)>``,
    where `<., .>` denotes the dot product.
    """
    signal = torch.randn(2, 1, height, 3, dtype=torch.float64)
    kernel = torch.randn(1, 1, kernel_height, 1, dtype=torch.float64)

    convolved = apply_conv2d(signal, kernel, strides=(2, 1))
    coefficients = torch.randn_like(convolved)
    transposed = apply_conv_transpose2d(
        coefficients, kernel, output_shape=signal.shape, strides=(2, 1)
    )

    torch.testing.assert_close(
        (convolved * coefficients).sum(), (signal * transposed).sum()
    )
    assert transposed.shape == signal.shape


@pytest.mark.parametrize(
    ("output_shape", "match"),
    [((3, 1, 5, 3), "batch size"), ((2, 2, 5, 3), "output channels")],
)
def test_conv_transpose2d_rejects_incompatible_output_shape(
    output_shape: tuple[int, int, int, int], match: str
) -> None:
    signal = torch.randn(2, 1, 3, 3)
    kernel = torch.randn(1, 1, 3, 1)

    with pytest.raises(ValueError, match=match):
        apply_conv_transpose2d(
            signal, kernel, output_shape=output_shape, strides=(2, 1)
        )
