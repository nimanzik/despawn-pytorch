from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from einops import rearrange
from numpy.typing import NDArray

from despawn_pytorch.layers import (
    HardThresholdAssym,
    HighPassTrans,
    HighPassWave,
    LowPassTrans,
    LowPassWave,
)

pytest.importorskip("tensorflow")
from .legacy.despawnLayers import HardThresholdAssym as LegacyHardThresholdAssym
from .legacy.despawnLayers import HighPassTrans as LegacyHighPassTrans
from .legacy.despawnLayers import HighPassWave as LegacyHighPassWave
from .legacy.despawnLayers import LowPassTrans as LegacyLowPassTrans
from .legacy.despawnLayers import LowPassWave as LegacyLowPassWave

pytestmark = pytest.mark.legacy_tf

type TFLayer = Callable[[], Any]
type TorchLayer = Callable[[], Any]
type Signal = NDArray[np.float32]
type Kernel = NDArray[np.float32]
type Output = NDArray[np.float32]
type OutputShape = tuple[int, int, int, int]


def tf_wave(layer_class: TFLayer, signal_nchw: Signal, kernel_oihw: Kernel) -> Output:
    signal_nhwc = rearrange(signal_nchw, "N iC iH iW -> N iH iW iC")
    kernel_hwio = rearrange(kernel_oihw, "oC iC kH kW -> kH kW iC oC")
    layer = layer_class()
    output_nhwc = layer([signal_nhwc, kernel_hwio]).numpy()
    return rearrange(output_nhwc, "N H W C -> N C H W")


def torch_wave(
    layer_class: TorchLayer, signal_nchw: Signal, kernel_oihw: Kernel
) -> Output:
    layer = layer_class()
    return (
        layer(torch.from_numpy(signal_nchw), torch.from_numpy(kernel_oihw))
        .detach()
        .numpy()
    )


def tf_trans(
    layer_class: TFLayer,
    signal_nchw: Signal,
    kernel_iohw: Kernel,
    output_shape_nchw: OutputShape,
) -> Output:
    """Call a TensorFlow transpose wave layer using PyTorch-shaped inputs."""
    signal_nhwc = rearrange(signal_nchw, "N iC iH iW -> N iH iW iC")
    kernel_hwio = rearrange(kernel_iohw, "iC oC kH kW -> kH kW oC iC")
    output_shape_nhwc = np.array(
        [
            output_shape_nchw[0],
            output_shape_nchw[2],
            output_shape_nchw[3],
            output_shape_nchw[1],
        ],
        dtype="int32",
    )
    layer = layer_class()
    output_nhwc = layer([signal_nhwc, kernel_hwio, output_shape_nhwc]).numpy()
    return rearrange(output_nhwc, "N H W C -> N C H W")


def torch_trans(
    layer_class: TorchLayer,
    signal_nchw: Signal,
    kernel_iohw: Kernel,
    output_shape_nchw: OutputShape,
) -> Output:
    layer = layer_class()
    return (
        layer(
            torch.from_numpy(signal_nchw),
            torch.from_numpy(kernel_iohw),
            output_shape_nchw,
        )
        .detach()
        .numpy()
    )


def tf_threshold(signal_nchw: Signal, init: float | int = 1.0) -> Output:
    signal_nhwc = rearrange(signal_nchw, "N C H W -> N H W C")
    layer = LegacyHardThresholdAssym(init=init)
    output_nhwc = layer(signal_nhwc).numpy()
    return rearrange(output_nhwc, "N H W C -> N C H W")


def torch_threshold(signal_nchw: Signal, init_value: float | int = 1.0) -> Output:
    layer = HardThresholdAssym(init_value=init_value)
    return layer(torch.from_numpy(signal_nchw)).detach().numpy()


def seed_from_request(request: pytest.FixtureRequest, namespace: str) -> int:
    cls_name = request.cls.__name__ or request.node.nodeid
    seed_key = f"{cls_name}:{namespace}"
    return int.from_bytes(hashlib.blake2s(seed_key.encode(), digest_size=4).digest())


@pytest.fixture
def random_wave_input(request: pytest.FixtureRequest) -> tuple[Signal, Kernel]:
    rng = np.random.default_rng(seed_from_request(request, "wave"))
    signal = rng.normal(size=(2, 1, 9, 3)).astype("float32")
    kernel = rng.normal(size=(1, 1, 4, 1)).astype("float32")
    return signal, kernel


@pytest.fixture
def range_wave_input() -> tuple[Signal, Kernel]:
    signal = np.arange(1, 21, dtype="float32").reshape(1, 1, -1, 1)
    kernel = np.array([1, -2, 1, -2, 1], dtype="float32").reshape(1, 1, -1, 1)
    return signal, kernel


@pytest.fixture
def random_trans_input(
    request: pytest.FixtureRequest,
) -> tuple[Signal, Kernel, OutputShape]:
    rng = np.random.default_rng(seed_from_request(request, "trans"))
    signal = rng.normal(size=(2, 1, 5, 3)).astype("float32")
    kernel = rng.normal(size=(1, 1, 4, 1)).astype("float32")
    output_shape = (2, 1, 9, 3)
    return signal, kernel, output_shape


@pytest.fixture
def range_trans_input() -> tuple[Signal, Kernel, OutputShape]:
    signal = np.arange(1, 6, dtype="float32").reshape(1, 1, -1, 1)
    kernel = np.array([1, -2, 1, -2, 1], dtype="float32").reshape(1, 1, -1, 1)
    output_shape = (1, 1, 10, 1)
    return signal, kernel, output_shape


class TestLowPassWave:
    def test_random_input(self, random_wave_input: tuple[Signal, Kernel]) -> None:
        signal, kernel = random_wave_input

        torch_output = torch_wave(LowPassWave, signal, kernel)
        tf_output = tf_wave(LegacyLowPassWave, signal, kernel)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_range_example(self, range_wave_input: tuple[Signal, Kernel]) -> None:
        signal, kernel = range_wave_input

        torch_output = torch_wave(LowPassWave, signal, kernel)
        tf_output = tf_wave(LegacyLowPassWave, signal, kernel)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)


class TestHighPassWave:
    def test_random_input(self, random_wave_input: tuple[Signal, Kernel]) -> None:
        signal, kernel = random_wave_input

        torch_output = torch_wave(HighPassWave, signal, kernel)
        tf_output = tf_wave(LegacyHighPassWave, signal, kernel)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_range_example(self, range_wave_input: tuple[Signal, Kernel]) -> None:
        signal, kernel = range_wave_input

        torch_output = torch_wave(HighPassWave, signal, kernel)
        tf_output = tf_wave(LegacyHighPassWave, signal, kernel)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)


class TestLowPassTrans:
    def test_random_input(
        self, random_trans_input: tuple[Signal, Kernel, OutputShape]
    ) -> None:
        signal, kernel, output_shape = random_trans_input

        torch_output = torch_trans(LowPassTrans, signal, kernel, output_shape)
        tf_output = tf_trans(LegacyLowPassTrans, signal, kernel, output_shape)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_range_example(
        self, range_trans_input: tuple[Signal, Kernel, OutputShape]
    ) -> None:
        signal, kernel, output_shape = range_trans_input

        torch_output = torch_trans(LowPassTrans, signal, kernel, output_shape)
        tf_output = tf_trans(LegacyLowPassTrans, signal, kernel, output_shape)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)


class TestHighPassTrans:
    def test_random_input(
        self, random_trans_input: tuple[Signal, Kernel, OutputShape]
    ) -> None:
        signal, kernel, output_shape = random_trans_input

        torch_output = torch_trans(HighPassTrans, signal, kernel, output_shape)
        tf_output = tf_trans(LegacyHighPassTrans, signal, kernel, output_shape)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_range_example(
        self, range_trans_input: tuple[Signal, Kernel, OutputShape]
    ) -> None:
        signal, kernel, output_shape = range_trans_input

        torch_output = torch_trans(HighPassTrans, signal, kernel, output_shape)
        tf_output = tf_trans(LegacyHighPassTrans, signal, kernel, output_shape)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)


class TestHardThresholdAssym:
    def test_random_input(self) -> None:
        rng = np.random.default_rng(927)
        signal = rng.normal(size=(2, 1, 9, 3)).astype("float32")

        torch_output = torch_threshold(signal, init_value=0.25)
        tf_output = tf_threshold(signal, init=0.25)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_default_init_is_one(self) -> None:
        signal = np.linspace(-2, 2, 12, dtype="float32").reshape(1, 1, -1, 1)

        torch_output = torch_threshold(signal)
        tf_output = tf_threshold(signal)

        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-6, atol=1e-6)

    def test_train_bias_false_freezes_thresholds(self) -> None:
        layer = HardThresholdAssym(init_value=0.5, learnable=False)

        assert not layer.positive_threshold.requires_grad
        assert not layer.negative_threshold.requires_grad
