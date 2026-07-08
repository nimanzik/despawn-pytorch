import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import tensorflow as tf  # ty: ignore[unresolved-import]
import torch
from einops import rearrange

from despawn_pytorch.model import Despawn

from .legacy import despawnLayers as legacy_layers

legacy_lib = ModuleType("lib")
legacy_lib.despawnLayers = legacy_layers
sys.modules.setdefault("lib", legacy_lib)
from .legacy import despawnCreate as legacy_create  # noqa: E402

LEGACY_CONSTRAINT_MAP = {
    "CQF": "cqf",
    "PerLayer": "per_layer",
    "PerFilter": "per_filter",
    "Free": "free",
}


type CreateDeSpaWN = Callable[..., tuple[Any, Any]]


class _LegacyCreateMathShim:
    @staticmethod
    def reduce_mean(x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        return tf.keras.ops.mean(x, axis=axis, keepdims=keepdims)

    abs = staticmethod(tf.keras.ops.absolute)


class _LegacyCreateTFShim:
    # Keras 3 compatibility shim for tests/legacy/despawnCreate.py.
    # The legacy model was written for TensorFlow 2.1 and calls a few raw `tf`
    # functions on symbolic Keras tensors. Keras 3 disallows that, so the
    # shim only replaces those model-construction helpers with Keras
    # operations. The legacy model topology and layer implementations are
    # unchanged.

    math = _LegacyCreateMathShim
    zeros = staticmethod(tf.zeros)

    @staticmethod
    def shape(x: Any) -> Any:
        return tf.keras.layers.Lambda(lambda z: tf.shape(z), dtype="int32")(x)

    @staticmethod
    def concat(values: list[Any], axis: int) -> Any:
        return tf.keras.ops.concatenate(values, axis=axis)


@pytest.fixture
def patched_legacy_create(monkeypatch: pytest.MonkeyPatch) -> CreateDeSpaWN:
    monkeypatch.setattr(legacy_create, "tf", _LegacyCreateTFShim)
    return legacy_create.createDeSpaWN


@pytest.fixture
def model_input() -> np.ndarray:
    rng = np.random.default_rng(731)
    return rng.normal(size=(2, 1, 15, 1)).astype("float32")


class TestDespawn:
    @pytest.mark.parametrize(
        ("constraint", "expected_kernel_params"),
        [("cqf", 1), ("per_layer", 3), ("per_filter", 6), ("free", 12)],
    )
    def test_constraint_parameter_sharing(
        self, constraint: str, expected_kernel_params: int
    ) -> None:
        model = Despawn(
            kernel_init=[0.2, -0.5, 0.7, 0.1],
            kernels_constraint=constraint,
            n_levels=3,
            threshold_init=0.25,
        )

        assert len(model.kern_store) == expected_kernel_params

        x = torch.randn(2, 1, 15, 1)
        recon, coeff_loss = model(x)
        recon2, approx, details = model(x, return_coeffs=True)

        assert recon.shape == x.shape
        assert coeff_loss.shape == (2, 1, 1, 1)
        assert recon2.shape == x.shape
        assert approx.shape == (2, 1, 2, 1)
        assert [detail.shape for detail in details] == [
            torch.Size([2, 1, 2, 1]),
            torch.Size([2, 1, 4, 1]),
            torch.Size([2, 1, 8, 1]),
        ]

    @pytest.mark.parametrize("constraint", ["CQF", "PerLayer", "PerFilter", "Free"])
    def test_rejects_legacy_constraint_names(self, constraint: str) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint=constraint)

    def test_rejects_unknown_constraint(self) -> None:
        with pytest.raises(ValueError, match="kernels_constraint"):
            Despawn(kernels_constraint="unknown")

    def test_loss_coeff_none_returns_zero(self) -> None:
        model = Despawn(loss_coeff=None)
        _, coeff_loss = model(torch.randn(2, 1, 8, 1))

        assert coeff_loss.shape == (1, 1, 1, 1)
        assert torch.equal(coeff_loss, torch.zeros_like(coeff_loss))


class TestDespawnLegacyParity:
    @pytest.mark.parametrize(
        ("legacy_constraint", "torch_constraint"), LEGACY_CONSTRAINT_MAP.items()
    )
    def test_reconstruction_and_loss_match_legacy(
        self,
        patched_legacy_create: CreateDeSpaWN,
        model_input: np.ndarray,
        legacy_constraint: str,
        torch_constraint: str,
    ) -> None:
        # Test model1 in legacy/despawnCreate.py
        tf.keras.backend.clear_session()
        kernel = np.array([0.2, -0.5, 0.7, 0.1], dtype="float32")

        tf_model, _ = patched_legacy_create(
            inputSize=15,
            kernelInit=kernel,
            kernTrainable=True,
            level=3,
            lossCoeff="l1",
            kernelsConstraint=legacy_constraint,
            initHT=0.25,
            trainHT=True,
        )
        torch_model = Despawn(
            kernel_init=kernel,
            kernel_learnable=True,
            kernels_constraint=torch_constraint,
            n_levels=3,
            loss_coeff="l1",
            threshold_init=0.25,
            threshold_learnable=True,
        )

        tf_recon, tf_coeff_loss = tf_model(rearrange(model_input, "N C H W -> N H W C"))
        with torch.no_grad():
            torch_recon, torch_coeff_loss = torch_model(torch.from_numpy(model_input))

        np.testing.assert_allclose(
            torch_recon.numpy(),
            rearrange(tf_recon.numpy(), "N H W C -> N C H W"),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            torch_coeff_loss.numpy(),
            rearrange(tf_coeff_loss.numpy(), "N H W C -> N C H W"),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        ("legacy_constraint", "torch_constraint"), LEGACY_CONSTRAINT_MAP.items()
    )
    def test_coefficients_match_legacy(
        self,
        patched_legacy_create: CreateDeSpaWN,
        model_input: np.ndarray,
        legacy_constraint: str,
        torch_constraint: str,
    ) -> None:
        # Test model2 in legacy/despawnCreate.py
        tf.keras.backend.clear_session()
        kernel = np.array([0.2, -0.5, 0.7, 0.1], dtype="float32")

        _, tf_coeff_model = patched_legacy_create(
            inputSize=15,
            kernelInit=kernel,
            kernTrainable=True,
            level=3,
            lossCoeff="l1",
            kernelsConstraint=legacy_constraint,
            initHT=0.25,
            trainHT=True,
        )
        torch_model = Despawn(
            kernel_init=kernel,
            kernel_learnable=True,
            kernels_constraint=torch_constraint,
            n_levels=3,
            loss_coeff="l1",
            threshold_init=0.25,
            threshold_learnable=True,
        )

        tf_recon, tf_approx, tf_details = tf_coeff_model(
            rearrange(model_input, "N C H W -> N H W C")
        )
        with torch.no_grad():
            torch_recon, torch_approx, torch_details = torch_model(
                torch.from_numpy(model_input), return_coeffs=True
            )

        torch_outputs = [torch_recon, torch_approx, *torch_details]
        tf_outputs = [tf_recon, tf_approx, *tf_details]

        for torch_output, tf_output in zip(torch_outputs, tf_outputs, strict=True):
            np.testing.assert_allclose(
                torch_output.numpy(),
                rearrange(tf_output.numpy(), "N H W C -> N C H W"),
                rtol=1e-6,
                atol=1e-6,
            )
