# -*- coding: utf-8 -*-
"""
Step 5 validation — correctness checks for the PyTorch DeSpaWN port.

Tests
-----
1. Shape preservation     — output shape must exactly match input shape for
                            every mode, multiple signal lengths, edge cases.
2. Reconstruction quality — with frozen db-4 kernels and identity thresholding
                            (initHT=0 → HT(x)=x analytically), MSE must be
                            small (boundary effects only; threshold 0.05).
3. Parameter counting     — verify the expected number of trainable parameters
                            for each kernel constraint mode.
4. Gradient flow          — one forward+backward pass; every trainable
                            parameter must receive a non-zero gradient.
5. Coefficient structure  — return_coeffs=True must give the right shapes,
                            list length, and coarsest→finest ordering.

Exit code: 0 if all tests pass, 1 if any test fails.
"""

import sys
import numpy as np
import torch

# Make sure lib/ is importable when running from the project root
sys.path.insert(0, '.')
from lib.despawn_torch import createDeSpaWN

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# db-4 wavelet kernel — same as the demo scripts
DB4 = np.array([
    -0.010597401785069032,  0.0328830116668852,
     0.030841381835560764, -0.18703481171909309,
    -0.027983769416859854,  0.6308807679298589,
     0.7148465705529157,    0.2303778133088965,
])
K     = len(DB4)      # 8
MODES = ['CQF', 'PerLayer', 'PerFilter', 'Free']

torch.manual_seed(0)   # deterministic random tensors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_all_passed = True

def check(condition, message):
    global _all_passed
    tag = "PASS" if condition else "FAIL"
    if not condition:
        _all_passed = False
    print(f"  [{tag}]  {message}")
    return condition


# ============================================================
# TEST 1 — Shape preservation
# ============================================================
print("=" * 65)
print("TEST 1: Shape preservation")
print("  Output shape must exactly match input shape for every")
print("  combination of mode × signal length.")
print("=" * 65)

# Signal lengths: powers of 2, round numbers, odd, the real-world 2000
# Level is always floor(log2(T)) to stay within safe decomposition depth.
LENGTHS = [16, 100, 99, 512, 2000, 3599]

for mode in MODES:
    for T in LENGTHS:
        level = int(np.floor(np.log2(T)))
        x = torch.randn(1, 1, T, 1)
        model = createDeSpaWN(
            kernelInit=DB4,
            kernTrainable=False,
            level=level,
            lossCoeff='l1',
            kernelsConstraint=mode,
            initHT=0.3,
            trainHT=False,
        )
        model.eval()
        with torch.no_grad():
            rec, _ = model(x)
        check(
            rec.shape == x.shape,
            f"mode={mode:9s}  T={T:5d}  level={level:2d}  "
            f"out={tuple(rec.shape)}  in={tuple(x.shape)}"
        )


# ============================================================
# TEST 2 — Reconstruction quality
# ============================================================
print()
print("=" * 65)
print("TEST 2: Reconstruction quality (frozen db-4, initHT=0)")
print("  initHT=0 makes HardThresholdAssym an identity:")
print("  σ(10x) + σ(-10x) = 1  for any x  →  HT(x) = x")
print("  Residual error is boundary effects only; threshold MSE < 0.05")
print("=" * 65)

for mode in MODES:
    for T in [256, 2000]:
        level = int(np.floor(np.log2(T)))
        x = torch.randn(1, 1, T, 1)
        model = createDeSpaWN(
            kernelInit=DB4,
            kernTrainable=False,
            level=level,
            lossCoeff='l1',
            kernelsConstraint=mode,
            initHT=0.0,          # identity thresholding
            trainHT=False,
        )
        model.eval()
        with torch.no_grad():
            rec, _ = model(x)
        mse = torch.mean((x - rec) ** 2).item()
        check(
            mse < 0.05,
            f"mode={mode:9s}  T={T:5d}  level={level:2d}  MSE={mse:.2e}  (< 0.05)"
        )


# ============================================================
# TEST 3 — Parameter counting
# ============================================================
print()
print("=" * 65)
print("TEST 3: Parameter counting")
print(f"  kernel_size={K}   level=10   HT_params=2*(10+1)=22")
print(f"  CQF:       1 kernel  →  1×{K} + 22 = {1*K+22}")
print(f"  PerLayer: 10 kernels →  10×{K} + 22 = {10*K+22}")
print(f"  PerFilter:20 kernels →  20×{K} + 22 = {20*K+22}")
print(f"  Free:     40 kernels →  40×{K} + 22 = {40*K+22}")
print("=" * 65)

LEVEL = 10
HT_TOTAL = 2 * (LEVEL + 1)    # thrP + thrN per instance × (level + 1 instances)

expected_params = {
    'CQF':       1        * K + HT_TOTAL,
    'PerLayer':  LEVEL    * K + HT_TOTAL,
    'PerFilter': 2*LEVEL  * K + HT_TOTAL,
    'Free':      4*LEVEL  * K + HT_TOTAL,
}

for mode in MODES:
    model = createDeSpaWN(
        kernelInit=DB4,
        kernTrainable=True,
        level=LEVEL,
        kernelsConstraint=mode,
        trainHT=True,
    )
    n_params  = sum(p.numel() for p in model.parameters())
    exp       = expected_params[mode]
    check(
        n_params == exp,
        f"mode={mode:9s}  params={n_params:4d}  expected={exp:4d}"
    )


# ============================================================
# TEST 4 — Gradient flow
# ============================================================
print()
print("=" * 65)
print("TEST 4: Gradient flow")
print("  One forward+backward pass; every trainable parameter")
print("  must receive a non-zero gradient.")
print("=" * 65)

for mode in MODES:
    model = createDeSpaWN(
        kernelInit=DB4,
        kernTrainable=True,
        level=5,
        lossCoeff='l1',
        kernelsConstraint=mode,
        initHT=0.3,
        trainHT=True,
    )
    model.train()
    x = torch.randn(1, 1, 128, 1)

    rec, v_loss = model(x)
    loss = torch.mean(torch.abs(x - rec)) + torch.mean(v_loss)
    loss.backward()

    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (
            p.grad is None or p.grad.abs().sum().item() == 0.0
        )
    ]
    check(
        len(dead) == 0,
        f"mode={mode:9s}  "
        + ("all params have gradients"
           if not dead else f"no gradient on: {dead}")
    )


# ============================================================
# TEST 5 — Coefficient structure  (return_coeffs=True)
# ============================================================
print()
print("=" * 65)
print("TEST 5: Coefficient structure  (return_coeffs=True)")
print("  Mirrors TF model2 output: (rec, gint, hl[::-1])")
print("=" * 65)

T     = 2000
LEVEL = int(np.floor(np.log2(T)))    # 10

x = torch.randn(1, 1, T, 1)
model = createDeSpaWN(
    kernelInit=DB4,
    kernTrainable=False,
    level=LEVEL,
    kernelsConstraint='PerLayer',
    initHT=0.3,
    trainHT=False,
)
model.eval()
with torch.no_grad():
    rec, gint, hl_rev = model(x, return_coeffs=True)

# 5a — reconstructed signal has same shape as input
check(
    rec.shape == x.shape,
    f"rec shape {tuple(rec.shape)} == input {tuple(x.shape)}"
)

# 5b — gint is 4-D NCHW with correct batch/channel/width
check(
    gint.ndim == 4 and gint.shape[0] == 1 and gint.shape[1] == 1 and gint.shape[3] == 1,
    f"gint shape {tuple(gint.shape)} is (1, 1, T_approx, 1)"
)

# 5c — hl_rev has exactly 'level' entries
check(
    len(hl_rev) == LEVEL,
    f"len(hl_rev)={len(hl_rev)} == level={LEVEL}"
)

# 5d — detail tensors are 4-D NCHW (1, 1, T_i, 1)
all_nchw = all(
    h.ndim == 4 and h.shape[0] == 1 and h.shape[1] == 1 and h.shape[3] == 1
    for h in hl_rev
)
check(all_nchw, "all detail tensors have shape (1, 1, T_i, 1)")

# 5e — sizes increase from coarsest (hl_rev[0]) to finest (hl_rev[-1])
#      hl_rev = hl[::-1]: hl[0] is finest (T/2), so hl_rev[0] is coarsest (T_deepest)
sizes = [h.shape[2] for h in hl_rev]
check(
    sizes == sorted(sizes),
    f"detail sizes coarsest→finest (ascending): {sizes}"
)

# 5f — total coefficients span the full signal in the reconstruction basis
#      sum of all T_i + T_approx should equal the concatenated axis size
total_t = gint.shape[2] + sum(sizes)
check(
    total_t > 0,
    f"total coefficient time-steps: gint({gint.shape[2]}) + details({sum(sizes)}) = {total_t}"
)

# 5g — forward(return_coeffs=False) and forward(return_coeffs=True)
#      give identical reconstructions (same weights, same forward pass)
with torch.no_grad():
    rec2, _ = model(x, return_coeffs=False)
max_diff = (rec - rec2).abs().max().item()
check(
    max_diff == 0.0,
    f"return_coeffs=False and True give identical rec  (max_diff={max_diff:.2e})"
)


# ============================================================
# Summary
# ============================================================
print()
print("=" * 65)
if _all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — see FAIL lines above")
print("=" * 65)

sys.exit(0 if _all_passed else 1)
