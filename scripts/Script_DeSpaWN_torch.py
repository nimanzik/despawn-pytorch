# -*- coding: utf-8 -*-
"""
Title: Fully Learnable Deep Wavelet Transform for Unsupervised Monitoring
------         of High-Frequency Time Series (DeSpaWN) — PyTorch port

Description:
--------------
Toy script to showcase the deep neural network DeSpaWN in PyTorch.
Direct port of Script_DeSpaWN.py from TensorFlow 2.1 to PyTorch.

Please cite the corresponding paper:
          Michau, G., Frusque, G., & Fink, O. (2022).
          Fully learnable deep wavelet transform for unsupervised monitoring of
          high-frequency time series.
          Proceedings of the National Academy of Sciences, 119(8).

Original author: Dr. Gabriel Michau,
                 Chair of Intelligent Maintenance Systems, ETH Zürich

Porting notes
-------------
* The only structural change vs. the TF script is tensor layout:
    TF  NHWC  (batch, time, 1, 1)  →  PyTorch  NCHW  (batch, 1, time, 1)
  Every shape index that was [1] in the TF script becomes [2] here.
* The two-model pattern (model1 / model2) is replaced by a single DeSpaWN
  module whose forward() accepts a return_coeffs flag.
* model1.fit()  →  manual training loop with torch.optim.NAdam.
* model1/2.predict()  →  model.eval() + torch.no_grad() calls.
* All plotting code mirrors the original structure exactly; only the
  indexing into output arrays is adapted for NCHW.
"""

# Usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from lib.despawn_torch import createDeSpaWN


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

signal = pd.read_csv("monthly-sunspots.csv")
lTrain = 2000   # length of the training section

# Normalise — identical to TF script
# TF shape:     (batch, time, 1, 1)   NHWC
# PyTorch shape:(batch, 1, time, 1)   NCHW  ← only difference in this file
signalT = (
    (signal['Sunspots'] - signal['Sunspots'].mean()) / signal['Sunspots'].std()
).values[np.newaxis, np.newaxis, :, np.newaxis].astype(np.float32)   # (1, 1, N_total, 1)

signal = signalT[:, :, :lTrain, :]                                   # (1, 1, 2000, 1)


# ---------------------------------------------------------------------------
# Hyperparameters  (identical to TF script)
# ---------------------------------------------------------------------------

# Number of decomposition levels: max floor(log2(T))
# shape[2] = T in NCHW  (was shape[1] in TF NHWC)
level = int(np.floor(np.log2(signal.shape[2])))

# Train hard thresholding (HT) coefficient?
trainHT = True
# Initialise HT value
initHT = 0.3
# Which loss to consider for wavelet coeffs ('l1' or None)
lossCoeff = 'l1'
# Weight for sparsity loss versus residual?
lossFactor = 1.0
# Train wavelets? (Trainable kernels)
kernTrainable = True
# Which training mode?
# cf (https://arxiv.org/pdf/2105.00899.pdf -- https://doi.org/10.1073/pnas.2106598119)
# [Section 4.4 Ablation Study]
#   CQF       => learn wavelet, infer all other kernels from the network
#   PerLayer  => learn one wavelet per level, infer others
#   PerFilter => learn wavelet + scaling function per level, infer others
#   Free      => learn everything
mode = 'PerLayer'   # CQF | PerLayer | PerFilter | Free

# Initialise wavelet kernel (here db-4)
kernelInit = np.array([
    -0.010597401785069032,  0.0328830116668852,
     0.030841381835560764, -0.18703481171909309,
    -0.027983769416859854,  0.6308807679298589,
     0.7148465705529157,    0.2303778133088965,
])

epochs  = 1000
verbose = 2     # 0 = silent | 2 = one line per epoch  (mirrors TF verbose arg)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def coeff_loss(v_loss):
    """
    Sparsity loss on wavelet coefficients.
    Mirrors TF's  coeffLoss = lossFactor * tf.reduce_mean(yPred).
    v_loss already contains mean(|coefficients|); we scale and reduce further.
    yTrue is unused, as in the original (an empty array was passed as target).
    """
    return lossFactor * torch.mean(v_loss)


def rec_loss(y_true, y_pred):
    """
    MAE reconstruction loss.
    Mirrors TF's  recLoss = tf.math.abs(yTrue - yPred), which Keras reduces
    to a mean over all elements.
    """
    return torch.mean(torch.abs(y_true - y_pred))


# ---------------------------------------------------------------------------
# Model + optimiser
# ---------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# createDeSpaWN returns a single DeSpaWN nn.Module that replaces both
# model1 (training) and model2 (inspection) from the TF script.
model = createDeSpaWN(
    inputSize=None,
    kernelInit=kernelInit,
    kernTrainable=kernTrainable,
    level=level,
    lossCoeff=lossCoeff,
    kernelsConstraint=mode,
    initHT=initHT,
    trainHT=trainHT,
).to(device)

# Nadam with the same hyperparameters as the TF script
opt = torch.optim.NAdam(
    model.parameters(),
    lr=0.001, betas=(0.9, 0.999), eps=1e-07,
)


# ---------------------------------------------------------------------------
# Training tensors
# ---------------------------------------------------------------------------

signal_t  = torch.tensor(signal,  device=device)   # (1, 1, 2000,    1)
signalT_t = torch.tensor(signalT, device=device)   # (1, 1, N_total, 1)


# ---------------------------------------------------------------------------
# Training loop
# Replaces: model1.compile(...) + model1.fit(signal, [signal, empty], ...)
# ---------------------------------------------------------------------------

# H stores per-epoch losses, mirroring TF's History object
H = {'loss': [], 'rec_loss': [], 'coeff_loss': []}

for epoch in range(epochs):
    model.train()
    opt.zero_grad()

    # Forward — model(x) returns (reconstructed, sparsity_loss_term)
    # equivalent to TF model1 output
    rec, v_loss = model(signal_t)

    loss_r = rec_loss(signal_t, rec)
    loss_c = coeff_loss(v_loss)
    loss   = loss_r + loss_c

    loss.backward()
    opt.step()

    H['loss'].append(loss.item())
    H['rec_loss'].append(loss_r.item())
    H['coeff_loss'].append(loss_c.item())

    if verbose == 2:
        print(
            f"Epoch {epoch + 1:4d}/{epochs} — "
            f"loss: {loss.item():.6f}  "
            f"rec: {loss_r.item():.6f}  "
            f"coeff: {loss_c.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Inference
# Replaces: model1.predict(...) and model2.predict(...)
# ---------------------------------------------------------------------------

indPlot = 0

model.eval()
with torch.no_grad():
    # --- Training segment ---
    # model(x)                      ≡  model1.predict(signal)
    # model(x, return_coeffs=True)  ≡  model2.predict(signal)
    out_rec, _           = model(signal_t)
    _, gint, hl_rev      = model(signal_t, return_coeffs=True)

    # --- Test segment (unseen portion of the signal) ---
    signal_test            = signalT_t[:, :, lTrain:, :]
    out_rec_te, _          = model(signal_test)
    _, gint_te, hl_rev_te  = model(signal_test, return_coeffs=True)


def to_np(t):
    """Detach, move to CPU, convert to numpy."""
    return t.detach().cpu().numpy()


# Build coefficient lists that mirror TF's outC[1:] and outCTe[1:]:
#   outC[1]   = gint  (approximation, deepest level)
#   outC[2:]  = hl[::-1] (details, coarsest → finest)
# hl_rev already equals hl[::-1], so [gint] + hl_rev matches outC[1:] exactly.
outC   = [to_np(gint)]    + [to_np(h) for h in hl_rev]
outCTe = [to_np(gint_te)] + [to_np(h) for h in hl_rev_te]


# ---------------------------------------------------------------------------
# Plotting  (structure mirrors the original TF script exactly)
# ---------------------------------------------------------------------------
# Shape note: in NCHW, time is dim=2.
#   TF indexing:     o[indPlot, :, 0, 0]   (NHWC: dim-1 is time)
#   PyTorch indexing: o[indPlot, 0, :, 0]  (NCHW: dim-2 is time)
#   np.squeeze(o[indPlot]) works for both, since the extra dims are size 1.

fig = plt.figure(1)
fig.clf()

# --- Top panel: signal reconstruction (train + test) ---
ax = fig.add_subplot(2, 1, 1)
ax.plot(np.arange(signal.shape[2]),          signal [indPlot, 0, :,      0])   # train original
ax.plot(np.arange(signal.shape[2]),          to_np(out_rec)[indPlot, 0, :, 0]) # train reconstructed
ax.plot(np.arange(signal.shape[2], signalT.shape[2]),
        signalT[indPlot, 0, lTrain:, 0])                                        # test original
ax.plot(np.arange(signal.shape[2], signalT.shape[2]),
        to_np(out_rec_te)[indPlot, 0, :, 0])                                    # test reconstructed
ax.legend(['Train Original', 'Train Reconstructed',
           'Test Original',  'Test Reconstructed'])

# --- Bottom-left: train coefficient distributions per decomposition level ---
ax = fig.add_subplot(2, 2, 3)
for e, o in enumerate(outC):
    # o shape: (N, 1, T_i, 1)  →  o[indPlot] is (1, T_i, 1)  →  squeeze → (T_i,)
    ax.boxplot(np.abs(np.squeeze(o[indPlot])), positions=[e], widths=0.8)
ax.set_xlabel('Decomposition Level')
ax.set_ylabel('Coefficient Distribution')
trainYLim = ax.get_ylim()
trainXLim = ax.get_xlim()

# --- Bottom-right: test coefficient distributions per decomposition level ---
ax = fig.add_subplot(2, 2, 4)
for e, o in enumerate(outCTe):
    print(o.shape[2])   # shape[2] = T_i in NCHW  (was shape[1] in TF NHWC)
    if o.shape[2] > 1:
        ax.boxplot(np.abs(np.squeeze(o[indPlot])), positions=[e], widths=0.8)
    else:
        ax.plot(e, np.abs(np.squeeze(o[indPlot])), 'o', color='k')
ax.set_xlabel('Decomposition Level')
ax.set_ylabel('Coefficient Distribution')
ax.set_ylim(trainYLim)
ax.set_xlim(trainXLim)

plt.tight_layout()
plt.show()
