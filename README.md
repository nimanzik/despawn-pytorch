# DeSpaWN-PyTorch

_**De**noising **Spa**rse **W**avelet **N**etwork (**DeSpaWN**) – PyTorch implementation_

## Monthly sunspot example

`examples/monthly_sunspots.py` ports the complete example from the original
TensorFlow project. It loads and standardizes the included sunspot data, trains
a model with the db4 kernel and NAdam, and plots the reconstruction and learned
coefficient distributions.

![DeSpaWN reconstruction and coefficient distributions](docs/assets/despawn-monthly-sunspots.png)

Run the example on CPU from the repository root:

```console
uv run --group examples --extra torch-cpu \
    python examples/monthly_sunspots.py
```

Training uses 1,000 epochs by default, as in the original script. Use
`--epochs 1 --no-show` for a quick check. Use `--output figure.png` to save the
plot. If Matplotlib cannot open a window, the example uses its noninteractive
backend and saves the figure in the system temporary directory.

The CSV loading, normalization, optimizer, training loop, db4 values, and plots
are part of the example. They are not part of the library API.

## Training

`DespawnLoss` combines mean absolute reconstruction error with the mean wavelet
coefficient penalty returned by the model.

```python
import torch

from despawn_pytorch import Despawn, DespawnLoss, get_num_levels

signal = torch.randn(8, 256)
n_levels = get_num_levels(signal.shape[-1])
model = Despawn(n_levels=n_levels)
criterion = DespawnLoss(sparsity_weight=1.0)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

reconstruction, coefficient_penalty = model(signal)
loss = criterion(signal, reconstruction, coefficient_penalty)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Set `sparsity_weight=0.0` to train with reconstruction loss only. A model
created with `loss_coeff=None` also returns a zero coefficient penalty.

## Tensor shapes

The time axis is always last. The model accepts tensors with shape
`(..., time)`, and it treats every item in the leading dimensions as an
independent signal. For example, both `(batch, time)` and
`(batch, sensors, time)` are valid. The model applies the same transform to
each sensor and does not mix information between sensors.

The reconstruction has the same shape as the input. The coefficient penalty
has shape `(...)`. `model.decompose(signal)` returns the reconstruction, the
final approximation, and a list of detail coefficients. Every coefficient
tensor preserves the input's leading dimensions and has its own reduced time
length as the last dimension.
