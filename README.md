# DeSpaWN-PyTorch

_**De**noising **Spa**rse **W**avelet **N**etwork (**DeSpaWN**) – PyTorch implementation_

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
