# DeSpaWN-PyTorch

_**De**noising **Spa**rse **W**avelet **N**etwork (**DeSpaWN**) – PyTorch implementation_

## Training

`DespawnLoss` combines mean absolute reconstruction error with the mean wavelet
coefficient penalty returned by the model.

```python
import torch

from despawn_pytorch import Despawn, DespawnLoss, get_num_levels

signal = torch.randn(8, 1, 256, 1)
n_levels = get_num_levels(signal.shape[2])
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
