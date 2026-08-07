"""Online running mean/std via Welford (batched Chan-Golub-LeVeque parallel form).

Numerically stable incremental accumulation over batches of vectors. Used to
normalize states / actions on the fly from a growing replay buffer, without
recomputing over the full dataset each time (see scout_design.md §3:
"running 统计(Welford 增量)").
"""

import torch


class RunningStats:
    """Accumulate mean and variance of (B, dim) batches incrementally.

    All vectors are 1-D of fixed ``dim``. Stats live on CPU float32; call
    ``normalize``/``unnormalize`` on CPU tensors (move stats with ``.to()`` if
    a GPU pipeline needs it -- deferred until later phases).
    """

    def __init__(self, dim, eps=1e-6):
        self.dim = int(dim)
        self.eps = float(eps)
        self.n = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float32)
        self.m2 = torch.zeros(self.dim, dtype=torch.float32)  # sum of squared deviations

    def update(self, x):
        """Incorporate a batch ``x`` of shape (B, dim) (or a single (dim,) vector)."""
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.dim, f"expected dim {self.dim}, got {x.shape[1]}"

        n_b = x.shape[0]
        mean_b = x.mean(dim=0)
        m2_b = ((x - mean_b) ** 2).sum(dim=0)

        if self.n == 0:
            self.mean = mean_b.clone()
            self.m2 = m2_b.clone()
            self.n = n_b
        else:
            delta = mean_b - self.mean           # (dim,)
            new_n = self.n + n_b
            self.mean = self.mean + delta * (n_b / new_n)
            self.m2 = self.m2 + m2_b + (delta ** 2) * (self.n * n_b / new_n)
            self.n = new_n

    @property
    def var(self):
        # population variance; eps-floor guards zero-variance dims
        return self.m2 / max(self.n, 1) + self.eps

    @property
    def std(self):
        return self.var.sqrt()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def __len__(self):
        return self.n
