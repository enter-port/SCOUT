"""VIB encoder + dynamics decoder + reparam (scout_design.md §2, §3).

Information-bottleneck skill latent + the dynamics head. Mirrors the SOE
``dp_ext.py:72-81`` EncoderMLP block pattern but with I/O shaped for SCOUT's
next-latent dynamics (latent-level target = ``E_s(S_{t+1}).detach()``, no state
decoder -- scout_design.md §3):

  (s̄_t, a_t) --[VIBEncoder]--> (μ, logvar) --reparam--> z
                                                       │
                    s̄_t ────────────────────────────── ├─[DynamicsDecoder: D_s]--> ŝ̄_{t+1}

Dims: ``style_dim``=16; ``s_bar_dim`` follows E_s (= 512*n_views + proprio_emb_dim
on the LPB-style image path). The encoder predicts ``2*style_dim`` (μ||logvar,
chunked in half along the last dim); D_s predicts the next s̄ (dim = ``s_bar_dim``).
No base-DP involvement here.
"""

import torch
import torch.nn as nn

from scout.model.mlp import EncoderMLP


def reparam(mu, logvar):
    """Reparameterise: ``z = μ + σ ⊙ ε``, ``σ = exp(0.5·logvar)``, ``ε~N(0,I)``.

    Matches SOE ``dp_ext.py`` and the standard VAE reparam. ``mu``/``logvar``
    are ``(B, style_dim)``; returns ``z`` of the same shape. No ``noise_scale``
    here -- that's an inference-only knob on the *guidance* side, not training.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class VIBEncoder(nn.Module):
    """``concat(s̄_t, a_t) -> (μ, logvar)`` via a single EncoderMLP.

    Output dim = ``2*style_dim``; chunked in half along the last dim into
    ``(μ, logvar)``. ``s_bar_dim`` is the E_s output dim (= 512*n_views +
    proprio_emb_dim on the LPB-style image path).
    """

    def __init__(self, action_dim, s_bar_dim, style_dim=16, hidden_dim=128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.s_bar_dim = int(s_bar_dim)
        self.style_dim = int(style_dim)
        self.net = EncoderMLP(
            input_dim=self.s_bar_dim + self.action_dim,
            output_dim=2 * self.style_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, s_bar, a):
        x = torch.cat([s_bar, a], dim=-1)
        mu, logvar = self.net(x).chunk(2, dim=-1)
        return mu, logvar


class DynamicsDecoder(nn.Module):
    """D_s: ``concat(z, s̄_t) -> ŝ̄_{t+1}`` via a single EncoderMLP.

    Predicts the next encoded observation (s̄-space); output dim = ``s_bar_dim``.
    Target is ``E_s(S_{t+1}).detach()`` (latent-level, scout_design.md §3) --
    no state decoder, no pixel decode.
    """

    def __init__(self, s_bar_dim, style_dim=16, hidden_dim=128):
        super().__init__()
        self.s_bar_dim = int(s_bar_dim)
        self.style_dim = int(style_dim)
        self.net = EncoderMLP(
            input_dim=self.style_dim + self.s_bar_dim,
            output_dim=self.s_bar_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, z, s_bar):
        x = torch.cat([z, s_bar], dim=-1)
        return self.net(x)
