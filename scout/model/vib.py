"""VIB encoder + decoder + reparam (scout_design.md §2, §3).

Information-bottleneck skill latent. Mirrors the SOE ``dp_ext.py:72-81`` block
pattern (EncoderMLP for down/up modules) but with I/O shaped for the latent
dynamics:

  (s̄_t, a_t) --[VIBEncoder]--> (μ, logvar) --reparam--> z --[VIBDecoder]--> ŝ̄_{t+1}
                                                                       (s̄_t,)

Dims (stage-1 low_dim lift): ``style_dim``=16, ``s_latent_dim``=32, hidden=128.
The encoder predicts ``2*style_dim`` (μ||logvar); the decoder predicts
``s_latent_dim`` (next latent). No base-DP involvement here.
"""

import math

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
    ``(μ, logvar)``.
    """

    def __init__(self, action_dim, s_latent_dim=32, style_dim=16, hidden_dim=128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.s_latent_dim = int(s_latent_dim)
        self.style_dim = int(style_dim)
        self.net = EncoderMLP(
            input_dim=self.s_latent_dim + self.action_dim,
            output_dim=2 * self.style_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, s_bar, a):
        x = torch.cat([s_bar, a], dim=-1)
        mu, logvar = self.net(x).chunk(2, dim=-1)
        return mu, logvar


class VIBDecoder(nn.Module):
    """``concat(z, s̄_t) -> ŝ̄_{t+1}`` via a single EncoderMLP.

    Output dim = ``s_latent_dim`` (predicted next state latent).
    """

    def __init__(self, s_latent_dim=32, style_dim=16, hidden_dim=128):
        super().__init__()
        self.s_latent_dim = int(s_latent_dim)
        self.style_dim = int(style_dim)
        self.net = EncoderMLP(
            input_dim=self.style_dim + self.s_latent_dim,
            output_dim=self.s_latent_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, z, s_bar):
        x = torch.cat([z, s_bar], dim=-1)
        return self.net(x)
