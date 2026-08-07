"""VIB encoder + dynamics decoder + state decoder + reparam (scout_design.md §2, §3).

Information-bottleneck skill latent + the dynamics/state heads. Mirrors the
SOE ``dp_ext.py:72-81`` block pattern (EncoderMLP for the down/up modules)
but with I/O shaped for the next-state dynamics:

  (s̄_t, a_t) --[VIBEncoder]--> (μ, logvar) --reparam--> z
                                                       │
                    s̄_t ────────────────────────────── ├─[DynamicsDecoder: D_s]--> ŝ̄_{t+1} --[StateDecoder]--> Ŝ_{t+1}

Dims (stage-1 low_dim lift): ``style_dim``=16, ``s_bar_dim``=state_dim (≈19,
since E_s is identity), hidden=128. The encoder predicts ``2*style_dim``
(μ||logvar); D_s predicts the next s̄ (dim = ``s_bar_dim``); state_dec maps
s̄ → env state (``state_dim``). No base-DP involvement here.
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
    ``(μ, logvar)``. ``s_bar_dim`` is the E_s output dim (= state_dim for the
    low_dim identity path).
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
    For low_dim (E_s identity) ``s_bar_dim`` = ``state_dim``; for the stage-2
    image path ``s_bar_dim`` = the frozen-ResNet feature dim. ``s_bar_dim`` here
    must match the ``s_bar_dim`` passed to :class:`VIBEncoder`.
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


class StateDecoder(nn.Module):
    """``ŝ̄_{t+1} -> Ŝ_{t+1}`` (low-dim **env state**, NOT image) via one EncoderMLP.

    Maps the predicted next s̄ to the env state used for the next-state MSE
    target (scout_design.md §3). Output dim = ``state_dim``. Always outputs
    low-dim env state even on the stage-2 image path -- this structurally
    avoids next-image prediction (scout_design.md §6, §7 risk #2).
    """

    def __init__(self, s_bar_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.s_bar_dim = int(s_bar_dim)
        self.state_dim = int(state_dim)
        self.net = EncoderMLP(
            input_dim=self.s_bar_dim,
            output_dim=self.state_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, s_bar_pred):
        return self.net(s_bar_pred)
