"""``ScoutVIB``: E_s + VIB encoder + dynamics decoder + joint latent-level loss
(scout_design.md §1, §3).

SCOUT's dynamics is **self-developed** (scout_design.md §0 reuse boundary):
``VIB_enc -> z (variational skill) -> D_s`` with a latent MSE + KL objective. It
is structurally distinct from LPB's ``VisualDynamicsModel`` (deterministic
embedding, no μ/logvar/KL) -- do not fork LPB's dynamics. Only E_s's front-end
components (frozen ResNet + trained proprio Conv1d) are borrowed; this module
owns everything from ``s̄_t`` onward.

Joint loss (one ``backward`` updates vib_enc / D_s / proprio_embed; the frozen
ResNet has ``requires_grad=False`` so no grads flow into it -- single-chain, no
gradient isolation needed, scout_design.md §3):

  s̄_t        = E_s(obs_t)                                    (B, s_bar_dim)
  (μ,logvar) = VIB_enc(s̄_t, a_t);  z = reparam
  ŝ̄_{t+1}    = D_s(z, s̄_t)
  s̄_{t+1}   = E_s(obs_{t+1}).detach()                        latent target

  latent_mse = MSE(ŝ̄_{t+1}, s̄_{t+1})                          (latent-level, LPB)
  kl         = 0.5*(μ² + exp(logvar) - 1 - logvar).sum(-1).mean()
  loss       = latent_mse + beta*kl

No state decoder, no pixel decode, no reconstruction (scout_design.md §3, §7
risk #2/#3 -- latent-level target only). The base Diffusion Policy is absent
during VIB training; this module never imports ``scout.policy``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scout.model.encoder import StateEncoder
from scout.model.vib import DynamicsDecoder, VIBEncoder, reparam


class ScoutVIB(nn.Module):
    def __init__(
        self,
        action_dim: int,
        E_s: StateEncoder,
        style_dim: int = 16,
        hidden_dim: int = 128,
        beta: float = 1.0e-3,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)

        # E_s is injected already-constructed (caller picks base-DP ckpt vs mock).
        # s_bar_dim follows whatever E_s reports (512*n_views + proprio_emb_dim).
        self.E_s = E_s
        s_bar_dim = int(E_s.s_bar_dim)

        self.vib_enc = VIBEncoder(
            action_dim=action_dim,
            s_bar_dim=s_bar_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
        )
        self.D_s = DynamicsDecoder(
            s_bar_dim=s_bar_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
        )

    def encode(self, obs: dict) -> torch.Tensor:
        """``obs`` (visual+proprio) -> ``s̄`` of shape ``(B, s_bar_dim)``.

        Squeezes the T=1 time dim produced by :class:`StateEncoder` so downstream
        heads see ``(B, s_bar_dim)`` (SCOUT processes one transition per step).
        """
        s_bar = self.E_s(obs)              # (B, 1, s_bar_dim)
        return s_bar.squeeze(1)

    def forward(
        self,
        obs_t: dict,
        a_t: torch.Tensor,
        obs_tp1: dict,
    ) -> dict:
        """One transition batch -> joint loss dict.

        Inputs:
          obs_t, obs_tp1 : ``{"visual": {view: (B,1,3,H,W)}, "proprio": (B,1,P)}``
          a_t            : ``(B, action_dim)``

        Returns ``{"loss","latent_mse","kl","mu","logvar"}``; losses are scalars
        (mean-reduced), ``mu``/``logvar`` are ``(B, style_dim)``.
        """
        s_bar_t = self.encode(obs_t)
        s_bar_tp1 = self.encode(obs_tp1).detach()          # latent target (no grad)

        mu, logvar = self.vib_enc(s_bar_t, a_t)
        z = reparam(mu, logvar)
        s_bar_pred = self.D_s(z, s_bar_t)                   # ŝ̄_{t+1}

        latent_mse = F.mse_loss(s_bar_pred, s_bar_tp1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()

        loss = latent_mse + self.beta * kl
        return {"loss": loss, "latent_mse": latent_mse, "kl": kl,
                "mu": mu, "logvar": logvar}
