"""``ScoutVIB``: E_s + VIB encoder + dynamics decoder + state decoder with the
joint next-state loss (scout_design.md §3).

Single-chain, single-``backward()``, NO base-DP involvement and NO gradient
isolation (contrast SOE ``DPExt.backward``). The base Diffusion Policy is
absent during VIB training -- this module never imports ``scout.policy``.

Joint loss (one ``backward`` updates vib_enc / D_s / state_dec together;
E_s is identity for low_dim → no params → no grads):

  s̄_t        = E_s(S_t)                              (identity for low_dim)
  (μ,logvar) = VIB_enc(s̄_t, A_t);  z = reparam
  ŝ̄_{t+1}    = D_s(z, s̄_t)
  Ŝ_{t+1}    = state_dec(ŝ̄_{t+1})

  next_state_mse = MSE(Ŝ_{t+1}, S_{t+1})               (next-state target)
  kl             = 0.5*(μ² + exp(logvar) - 1 - logvar).sum(-1).mean()
  loss           = next_state_mse + beta*kl

No AE, no reconstruction, no latent-level loss (scout_design.md §3, §7 risk
#3). Anti-collapse is structural: low_dim E_s=identity (cannot collapse),
image E_s ResNet frozen; D_s/state_dec are pinned by the next-state MSE itself
(a dynamics that ignores z and predicts a constant has high loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scout.model.encoder import StateEncoder
from scout.model.vib import DynamicsDecoder, StateDecoder, VIBEncoder, reparam


class ScoutVIB(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        modality="low_dim",
        style_dim=16,
        hidden_dim=128,
        beta=1.0e-3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.modality = modality
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)

        # E_s: LPB-style encoder, NO autoencoder. Identity for low_dim (no params).
        # Exposes s_bar_dim so the downstream heads size themselves off E_s.
        self.E_s = StateEncoder.from_config(
            modality, state_dim, hidden_dim=hidden_dim
        )
        s_bar_dim = self.E_s.s_bar_dim

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
        self.state_dec = StateDecoder(
            s_bar_dim=s_bar_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, S_t, A_t, S_tp1):
        """One transition batch -> joint loss dict.

        Inputs are ``(B, state_dim)`` / ``(B, action_dim)`` / ``(B, state_dim)``.
        Returns ``{"loss","next_state_mse","kl","mu","logvar"}``; all losses
        are scalars (mean-reduced), ``mu``/``logvar`` are ``(B, style_dim)``.
        """
        s_bar_t = self.E_s(S_t)

        mu, logvar = self.vib_enc(s_bar_t, A_t)
        z = reparam(mu, logvar)
        s_bar_pred = self.D_s(z, s_bar_t)          # ŝ̄_{t+1}
        S_pred = self.state_dec(s_bar_pred)         # Ŝ_{t+1}

        next_state_mse = F.mse_loss(S_pred, S_tp1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()

        loss = next_state_mse + self.beta * kl
        return {"loss": loss, "next_state_mse": next_state_mse, "kl": kl,
                "mu": mu, "logvar": logvar}
