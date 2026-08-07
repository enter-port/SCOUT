"""``ScoutVIB``: state AE + VIB enc/dec with the joint loss (scout_design.md §3).

Single-chain, single-``backward()``, NO base-DP involvement and NO gradient
isolation (contrast SOE ``DPExt.backward``). The base Diffusion Policy is
absent during VIB training -- this module never imports ``scout.policy``.

Joint loss (one ``backward`` updates E_s/D_s/VIB_enc/VIB_dec together):

  s̄_t   = E_s(S_t)                         (used twice: AE + as VIB input)
  s̄_t1  = E_s(S_{t+1})                     NOT detached
  (μ,logvar) = VIB_enc(s̄_t, A_t);  z = reparam
  ŝ̄_{t+1} = VIB_dec(z, s̄_t)

  ae_loss  = MSE(D_s(s̄_t), S_t) + MSE(D_s(s̄_{t+1}), S_{t+1})    # anti-collapse anchor
  dyn_loss = MSE(ŝ̄_{t+1}, s̄_{t+1})                              # next-latent, not detached
  kl       = 0.5*(μ² + exp(logvar) - 1 - logvar).sum(-1).mean()  # KL(N(μ,σ²)||N(0,I))
  loss     = ae_loss + dyn_loss + beta*kl

The AE term is the make-or-break anti-collapse anchor: because the dynamics
target ``s̄_{t+1}`` is *not* detached, the dynamics pull would otherwise drift
``E_s`` toward a trivial constant; anchoring ``D_s(E_s(S))`` back to ``S`` on
both timesteps pins "reconstructible" (scout_design.md §3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scout.model.state_ae import StateAE
from scout.model.vib import VIBDecoder, VIBEncoder, reparam


class ScoutVIB(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        modality="low_dim",
        s_latent_dim=32,
        style_dim=16,
        hidden_dim=128,
        beta=1.0e-3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.modality = modality
        self.s_latent_dim = int(s_latent_dim)
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)

        self.ae = StateAE.from_config(
            modality, state_dim, latent_dim=s_latent_dim, hidden_dim=hidden_dim
        )
        self.vib_enc = VIBEncoder(
            action_dim=action_dim,
            s_latent_dim=s_latent_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
        )
        self.vib_dec = VIBDecoder(
            s_latent_dim=s_latent_dim,
            style_dim=style_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, S_t, A_t, S_tp1):
        """One transition batch -> joint loss dict.

        Inputs are ``(B, state_dim)`` / ``(B, action_dim)`` / ``(B, state_dim)``.
        Returns ``{"loss","ae","dyn","kl","mu","logvar"}``; all losses are
        scalars (mean-reduced), ``mu``/``logvar`` are ``(B, style_dim)``.
        """
        s_bar_t = self.ae.encode(S_t)
        s_bar_tp1 = self.ae.encode(S_tp1)                 # NOT detached (anchor rationale above)

        mu, logvar = self.vib_enc(s_bar_t, A_t)
        z = reparam(mu, logvar)
        s_bar_pred = self.vib_dec(z, s_bar_t)

        ae_loss = F.mse_loss(self.ae.decode(s_bar_t), S_t) \
                  + F.mse_loss(self.ae.decode(s_bar_tp1), S_tp1)
        dyn_loss = F.mse_loss(s_bar_pred, s_bar_tp1)      # target not detached
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()

        loss = ae_loss + dyn_loss + self.beta * kl
        return {"loss": loss, "ae": ae_loss, "dyn": dyn_loss, "kl": kl,
                "mu": mu, "logvar": logvar}
