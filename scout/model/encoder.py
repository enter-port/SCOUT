"""E_s state encoder (scout_design.md §2, §3) -- LPB-style, NO autoencoder.

Stage-1 low_dim: **identity** (s̄_t = S_t; no parameters, nothing learned).
Stage-2 image: frozen base-DP ResNet + trained proprio embed (LPB-style) -- a
drop-in via :meth:`StateEncoder.from_config('image', ...)`, which currently
raises ``NotImplementedError`` (see **【image 接口点】** below).

Why no AE (scout_design.md §3, §7 risk #3): the previous design used a state
AE both to produce ``s̄`` and as an anti-collapse anchor (``D_s(E_s(S)) → S``).
The redesign drops the AE entirely:

  - low_dim: E_s is identity → cannot collapse (nothing to learn);
  - image:   ResNet is frozen → cannot drift;
  - D_s / state_dec are pinned by the **next-state MSE** itself (a dynamics
    that predicts a constant has high loss). No reconstruction anchor needed.
"""

import torch.nn as nn


class StateEncoder(nn.Module):
    """E_s: state observation -> s̄ (LPB-style encoder, NO autoencoder).

    - low_dim: identity passthrough (s̄ = S); **zero parameters**.
    - image:   NotImplementedError (stage-2; frozen ResNet + proprio embed).

    Callers (VIB encoder, D_s, cost, guidance) only consume the ``s̄`` vector,
    so they are modality-agnostic -- swapping modality here is the only change
    needed for the image path (scout_design.md §6). ``s_bar_dim`` exposes the
    encoded dim so downstream modules can size themselves off E_s.
    """

    def __init__(self, state_dim, modality="low_dim", hidden_dim=128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.modality = modality
        if modality == "low_dim":
            # s̄_t = S_t. nn.Identity has no params → no grads flow to E_s
            # (correct: E_s must not train in low_dim, scout_design.md §3).
            self._impl = nn.Identity()
            self.s_bar_dim = self.state_dim
        elif modality == "image":
            # **【image 接口点】** stage-2: frozen base-DP ResNet + trained
            # proprio embed (LPB ResNetEncoder + ProprioceptiveEmbedding).
            # Keep forward() signature identical so the rest of the pipeline
            # is untouched; set self.s_bar_dim to the fused feature dim.
            raise NotImplementedError(
                "StateEncoder(image) is stage-2 work -- see scout_design.md §6. "
                "Stage-1 is low_dim only; use from_config('low_dim', ...)."
            )
        else:
            raise ValueError(f"unknown StateEncoder modality: {modality!r}")

    def forward(self, S):
        """S (B, state_dim) -> s̄ (B, s_bar_dim). Identity for low_dim."""
        return self._impl(S)

    @staticmethod
    def from_config(modality, state_dim, hidden_dim=128):
        """Factory by modality string ('low_dim' stage-1, 'image' stage-2)."""
        return StateEncoder(state_dim, modality=modality, hidden_dim=hidden_dim)
