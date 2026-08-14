"""E_s state encoder (scout_design.md §0, §2, §3) -- LPB-style dual input, NO AE.

Always image + proprio (LPB-style; scout_design.md §0 "永远 image + proprio 同时输入",
no low_dim/image stage split). The encoder fuses two **non-dynamics** components
borrowed from LPB (scout_design.md §0 reuse boundary):

  - image : ``dyn_model.models.resnet_encoder.ResNetEncoder`` -- per-view frozen
            base-DP ResNet-18 + AdaptiveAvgPool2d -> 512 / view (NOT trained;
            ``requires_grad=False`` so the stable-anchor argument of §3/§7 #3 holds).
  - proprio: ``dyn_model.models.proprio.ProprioceptiveEmbedding`` -- Conv1d, trained.

These are **front-end only**; they emit ``s̄_t``. The SCOUT-self-developed dynamics
(``VIB_enc -> z -> D_s``) lives in :mod:`scout.model.scout_vib` and is NOT forked
from LPB (LPB's z is a deterministic embedding with no μ/logvar/KL).

Forward signature mirrors the LPB ``VisualDynamicsModel.encode_obs`` visual+proprio
fusion -- **but with no action** (SCOUT's action enters VIB_enc, not E_s; the
design's reuse boundary is explicit on this point). Output time dim is kept (T=1
for SCOUT's per-transition forward; callers squeeze it).

  forward({"visual": {view: (B,T,3,H,W)}, "proprio": (B,T,proprio_dim)})
    -> (B, T, s_bar_dim),  s_bar_dim = 512 * n_views + proprio_emb_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from dyn_model.models.proprio import ProprioceptiveEmbedding


class StateEncoder(nn.Module):
    """E_s: LPB-style dual-input encoder (image + proprio), no autoencoder.

    The ResNet is **frozen** (``requires_grad=False`` on construction; it is also
    kept in ``eval()`` mode via :meth:`train` so BN/dropout do not update even if
    a future caller flips the parent to ``.train()``). The proprio Conv1d is
    **trained**. ``s_bar_dim`` is exposed so VIB_enc / D_s size themselves off
    E_s (scout_design.md §2).
    """

    def __init__(
        self,
        resnet_encoder: nn.Module,
        view_names: list[str],
        proprio_dim: int,
        proprio_emb_dim: int = 64,
    ):
        super().__init__()
        self.view_names = list(view_names)
        self.proprio_dim = int(proprio_dim)
        self.proprio_emb_dim = int(proprio_emb_dim)

        # image branch: per-view frozen base-DP ResNet (LPB ResNetEncoder).
        # emb_dim per view is 512 (ResNetEncoder.emb_dim); fall back if absent.
        self.resnet = resnet_encoder
        self.emb_dim_per_view = int(getattr(self.resnet, "emb_dim", 512))
        self._freeze_resnet()

        # proprio branch: trained Conv1d (LPB ProprioceptiveEmbedding).
        # num_frames/tubelet keep LPB defaults; in_chans=proprio_dim, emb_dim set.
        self.proprio_embed = ProprioceptiveEmbedding(
            num_frames=1,
            tubelet_size=1,
            in_chans=self.proprio_dim,
            emb_dim=self.proprio_emb_dim,
        )

        # fused dim = visual (per-view 512) + proprio embed.
        self.visual_dim = self.emb_dim_per_view * len(self.view_names)
        self.s_bar_dim = self.visual_dim + self.proprio_emb_dim

    # ------------------------------------------------------------------ #
    # factories
    # ------------------------------------------------------------------ #
    @classmethod
    def from_base_dp_ckpt(
        cls,
        base_dp_ckpt: str,
        view_names: list[str],
        proprio_dim: int,
        proprio_emb_dim: int = 64,
    ) -> "StateEncoder":
        """Canonical construction: rip the frozen ResNet out of a base DP ckpt.

        Lazy-imports :mod:`dyn_model.models.resnet_encoder` so this module (and
        its trainers) import cleanly in environments without the LPB
        diffusion_policy/hydra stack -- only the actual ckpt load pulls those in.
        """
        from dyn_model.models.resnet_encoder import ResNetEncoder

        resnet = ResNetEncoder(base_dp_ckpt, view_names)
        return cls(resnet, view_names, proprio_dim, proprio_emb_dim)

    # ------------------------------------------------------------------ #
    # freeze bookkeeping
    # ------------------------------------------------------------------ #
    def _freeze_resnet(self):
        for p in self.resnet.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Parent .train(); **force ResNet to eval + frozen** so BN stats and any
        buffers (running_mean etc.) never update -- the frozen-anchor contract
        (scout_design.md §3, §7 risk #3).
        """
        super().train(mode)
        self.resnet.eval()
        return self

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self, obs: dict[str, dict[str, torch.Tensor] | torch.Tensor]
    ) -> torch.Tensor:
        """``obs = {"visual": {view: (B,T,3,H,W)}, "proprio": (B,T,proprio_dim)}``
        -> ``s̄`` of shape ``(B, T, s_bar_dim)``.

        Mirrors LPB ``VisualDynamicsModel.encode_obs`` visual+proprio fusion but
        with **no action** term (SCOUT's action goes into VIB_enc, not E_s).
        """
        view_embs = self.resnet(obs["visual"])              # {view: (B,T,1,512)}
        visual_parts = [view_embs[v].squeeze(-2) for v in self.view_names]  # (B,T,512) each
        visual = torch.cat(visual_parts, dim=-1)            # (B,T, 512*n_views)

        proprio_emb = self.proprio_embed(obs["proprio"])    # (B,T,proprio_emb_dim)

        s_bar = torch.cat([visual, proprio_emb], dim=-1)    # (B,T,s_bar_dim)
        return s_bar

    def forward_from_feats(
        self,
        visual_feats: dict[str, torch.Tensor],
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Same fusion as :meth:`forward` but with the frozen-ResNet outputs
        ALREADY computed (``visual_feats = {view: (B,T,512)}`` -- e.g. from
        the precomputed feature bank, :mod:`scout.feat_cache`). Identical
        result to ``forward`` for the same crops (the ResNet is frozen +
        eval, so its output is a constant per (frame, view, offset)); only
        the live proprio branch runs.

        ``proprio``: ``(B,T,proprio_dim)`` -- same layout as ``forward``.
        Returns ``(B,T,s_bar_dim)``.
        """
        visual_parts = [visual_feats[v] for v in self.view_names]
        visual = torch.cat(visual_parts, dim=-1)            # (B,T, 512*n_views)

        proprio_emb = self.proprio_embed(proprio)           # (B,T,proprio_emb_dim)

        s_bar = torch.cat([visual, proprio_emb], dim=-1)    # (B,T,s_bar_dim)
        return s_bar
