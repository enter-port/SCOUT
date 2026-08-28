"""rand_ushuffle -- per-chunk direction redraw (width-source decomposition).

CONTEXT (2026-08-28, campaign re-aim = retry-distribution WIDTH).  The pshell
screen found the campaign's WIDEST arm is the chunk-refreshed anchor
(pshell_c, PR 1.72 / d_act 0.77 = 方案A original anchor chasing), while the
persistent anchor is ~25-30% NARER on every width metric.  But 方案A/pshell
hold TWO things fixed per retry and refresh ONE per chunk:

    anchor (mu^0, sigma^0^2)  -- pshell knob rand_anchor_refresh
    direction u               -- always per-retry (frozen, ShellTarget-
                                 CostPlanner._u_for keyed by (init, try))

So the chunk arm's width is confounded: it could come from the anchor
TRACKING the deflected intent, from the direction being random, or from
their interaction.  ushuffle is the missing grid cell: decouple the
direction clock from the anchor clock with ``rand_u_resample``:

    retry (default) -- u frozen per (init, try), byte-equal to 方案A/pshell
                       u behavior (rng seed [shell_seed, init, try]).
    chunk           -- u redrawn at EVERY chunk from
                       rng([shell_seed, init, try, chunk_idx]) (1-based
                       select_z firing counter), so both the well center's
                       anchor and its displacement direction move per chunk.

Target arm ushuffle_c_k25: rand_anchor_refresh=chunk + rand_u_resample=
chunk, kappa=2.5, eta=0.35 -- identical to pshell_c_k25 EXCEPT u redraws
per chunk.  Two honest outcomes, decided by width caliber (pshell_width.py):
    ushuffle_c >> pshell_c  -> direction randomness is a width SOURCE;
    ushuffle_c ~= pshell_c  -> width comes from anchor tracking, not from
                               the direction being random.

Mechanics: inherits PShellCostPlanner verbatim (anchor_refresh retry|chunk|w5
passed through), overrides ONLY the direction supply:
  * select_z -- additionally ticks a per-(init,try) chunk counter
    (self._u_ctr) on every firing, regardless of anchor mode (pshell's own
    _chunk_ctr only ticks when anchor_refresh != "chunk", so ushuffle keeps
    an independent clock);
  * _u_for -- dispatches by u_resample: "retry" delegates to the parent
    (byte-equal), "chunk" draws from the 4-element seed list and caches by
    (init, try, chunk_idx).
compute_loss is inherited UNCHANGED from pshell (it already dispatches the
anchor by refresh mode and calls self._u_for for the direction -- the
override above is the only behavioral seam).

ek knobs: rand_u_resample ("retry"|"chunk", default "retry"),
rand_anchor_refresh ("retry"|"chunk"|"w5", default "retry" -- passthrough),
shell_kappa (2.5), shell_seed (42).
"""
from __future__ import annotations

import numpy as np
import torch

from scout.guidance.rand_costs.pshell import (
    _REFRESH_MODES,
    PShellCostPlanner,
)

NAME = "ushuffle"

_U_MODES = ("retry", "chunk")


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    u_mode = str(ek.get("rand_u_resample", "retry"))
    if u_mode not in _U_MODES:
        raise ValueError(f"rand_u_resample must be one of {_U_MODES},"
                         f" got {u_mode!r}")
    refresh = str(ek.get("rand_anchor_refresh", "retry"))
    if refresh not in _REFRESH_MODES:
        raise ValueError(f"rand_anchor_refresh must be one of {_REFRESH_MODES},"
                         f" got {refresh!r}")
    return UShuffleCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        anchor_refresh=refresh,
        u_resample=u_mode,
        shell_kappa=float(ek.get("shell_kappa", 2.5)),
        shell_seed=int(ek.get("shell_seed", 42)),
    )


class UShuffleCostPlanner(PShellCostPlanner):
    """pshell (方案A KL well) + an independent per-chunk direction clock."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 anchor_refresh: str = "retry", u_resample: str = "retry",
                 shell_kappa: float = 2.5, shell_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         anchor_refresh=anchor_refresh,
                         shell_kappa=shell_kappa, shell_seed=shell_seed)
        if u_resample not in _U_MODES:
            raise ValueError(f"u_resample must be one of {_U_MODES},"
                             f" got {u_resample!r}")
        self.u_resample = str(u_resample)
        # (init_idx, try_idx) -> select_z firings seen (ticks EVERY chunk,
        # even in anchor_refresh=chunk mode where pshell's _chunk_ctr does
        # not advance -- the two clocks are deliberately independent).
        self._u_ctr: dict = {}
        # (init_idx, try_idx, chunk_idx) -> unit u tensor (chunk mode only;
        # retry mode uses the parent _u_cache untouched).
        self._ush_u_cache: dict = {}

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Parent capture (baseline + anchor per refresh mode), then tick
        the direction clock once per chunk for every job-keyed row."""
        super().select_z(x0_hat, current_obs)
        if self.u_resample != "chunk":
            return None
        for i in range(len(self._base_mu)):        # rows of THIS batch
            key = self._row_key(i)
            if key[0] is None:
                continue                            # no job context -> no clock
            self._u_ctr[key] = self._u_ctr.get(key, 0) + 1
        return None

    def _u_for(self, key, device, dtype) -> torch.Tensor:
        """retry -> parent supply (byte-equal 方案A).  chunk -> redraw per
        chunk_idx with seed [shell_seed, init, try, chunk_idx]."""
        if self.u_resample != "chunk":
            return super()._u_for(key, device, dtype)
        ukey = (int(key[0]), int(key[1]),
                int(self._u_ctr.get(key, 0)))       # 1-based per chunk
        u = self._ush_u_cache.get(ukey)
        if u is None:
            rng = np.random.default_rng([self.shell_seed,
                                         ukey[0], ukey[1], ukey[2]])
            v = rng.standard_normal(int(self.scout_vib.style_dim))
            n = float(np.linalg.norm(v))
            u = torch.as_tensor(v / (n if n > 0.0 else 1.0),
                                dtype=torch.float32)
            self._ush_u_cache[ukey] = u
        return u.to(device=device, dtype=dtype)
