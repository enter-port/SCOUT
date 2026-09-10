"""Hermetic smoke for WalkCloudCostPlanner (drift-dev, user 2026-09-10,
idea/walkcloud_plan.md §4).

Run:  python -m scout.eval._smoke_walkcloud

Checks (all CPU, mock VIB -- no robomimic / no LPB stack):
  1. t=1 empty cloud: graph-connected zero (value AND gradient AND
     grad_fn) from _kl_backward and guided_step; the t=1 posterior is
     still pushed (it becomes the anchor for t=2).
  2. t=2 bitwise == atypical (J=1 fast path) under ANY tau mode/value:
     identical guided_step (cond_grad, row_losses) over a 2-step walk;
     t=1 is bitwise equal too (both exactly zero); t>=3 departs.
  3. history semantics: H grows by exactly 1 per _kl_backward; select_z
     resets to empty; compute_loss is read-only (no append).
  4. soft-min formula vs an independent per-row python loop (adapt tau
     included); kernel gradient dS/dKL_j == softmax(-KL/tau); weights
     sum to 1.
  5. trajectory gradient vs central finite differences (fixed-tau mode:
     tau is a detached schedule constant, so the FD must freeze it --
     the injected gradient is the tau-fixed partial).
  6. RNG: a full guided_step walk consumes NO torch RNG (bit-comparability
     with the other KL-cost arms on the same seed).
  7. cloud_hist_max > 0 keeps the NEWEST entries (oldest dropped).
"""
import torch

from scout.guidance.entropy_costs import KLCostPlanner, _enc_forward
from scout.guidance.walk_cloud_costs import (
    WalkCloudCostPlanner,
    _cloud_softmin,
)

Ds, Da, Dz = 6, 5, 4
torch.manual_seed(0)
W = torch.randn(Dz, Da)
B = 3


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                                # (B, Dz)
        logvar = -0.5 + 0.1 * mu[:, :1]             # varies with a
        return mu, torch.broadcast_to(logvar, mu.shape).clone()


class MockVib:
    style_dim = Dz

    def __init__(self):
        self.vib_enc = MockEncNet()

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def encode(self, obs):
        return obs                                  # s_bar passthrough


def _traj(x):
    """(B, 1, Da) leaf trajectory + graph-connected x0_hat view of it."""
    t = x.unsqueeze(1).clone().requires_grad_(True)
    return t, t * 1.0


def _kl_diag(mu, logvar, m0, lv0):
    var, var0 = torch.exp(logvar), torch.exp(lv0)
    return 0.5 * (((mu - m0) ** 2 / var0)
                  + (var / var0) - 1.0 - (logvar - lv0)).sum(dim=-1)


def main():
    vib = MockVib()
    s_bar = torch.randn(B, Ds)
    anchor_x = torch.randn(B, Da)

    # ---------------- 1. t=1 empty cloud: graph-connected zero ---------- #
    wc = WalkCloudCostPlanner(vib, cap=2.5)
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    assert wc._hist_mu is None and wc._hist_lv is None
    t, x0 = _traj(anchor_x)
    S, g = wc._kl_backward(t, x0, None)
    assert float(S.detach().abs().max()) == 0.0 \
        and float(g.detach().abs().max()) == 0.0
    assert S.grad_fn is not None, "t=1 zero must stay graph-connected"
    assert wc._hist_mu.shape == (B, 1, Dz), "t=1 must still push the anchor"
    cg, disp, p2, rl = wc.guided_step(*_traj(anchor_x), None)
    assert disp is None and p2 is None
    assert float(cg.detach().abs().max()) == 0.0 \
        and float(rl.abs().max()) == 0.0
    print("[1] t=1 empty cloud: value+grad zero, graph-connected, "
          "anchor pushed OK")

    # ---------------- 2. t=2 bitwise atypical (J=1 fast path) ----------- #
    van = KLCostPlanner(vib, cap=2.5)
    configs = [
        dict(),                                        # adapt defaults
        dict(cloud_tau_mode="fixed"),                  # fixed 0.5
        dict(cloud_tau_mode="fixed", cloud_tau=0.05),  # fixed extreme
    ]
    for ci, kw in enumerate(configs):
        wcx = WalkCloudCostPlanner(vib, cap=2.5, **kw)
        for pl in (van, wcx):
            pl.set_current_obs(s_bar)
            pl.select_z(anchor_x.unsqueeze(1) * 1.0)
        xs = [anchor_x,                                # t=1 (anchor step)
              anchor_x + 0.05,                         # t=2 (cloud={anchor})
              anchor_x + 0.10]                         # t=3 (cloud grows)
        # perturbations are deliberately SMALL: pairwise KLs must stay
        # below kappa so the row cap cannot mask the t>=3 departure
        # (with KLs > 2.5 BOTH arms cap to -2.5 and zero their gradients
        # -- bitwise equal for the wrong reason).
        for step, x in enumerate(xs):
            t_v, x_v = _traj(x)
            t_g, x_g = _traj(x)
            cg_v, _, _, rl_v = van.guided_step(t_v, x_v, None)
            cg_g, _, _, rl_g = wcx.guided_step(t_g, x_g, None)
            if step < 2:
                assert torch.equal(cg_v, cg_g), \
                    f"cfg{ci} step {step}: cond_grad mismatch"
                assert torch.equal(rl_v, rl_g), \
                    f"cfg{ci} step {step}: row_losses mismatch"
            else:
                # t>=3: the cloud departs from the single-anchor reference
                # -- the mechanism must NOT stay bitwise atypical.
                assert not torch.allclose(rl_v, rl_g), \
                    f"cfg{ci} step 3: still identical to atypical?"
    print("[2] t=2 bitwise == atypical under any tau (J=1 fast path); "
          "t=1 zero-zero; t>=3 departs OK")

    # ---------------- 3. history push / reset semantics ------------------ #
    wc = WalkCloudCostPlanner(vib, cap=2.5)
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    lens = []
    for step in range(4):
        t, x0 = _traj(anchor_x + 0.2 * step)
        wc._kl_backward(t, x0, None)
        lens.append(wc._hist_mu.shape[1])
        wc.compute_loss((anchor_x + 0.1).unsqueeze(1) * 1.0, None)  # read-only
        assert wc._hist_mu.shape[1] == lens[-1], \
            "compute_loss must not append"
    assert lens == [1, 2, 3, 4], f"history lens {lens} != [1,2,3,4]"
    wc.select_z((anchor_x + 5.0).unsqueeze(1) * 1.0)   # new chunk -> reset
    assert wc._hist_mu is None and wc._hist_lv is None
    wc.reset()                                          # reused-instance path
    assert wc._hist_mu is None
    print("[3] history semantics: +1/step, select_z resets, "
          "compute_loss read-only OK")

    # ---------------- 4. soft-min formula + kernel gradient -------------- #
    wc = WalkCloudCostPlanner(vib, cap=2.5)             # adapt defaults
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    hist_mu, hist_lv = [], []
    for step in range(3):                               # build cloud H=3
        t, x0 = _traj(anchor_x + 0.3 * step)
        wc._kl_backward(t, x0, None)
    hist_mu = [wc._hist_mu[:, j].clone() for j in range(wc._hist_mu.shape[1])]
    hist_lv = [wc._hist_lv[:, j].clone() for j in range(wc._hist_lv.shape[1])]
    for step in range(2):                               # H=3 -> softmin path
        x = anchor_x + 0.5 * (step + 1) * torch.ones(B, Da)
        t, x0 = _traj(x)
        S, _ = wc._kl_backward(t, x0, None)
        with torch.no_grad():
            mu, logvar = vib.vib_enc(s_bar, x)
            kls = torch.stack([_kl_diag(mu, logvar, m0, lv0)
                               for m0, lv0 in zip(hist_mu, hist_lv)], dim=1)
            tau = torch.clamp(0.3 * (kls.amax(dim=1) - kls.amin(dim=1)),
                              min=0.02, max=0.5)
            ref = -tau * torch.log(
                torch.stack([torch.exp(-kls[b] / tau[b]).sum()
                             for b in range(B)]))
        assert torch.allclose(S, ref, atol=1e-6), \
            f"step {step}: S {S} vs ref {ref}"
        # independent loop pushed its own history view -- keep aligned
        hist_mu.append(mu.clone())
        hist_lv.append(logvar.clone())
    print("[4] vectorized soft-min (adapt tau) == independent loop OK")

    kl = torch.randn(B, 5, requires_grad=True)          # standalone kernel
    tau = torch.tensor([0.1, 0.25, 0.4])
    S, w = _cloud_softmin(kl, tau)
    gk = torch.autograd.grad(S.sum(), kl)[0]
    assert torch.allclose(gk, w, atol=1e-7), "dS/dKL != softmax weights"
    assert torch.allclose(w.sum(dim=1), torch.ones(B), atol=1e-6)
    print("[4b] kernel gradient dS/dKL_j == softmax(-KL/tau), "
          "weights sum to 1 OK")

    # ---------------- 5. trajectory gradient vs finite differences ------- #
    # fixed-tau mode: tau is a DETACHED schedule constant, so the FD
    # reference must treat it as frozen -- the injected gradient is the
    # tau-fixed partial (adapt mode recomputes tau from the perturbed KLs
    # and would differ from autograd by the dtau/dx term by design).
    wc = WalkCloudCostPlanner(vib, cap=2.5, cloud_tau_mode="fixed",
                              cloud_tau=0.1)
    s1 = s_bar[0:1]
    wc.set_current_obs(s1)
    wc.select_z(torch.randn(1, Da).unsqueeze(1) * 1.0)
    for d in (0.15, 0.35):                              # cloud H=2
        wc._kl_backward(*_traj(torch.full((1, Da), d)), None)
    x = torch.full((1, Da), 0.45)
    t, x0 = _traj(x)
    mu, logvar = vib.vib_enc(s1, _enc_forward(wc, x0))
    S = wc._cloud_rows(mu, logvar, x0)
    ana = float(torch.autograd.grad(S.sum(), t)[0][0, 0, 0])
    eps = 1e-4
    fd = []
    for sgn in (-1, +1):
        xp = x.clone()
        xp[0, 0] += sgn * eps
        with torch.no_grad():
            x0p = xp.unsqueeze(1) * 1.0
            m2, l2 = vib.vib_enc(s1, _enc_forward(wc, x0p))
            fd.append(float(wc._cloud_rows(m2, l2, x0p)[0]))
    num = (fd[1] - fd[0]) / (2 * eps)
    assert abs(ana - num) < 1e-3 * max(1.0, abs(num)), f"{ana} vs fd {num}"
    print(f"[5] autograd == central finite differences ({ana:.6f} vs "
          f"{num:.6f}) OK")

    # ---------------- 6. zero RNG ---------------------------------------- #
    st = torch.get_rng_state()
    wc = WalkCloudCostPlanner(vib, cap=2.5)
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    for step in range(4):
        wc.guided_step(*_traj(anchor_x + 0.25 * step), None)
    assert torch.equal(st, torch.get_rng_state()), \
        "guided_step walk consumed torch RNG"
    print("[6] full guided_step walk consumes zero RNG OK")

    # ---------------- 7. cloud_hist_max keeps the newest ----------------- #
    wc = WalkCloudCostPlanner(vib, cap=2.5, cloud_hist_max=2)
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    lens, posts = [], {}
    for step in range(1, 5):
        x = anchor_x + 0.2 * step
        wc._kl_backward(*_traj(x), None)
        lens.append(wc._hist_mu.shape[1])
        with torch.no_grad():
            posts[step] = wc.scout_vib.vib_enc(s_bar, x)[0].clone()
    assert lens == [1, 2, 2, 2], f"hist_max lens {lens} != [1,2,2,2]"
    assert torch.equal(wc._hist_mu[:, -1], posts[4]), "newest entry wrong"
    assert torch.equal(wc._hist_mu[:, 0], posts[3]), "oldest not dropped"
    print("[7] cloud_hist_max=2 keeps newest (x4 newest, x3 oldest kept) OK")

    # ---------------- 8. B-mismatch guard: zero + cloud restart --------- #
    wc = WalkCloudCostPlanner(vib, cap=2.5)
    wc.set_current_obs(s_bar)
    wc.select_z(anchor_x.unsqueeze(1) * 1.0)
    for step in range(3):                       # row-aligned B=3 cloud
        wc._kl_backward(*_traj(anchor_x + 0.2 * step), None)
    t2, x2 = _traj(anchor_x[:2] + 0.5)          # mismatched B=2 call
    S2, g2 = wc._kl_backward(t2, x2, None)
    assert float(S2.detach().abs().max()) == 0.0 \
        and float(g2.detach().abs().max()) == 0.0, "mismatch must not guide"
    assert wc._hist_mu.shape == (2, 1, 4), \
        f"mismatch must restart the cloud, got {tuple(wc._hist_mu.shape)}"
    t2b, x2b = _traj(anchor_x[:2] + 0.8)        # aligned follow-up: guides
    S2b, _ = wc._kl_backward(t2b, x2b, None)
    assert float(S2b.detach().abs().max()) > 0.0, "restart must resume guidance"
    print("[8] B-mismatch: graph-zero step + cloud restart (next call "
          "guides again) OK")

    print("\n[walkcloud-smoke] ALL CHECKS GREEN")


if __name__ == "__main__":
    main()
