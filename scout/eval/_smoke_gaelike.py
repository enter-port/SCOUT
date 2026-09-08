"""Hermetic smoke for GAELikeCostPlanner (GAElike-dev, user 2026-09-08).

Run:  python -m scout.eval._smoke_gaelike

Checks (all CPU, mock VIB -- no robomimic / no LPB stack):
  1. gamma=0 + norm=1 reduces to KLCostPlanner EXACTLY: identical guided_step
     (cond_grad, row_losses) and compute_loss across a guided-step sequence.
  2. history semantics: H grows by 1 per _kl_backward AFTER the anchor step
     (the anchor step does not duplicate itself); select_z resets H to 1.
  3. formula: planner's per-row kl matches an independent per-term python
     loop (gamma=0.5, cap 2.5, growing history).
  4. per-term cap + normalization: weighted normalized sum <= kappa even
     with large divergences; gradients stay finite.
  5. gradient: autograd grad w.r.t. the trajectory matches central finite
     differences (evaluated at the pre-push history snapshot); history
     columns carry no grad (detached constants).
  6. compute_loss is read-only (no history append); _kl_backward advances.
  7. RNG: a full guided_step walk consumes NO torch RNG (bit-comparability
     with the other KL-cost arms on the same seed).
  8. pre-select_z / B-mismatch guard: graph-connected zero rows, backward
     works.
"""
import torch

from scout.guidance.entropy_costs import KLCostPlanner
from scout.guidance.gae_costs import GAELikeCostPlanner

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

    # ---------------- 1. gamma=0 == atypical, bit-level ------------------- #
    van = KLCostPlanner(vib, cap=2.5)
    gae0 = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.0, gae_normalize=True)
    for pl in (van, gae0):
        pl.set_current_obs(s_bar)
        pl.select_z(anchor_x.unsqueeze(1) * 1.0)
    for step in range(5):
        x = anchor_x + 0.3 * (step + 1) * torch.randn(B, Da)
        t_v, x_v = _traj(x)
        t_g, x_g = _traj(x)
        cg_v, d_v, p_v, rl_v = van.guided_step(t_v, x_v, None)
        cg_g, d_g, p_g, rl_g = gae0.guided_step(t_g, x_g, None)
        assert torch.equal(cg_v, cg_g), f"step {step}: cond_grad mismatch"
        assert d_g is None and p_g is None
        assert torch.equal(rl_v, rl_g), f"step {step}: row_losses mismatch"
        assert torch.equal(van.compute_loss(x_v, None),
                           gae0.compute_loss(x_g, None))
    print("[1] gamma=0 == atypical: guided_step + compute_loss bitwise OK")

    # ---------------- 2. history push / reset semantics ------------------- #
    gae = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5, gae_normalize=False)
    gae.set_current_obs(s_bar)
    gae.select_z(anchor_x.unsqueeze(1) * 1.0)
    assert gae._hist_mu.shape == (B, 1, Dz) and gae._gae_fresh is True
    lens = []
    for step in range(4):
        t, x0 = _traj(anchor_x + 0.2 * step)
        gae._kl_backward(t, x0, None)
        lens.append(gae._hist_mu.shape[1])
    assert lens == [1, 2, 3, 4], f"history lens {lens} != [1,2,3,4]"
    gae.select_z((anchor_x + 5.0).unsqueeze(1) * 1.0)   # new chunk -> reset
    assert gae._hist_mu.shape[1] == 1 and gae._gae_fresh is True
    print("[2] history push/reset semantics OK (fresh step no dup, +1/step, "
          "select_z resets)")

    # ---------------- 3. formula vs independent loop ---------------------- #
    gae = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5, gae_normalize=True)
    gae.set_current_obs(s_bar)
    gae.select_z(anchor_x.unsqueeze(1) * 1.0)
    hist_mu = [gae._hist_mu[:, 0].clone()]
    hist_lv = [gae._hist_lv[:, 0].clone()]
    for step in range(3):
        x = anchor_x + 0.4 * (step + 1) * torch.ones(B, Da)
        t, x0 = _traj(x)
        kl, _ = gae._kl_backward(t, x0, None)
        with torch.no_grad():
            mu, logvar = vib.vib_enc(s_bar, x)
            ref = torch.zeros(B)
            wsum = 0.0
            for k, (m0, lv0) in enumerate(zip(hist_mu, hist_lv)):
                w = 0.5 ** k
                ref += w * torch.clamp(_kl_diag(mu, logvar, m0, lv0),
                                       max=2.5)
                wsum += w
            ref /= wsum
        assert torch.allclose(kl, ref, atol=1e-6), \
            f"step {step}: kl {kl} vs ref {ref}"
        hist_mu.append(mu.clone())
        hist_lv.append(logvar.clone())
    print("[3] vectorized formula == independent per-term loop OK")

    # ---------------- 4. per-term cap + normalization envelope ------------ #
    gae = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.9, gae_normalize=True)
    gae.set_current_obs(s_bar)
    gae.select_z(torch.zeros(B, Da).unsqueeze(1) * 1.0)
    for step in range(6):
        t, x0 = _traj(torch.full((B, Da), 3.0 * (step + 1)))  # large KLs
        kl, g = gae._kl_backward(t, x0, None)
        assert float(kl.detach().max()) <= 2.5 + 1e-5, f"envelope broken: {kl}"
        assert torch.isfinite(g).all()
    print("[4] per-term kappa cap + normalized envelope <= kappa OK")

    # ---------------- 5. autograd vs finite differences (B=1) ------------- #
    # small neighborhood so every pairwise KL sits BELOW the cap (a saturated
    # row has a masked/zero gradient and would make the check vacuous).
    g1 = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5, gae_normalize=True)
    s1 = s_bar[0:1]
    g1.set_current_obs(s1)
    g1.select_z(torch.randn(1, Da).unsqueeze(1) * 1.0)
    for d in (0.02, 0.05):                          # history H=3, tiny moves
        t, x0 = _traj(torch.full((1, Da), d))
        g1._kl_backward(t, x0, None)
    assert g1._hist_mu.shape[0] == 1
    x = torch.full((1, Da), 0.07)
    snap_mu, snap_lv = g1._hist_mu.clone(), g1._hist_lv.clone()
    t, x0 = _traj(x)
    kl, g = g1._kl_backward(t, x0, None)
    assert float(kl.detach().max()) < 2.5, "FD row saturated -- shrink the neighborhood"
    ana = float(g[0, 0, 0])                          # d kl_row0 / d traj0
    g1._hist_mu, g1._hist_lv = snap_mu, snap_lv      # undo the push for FD
    eps = 1e-4
    fd = []
    for s in (-1, +1):
        t2 = x.unsqueeze(1).clone()
        t2[0, 0, 0] += s * eps
        with torch.no_grad():
            mu, logvar = vib.vib_enc(s1, t2[:, 0, :])
            _, kl2 = g1._gae_rows(mu, logvar, t2 * 1.0)
        fd.append(float(kl2[0]))
    num = (fd[1] - fd[0]) / (2 * eps)
    assert abs(ana - num) < 1e-3 * max(1.0, abs(num)), f"grad {ana} vs fd {num}"
    assert (not g1._hist_mu.requires_grad
            and not g1._hist_lv.requires_grad)
    print(f"[5] autograd == finite differences ({ana:.6f} vs {num:.6f}); "
          "history detached OK")

    # ---------------- 6. compute_loss read-only --------------------------- #
    h_before = g1._hist_mu.shape[1]
    t, x0 = _traj(torch.randn(1, Da))
    g1.compute_loss(x0, None, reduction="sum")
    assert g1._hist_mu.shape[1] == h_before, "compute_loss must not append"
    print("[6] compute_loss read-only OK")

    # ---------------- 7. RNG non-consumption ------------------------------ #
    g2 = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.9)
    g2.set_current_obs(s_bar)
    g2.select_z(torch.randn(B, Da).unsqueeze(1) * 1.0)
    xs = [torch.randn(B, Da) for _ in range(3)]      # pre-draw ALL inputs
    st = torch.get_rng_state()                       # snapshot AFTER setup
    for x in xs:
        t, x0 = _traj(x)
        g2.guided_step(t, x0, None)
    assert torch.equal(st, torch.get_rng_state()), "RNG was consumed"
    print("[7] guided_step walk consumes no RNG OK")

    # ---------------- 8. pre-select_z / B-mismatch guards ------------------ #
    g3 = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.9)
    g3.set_current_obs(s_bar)
    t, x0 = _traj(torch.randn(B + 1, Da))            # NO history -> zeros
    kl, g = g3._kl_backward(t, x0, None)
    assert torch.equal(kl, torch.zeros(B + 1)) and torch.equal(
        g, torch.zeros_like(g))
    assert g3._hist_mu is None, "push must no-op without history"
    # history EXISTS but the batch row count changed mid-flight -> zeros,
    # no buffer corruption (review P2-4)
    g3.select_z(torch.randn(B, Da).unsqueeze(1) * 1.0)
    t, x0 = _traj(torch.randn(B + 1, Da))
    kl, g = g3._kl_backward(t, x0, None)
    assert torch.equal(kl, torch.zeros(B + 1)) and torch.equal(
        g, torch.zeros_like(g))
    assert g3._hist_mu.shape[:2] == (B, 1), "buffer corrupted by B mismatch"
    print("[8] pre-select_z / B-mismatch guards OK (zeros, no corruption)")

    # ---------------- 9. add-mode: lambda=0 / gamma=0 == atypical --------- #
    for kw in ({"gae_agg": "add", "gae_hist_weight": 0.0, "gae_gamma": 0.9},
               {"gae_agg": "add", "gae_hist_weight": 0.15, "gae_gamma": 0.0}):
        van2 = KLCostPlanner(vib, cap=2.5)
        gad = GAELikeCostPlanner(vib, cap=2.5, **kw)
        for pl in (van2, gad):
            pl.set_current_obs(s_bar)
            pl.select_z(anchor_x.unsqueeze(1) * 1.0)
        for step in range(4):
            x = anchor_x + 0.3 * (step + 1) * torch.randn(B, Da)
            t_v, x_v = _traj(x)
            t_g, x_g = _traj(x)
            cg_v, _, _, rl_v = van2.guided_step(t_v, x_v, None)
            cg_g, _, _, rl_g = gad.guided_step(t_g, x_g, None)
            assert torch.equal(cg_v, cg_g), f"{kw} step {step}: grad mismatch"
            assert torch.equal(rl_v, rl_g), f"{kw} step {step}: rl mismatch"
    print("[9] add-mode lambda=0 / gamma=0 == atypical bitwise OK")

    # ---------------- 10. add-mode: anchor trust region + value form ------- #
    ga = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5,
                            gae_agg="add", gae_hist_weight=0.3)
    ga.set_current_obs(s_bar)
    ga.select_z(torch.zeros(B, Da).unsqueeze(1) * 1.0)
    # walk far from the anchor: anchor KL saturates at kappa, history terms
    # are live -> the row must die (cond_grad exactly 0) because the ANCHOR
    # is the trust region in add mode
    for step in range(4):
        t, x0 = _traj(torch.full((B, Da), 4.0 * (step + 1)))
        kl, g = ga._kl_backward(t, x0, None)
    assert float(kl.detach().max()) <= 2.5 + 1e-5
    cg, _, _, rl = ga.guided_step(*_traj(torch.full((B, Da), 20.0)), None)
    assert torch.equal(cg, torch.zeros_like(cg)), \
        "saturated anchor must kill the row in add mode"
    # value form: anchor + lambda*sum_{k>=1} g^k*min(KL_k,kappa)
    ga2 = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5,
                             gae_agg="add", gae_hist_weight=0.3)
    ga2.set_current_obs(s_bar)
    ga2.select_z(torch.zeros(B, Da).unsqueeze(1) * 1.0)
    hist_mu = [ga2._hist_mu[:, 0].clone()]
    hist_lv = [ga2._hist_lv[:, 0].clone()]
    for step in range(3):
        x = torch.full((B, Da), 0.4 * (step + 1))
        t, x0 = _traj(x)
        anchor, agg = ga2._gae_rows(*_enc(ga2, x, s_bar), x0)
        with torch.no_grad():
            mu, logvar = vib.vib_enc(s_bar, x)
            ref_a = torch.clamp(_kl_diag(mu, logvar, hist_mu[0], hist_lv[0]),
                                max=2.5)
            ref_h = torch.zeros(B)
            for k, (m0, lv0) in enumerate(zip(hist_mu, hist_lv)):
                if k == 0:
                    continue
                ref_h += (0.5 ** k) * torch.clamp(
                    _kl_diag(mu, logvar, m0, lv0), max=2.5)
            ref = ref_a + 0.3 * ref_h
        assert torch.allclose(agg, ref, atol=1e-6), f"add form step {step}"
        hist_mu.append(mu.clone())
        hist_lv.append(logvar.clone())
        ga2._push_history(mu, logvar)          # advance like _kl_backward
        ga2._gae_fresh = False
    print("[10] add-mode anchor trust region + value form OK")

    # ---------------- 11. add-mode finite differences ---------------------- #
    gb = GAELikeCostPlanner(vib, cap=2.5, gae_gamma=0.5,
                            gae_agg="add", gae_hist_weight=0.3)
    gb.set_current_obs(s1 := s_bar[0:1])
    gb.select_z(torch.randn(1, Da).unsqueeze(1) * 1.0)
    for d in (0.02, 0.05):
        t, x0 = _traj(torch.full((1, Da), d))
        anchor, agg = gb._gae_rows(*_enc(gb, t[:, 0, :], s1), x0)
        gb._push_history(*_enc(gb, t[:, 0, :], s1))
        gb._gae_fresh = False
    x = torch.full((1, Da), 0.07)
    t, x0 = _traj(x)
    mu, logvar = _enc(gb, t[:, 0, :], s1)
    anchor, agg = gb._gae_rows(mu, logvar, x0)
    g_ = torch.autograd.grad(agg.sum(), t)[0]
    ana = float(g_[0, 0, 0])
    eps = 1e-4
    fd = []
    for sgn in (-1, +1):
        t2 = x.unsqueeze(1).clone()
        t2[0, 0, 0] += sgn * eps
        with torch.no_grad():
            m2, l2 = _enc(gb, t2[:, 0, :], s1)
            _, a2 = gb._gae_rows(m2, l2, t2 * 1.0)
        fd.append(float(a2[0]))
    num = (fd[1] - fd[0]) / (2 * eps)
    assert abs(ana - num) < 1e-3 * max(1.0, abs(num)), f"{ana} vs fd {num}"
    print(f"[11] add-mode autograd == finite differences ({ana:.6f} vs "
          f"{num:.6f}) OK")

    print("\n[gae-smoke] ALL CHECKS GREEN")


def _enc(planner, a, s_bar):
    mu, logvar = planner.scout_vib.vib_enc(s_bar, a)
    return mu, logvar


if __name__ == "__main__":
    main()
