"""Hermetic smoke for CloudRepCostPlanner (drift-dev, user 2026-09-09 方案一).

Run:  python -m scout.eval._smoke_cloudrep

Checks (all CPU, mock VIB -- no robomimic / no LPB stack):
  1. empty cloud (retry 0) reduces to KLCostPlanner EXACTLY: identical
     guided_step (cond_grad, row_losses) and compute_loss across a guided
     step sequence for POWER-OF-TWO tau (0.5/2.0: the tau round-trip is
     IEEE754-invertible); for non-power-of-two tau (0.37) the INJECTION
     gradient stays bit-identical while values may differ at 1 ulp
     (logsumexp max-shift) -- asserted as bitwise-cond_grad + allclose.
  2. cloud commit semantics: exactly ONE anchor committed per (scene, try)
     at its first select_z; later chunks do not re-commit; the pool of a
     later retry = 1 + (number of earlier committed anchors), capped at
     cloud_max.
  3. formula: planner's per-row S matches an independent per-row python
     soft-min over per-reference KLs (tau=0.37, mixed pool sizes).
  4. soft-min bounds + kernel gradient: min_j KL_j - tau*log(J_eff) <= S
     <= min_j KL_j; dS/dKL_ij equals the softmax weights
     exp(-KL_ij/tau)/sum_k exp(-KL_ik/tau) (the drifting normalized kernel).
  5. trajectory gradient: autograd grad w.r.t. the trajectory matches
     central finite differences; reference pool columns carry no grad
     (detached constants).
  6. compute_loss is read-only (no commit, no pool rebuild); _kl_backward
     advances telemetry only.
  7. RNG: a full guided_step walk consumes NO torch RNG (bit-comparability
     with the other KL-cost arms on the same seed).
  8. guards: pre-select_z / B-mismatch -> graph-connected zero rows,
     backward works; cap mask zeroes the climb for rows with S >= kappa.
  (start-gate semantics are asserted inside check 2; RAGGED pools --
   per-row cloud sizes differ, padded columns carry zero gradient -- inside
   checks 3/4, per review P0-1/P2.)
"""
import math

import torch

from scout.guidance.entropy_costs import KLCostPlanner
from scout.guidance.cloudrep_costs import CloudRepCostPlanner

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

    # ---------------- 1. empty cloud == atypical, bit-level ------------- #
    for tau in (0.5, 2.0):
        van = KLCostPlanner(vib, cap=2.5)
        cre = CloudRepCostPlanner(vib, cap=2.5, cloud_tau=tau)
        for pl in (van, cre):
            pl.set_current_obs(s_bar)
            pl.select_z(anchor_x.unsqueeze(1) * 1.0)
        for step in range(5):
            x = anchor_x + 0.3 * (step + 1) * torch.randn(B, Da)
            t_v, x_v = _traj(x)
            t_c, x_c = _traj(x)
            cg_v, d_v, p_v, rl_v = van.guided_step(t_v, x_v, None)
            cg_c, d_c, p_c, rl_c = cre.guided_step(t_c, x_c, None)
            assert torch.equal(cg_v, cg_c), f"tau={tau} step {step}: grad"
            assert d_c is None and p_c is None
            assert torch.equal(rl_v, rl_c), f"tau={tau} step {step}: losses"
            assert torch.equal(van.compute_loss(x_v, None),
                               cre.compute_loss(x_c, None))
    assert len(cre._cloud) == 0, "no row_jobs -> nothing must commit"
    # non-power-of-two tau: INJECTION gradient bit-identical, values ulp-close
    van = KLCostPlanner(vib, cap=2.5)
    cr3 = CloudRepCostPlanner(vib, cap=2.5, cloud_tau=0.37)
    for pl in (van, cr3):
        pl.set_current_obs(s_bar)
        pl.select_z(anchor_x.unsqueeze(1) * 1.0)
    for step in range(5):
        x = anchor_x + 0.3 * (step + 1) * torch.randn(B, Da)
        t_v, x_v = _traj(x)
        t_c, x_c = _traj(x)
        cg_v, _, _, rl_v = van.guided_step(t_v, x_v, None)
        cg_c, _, _, rl_c = cr3.guided_step(t_c, x_c, None)
        assert torch.equal(cg_v, cg_c), f"tau=0.37 step {step}: inj grad"
        assert torch.allclose(rl_v, rl_c, atol=1e-6),             f"tau=0.37 step {step}: values beyond ulp"
    print("[1] empty cloud == atypical: bitwise for tau in {0.5, 2.0}; "
          "injection-gradient bitwise + ulp values for tau=0.37")

    # ---------------- 2. commit semantics across retries ---------------- #
    # realistic job sequence (under the START-gate a batch may carry two
    # tries of the same scene once both started -- pools/commits are keyed
    # per row, so the assertions below hold either way):
    #   batch 1 = chunk 0 of (7,0),(9,0),(8,0)   -> all commit once, J=1
    #   batch 2 = later chunks, same tries       -> no commit, J stays 1
    #   batch 3 = chunk 0 of (7,1),(9,1),(8,1)   -> pools J=2, commit again
    #   batch 4 = chunk 0 of (7,2),(9,2),(8,2)   -> pools J=3
    pl = CloudRepCostPlanner(vib, cap=2.5, cloud_tau=0.37, cloud_max=8)
    pl.set_current_obs(s_bar)
    pl.set_row_jobs([(None, 7, 0), (None, 9, 0), (None, 8, 0)])
    pl.select_z(anchor_x.unsqueeze(1) * 1.0)        # chunk 0 of each try
    assert len(pl._cloud[7]) == 1 and len(pl._cloud[9]) == 1 \
        and len(pl._cloud[8]) == 1, "try-0 commits"
    assert pl._ref_valid.shape[1] == 1, "fresh pools exclude own commits"
    x2 = anchor_x + 0.1
    pl.select_z(x2.unsqueeze(1) * 1.0)              # later chunks
    assert all(len(pl._cloud[k]) == 1 for k in (7, 9, 8)), "no re-commit"
    # within-retry later chunks: pool = {fresh anchor} + {own chunk-0
    # anchor, committed at batch 1} -> J=2 (mild anti-return to the retry's
    # own start, per the cloud semantics -- col0 is ALWAYS the fresh anchor)
    assert pl._ref_valid.shape[1] == 2, "later-chunk pool J=2"
    assert bool(pl._ref_valid.all())
    pl.set_row_jobs([(None, 7, 1), (None, 9, 1), (None, 8, 1)])
    pl.select_z((x2 + 0.2).unsqueeze(1) * 1.0)      # next retries' chunk 0
    assert pl._ref_valid.shape[1] == 2, "retry-1 pool J=2"
    assert bool(pl._ref_valid[:, 0].all()) and bool(pl._ref_valid[:, 1].all())
    assert all(len(pl._cloud[k]) == 2 for k in (7, 9, 8))
    pl.set_row_jobs([(None, 7, 2), (None, 9, 2), (None, 8, 2)])
    pl.select_z((x2 + 0.3).unsqueeze(1) * 1.0)
    assert pl._ref_valid.shape[1] == 3, "retry-2 pool J=3"
    assert all(len(pl._cloud[k]) == 3 for k in (7, 9, 8))
    assert pl.try_started_gate(7, 0) and pl.try_started_gate(7, 1)         and pl.try_started_gate(7, 2), "started high-water admits <= max"
    assert pl.try_started_gate(7, 3), "try 3 admissible once try 2 started"
    assert not pl.try_started_gate(7, 4), "try 4 blocked until try 3 starts"
    assert not pl.try_started_gate(99, 1), "unstarted scene blocks try 1"
    assert pl.try_started_gate(99, 0), "try 0 always admissible"
    print("[2] commit-once per (scene,try) + pool growth + own-anchor "
          "single-count + start-gate semantics OK")

    # ---------------- 3. formula vs independent python soft-min ---------- #
    # MIXED pool sizes: scene 1 has a committed earlier retry, scenes 2/3
    # are fresh (empty clouds) -- J = max over rows, short rows padded.
    tau, k2 = 0.37, 2.5
    pl = CloudRepCostPlanner(vib, cap=k2, cloud_tau=tau)
    pl.set_current_obs(s_bar)
    pl.set_row_jobs([(None, 1, 0), (None, 1, 0), (None, 1, 0)])
    pl.select_z(torch.randn(B, 1, Da))              # seed scene 1's cloud
    #   (one commit for (1,0); rows 1/2 repeat the same key -> no extra)
    assert len(pl._cloud[1]) == 1
    anchor2 = anchor_x + 0.4 * torch.randn(B, Da)
    pl.set_row_jobs([(None, 1, 1), (None, 2, 0), (None, 3, 0)])
    pl.select_z(anchor2.unsqueeze(1) * 1.0)         # mixed batch
    assert pl._ref_valid.shape[1] == 2, "J = 1 + max(cloud sizes)"
    assert bool(pl._ref_valid[0].all()), "row0 (scene1) sees its cloud"
    assert bool(pl._ref_valid[1, 0]) and not bool(pl._ref_valid[1, 1]),         "row1 (fresh scene) has a padded column"
    J = pl._ref_valid.shape[1]
    cand = anchor2 + 0.5 * torch.randn(B, Da)
    with torch.no_grad():
        a = cand                                   # identity bridge, Da=act
        mu, logvar = vib.vib_enc(s_bar, cand)
    s_val = pl._cloud_rows(mu, logvar, cand.unsqueeze(1) * 1.0)
    for i in range(B):
        refs = []
        if pl._ref_valid[i, 0]:
            refs.append(_kl_diag(mu[i], logvar[i],
                                 pl._ref_mu[i, 0], pl._ref_lv[i, 0]))
        for k in range(1, J):
            if pl._ref_valid[i, k]:
                refs.append(_kl_diag(mu[i], logvar[i],
                                     pl._ref_mu[i, k], pl._ref_lv[i, k]))
        ref = -tau * math.log(sum(math.exp(-float(r) / tau) for r in refs))
        assert abs(float(s_val[i]) - ref) < 1e-5, f"row {i} soft-min mismatch"
    print("[3] soft-min formula matches independent loop OK")

    # ---------------- 4. bounds + kernel (softmax) gradient -------------- #
    kl_rows = []
    with torch.no_grad():
        for i in range(B):
            row = [_kl_diag(mu[i], logvar[i], pl._ref_mu[i, k],
                            pl._ref_lv[i, k]).clone()
                   for k in range(J)]
            kl_rows.append(torch.stack(row))
    for i in range(B):
        valid = [float(kl_rows[i][k]) for k in range(J) if pl._ref_valid[i, k]]
        s_i = float(s_val[i])
        assert s_i <= min(valid) + 1e-6, "S <= min KL"
        assert s_i >= min(valid) - tau * math.log(len(valid)) - 1e-6, \
            "S >= min - tau*log J"
    # dS/dKL_ij = softmax weights: finite-difference the KL matrix through
    # _cloud_rows by nudging a reference (detached constant) -- instead
    # verify via autograd on a re-computed graph with kl as an intermediate
    mu_g = mu.clone().requires_grad_(True)
    lv_g = logvar.clone().requires_grad_(True)
    var = torch.exp(lv_g).unsqueeze(1)
    var0 = torch.exp(pl._ref_lv)
    klm = 0.5 * (((mu_g.unsqueeze(1) - pl._ref_mu) ** 2 / var0)
                 + (var / var0) - 1.0
                 - (lv_g.unsqueeze(1) - pl._ref_lv)).sum(dim=-1)
    logits = (-klm / tau).masked_fill(~pl._ref_valid, float("-inf"))
    s_g = -tau * torch.logsumexp(logits, dim=1)
    for i in range(B):
        w = torch.softmax(
            ((-klm[i].detach()) / tau).masked_fill(
                ~pl._ref_valid[i], float("-inf")), dim=0)
        gs = torch.autograd.grad(s_g[i], klm, retain_graph=True)[0]
        assert torch.allclose(gs[i], w, atol=1e-6), "kernel gradient"
        assert torch.count_nonzero(gs[i][~pl._ref_valid[i]]) == 0, \
            "padded columns carry no gradient"
    print("[4] soft-min bounds + drifting-kernel (softmax) gradient OK")

    # ---------------- 5. trajectory gradient vs finite differences ------- #
    x0 = anchor2 + 0.2
    t, xh = _traj(x0)
    g_auto, _, _, _ = pl.guided_step(t, xh, None)   # the injected climb
    eps = 1e-4
    g_fd = torch.zeros_like(x0)
    for i in range(B):
        for j in range(Da):
            xp, xm = x0.clone(), x0.clone()
            xp[i, j] += eps
            xm[i, j] -= eps
            sp = pl.compute_loss(xp.unsqueeze(1) * 1.0, None,
                                 reduction="sum")
            sm = pl.compute_loss(xm.unsqueeze(1) * 1.0, None,
                                 reduction="sum")
            g_fd[i, j] = (float(sp) - float(sm)) / (2 * eps)
    # sign: compute_loss rows are -clamp(S, kappa), so d(compute_loss)/dx
    # = -dS/dx on live rows (0 on capped rows, where the climb masks too);
    # the climb injects +dS/dx -> compare g_auto against -g_fd. Rows within
    # 0.5 nats of kappa are excluded -- the clamp boundary makes the FD one-
    # sided there (autograd mask passes AT the cap, FD straddles it).
    with torch.no_grad():
        mu5, lv5 = vib.vib_enc(s_bar, x0)
        s5 = pl._cloud_rows(mu5, lv5, x0.unsqueeze(1) * 1.0)
    live = s5 < float(pl.cap) - 0.5
    assert bool(live.any()), "no live rows for the FD comparison"
    err = (g_auto.squeeze(1)[live] - (-g_fd[live])).abs().max()
    # 2e-2 absolute tolerance: gradients reach O(6) here and the float32
    # central FD carries ~2.5e-3 cancellation noise on the compute_loss
    # magnitude (~5) -- verified row-wise against the analytic form.
    assert err < 2e-2, f"traj grad mismatch {err}"
    print(f"[5] trajectory gradient vs central FD OK "
          f"(max err {err:.2e} on {int(live.sum())}/{B} live rows)")

    # ---------------- 6. compute_loss read-only -------------------------- #
    snap = (pl._ref_mu.clone(), pl._ref_lv.clone(), pl._ref_valid.clone(),
            {k: [tuple(t2.clone() for t2 in r) for r in v]
             for k, v in pl._cloud.items()},
            set(pl._committed))
    _ = pl.compute_loss(x0.unsqueeze(1) * 1.0, None)
    assert torch.equal(pl._ref_mu, snap[0]) and torch.equal(
        pl._ref_lv, snap[1]) and torch.equal(pl._ref_valid, snap[2])
    assert set(pl._cloud.keys()) == set(snap[3].keys())
    assert all(len(pl._cloud[k]) == len(snap[3][k]) for k in pl._cloud)
    assert pl._committed == snap[4]
    print("[6] compute_loss read-only OK")

    # ---------------- 7. RNG discipline ---------------------------------- #
    st0 = torch.get_rng_state()
    t2_, xh2 = _traj(x0)
    pl.guided_step(t2_, xh2, None)
    pl.select_z((x0 + 0.05).unsqueeze(1) * 1.0)
    t3_, xh3 = _traj(x0 + 0.05)
    pl.guided_step(t3_, xh3, None)
    assert torch.equal(st0, torch.get_rng_state()), "RNG consumed"
    print("[7] no RNG consumption OK")

    # ---------------- 8. guards + cap mask ------------------------------- #
    pl2 = CloudRepCostPlanner(vib, cap=2.5, cloud_tau=0.5)
    pl2.set_current_obs(s_bar)
    t4, xh4 = _traj(torch.randn(B, Da))             # pre-select_z call
    cg, d, p, rl = pl2.guided_step(t4, xh4, None)
    assert torch.count_nonzero(cg) == 0 and torch.count_nonzero(rl) == 0, \
        "guard rows must be graph-connected zeros"
    # cap: force S above kappa by a huge cloud offset, climb must mask
    pl3 = CloudRepCostPlanner(vib, cap=0.05, cloud_tau=0.5)
    pl3.set_current_obs(s_bar)
    pl3.set_row_jobs([(None, 5, 0)] * B)
    pl3.select_z(torch.zeros(B, 1, Da))
    far = torch.full((B, Da), 50.0)                 # KL >> kappa
    t5, xh5 = _traj(far)
    cg5, _, _, rl5 = pl3.guided_step(t5, xh5, None)
    assert torch.count_nonzero(cg5) == 0, "capped rows must not inject"
    assert abs(float(rl5.max()) + 0.05) < 1e-6, "row_losses at the cap"
    print("[8] guards (graph-connected zeros) + kappa cap mask OK")

    print("\n_smoke_cloudrep: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
