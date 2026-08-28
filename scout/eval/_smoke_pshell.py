"""Hermetic smoke for rand_pshell (persistent anchored shell, 2026-08-28).

Checks (per the campaign contract):
  * refresh=chunk is BITWISE-EQUAL to ShellTargetCostPlanner (方案A):
    cost AND gradient, both reductions, over a 2-chunk / 2-row trajectory
    with changing intents (the parent's own per-chunk re-capture path);
  * refresh=retry: the anchor (mu^0, logvar^0) used by the SECOND chunk of
    the same (init, try) is bit-identical to the FIRST chunk's capture
    (chunk mode differs -> its second-chunk cost changes when the intent
    moves, retry mode's target mu* stays CONSTANT across chunks -- checked
    against an independent closed-form KL built from the frozen anchor);
  * a new retry (different try_idx) gets a FRESH anchor (its own first
    chunk), while the old retry's anchor is untouched;
  * refresh=w5: captures at chunks 1 and 6, NOT at 2-5 (anchor identity
    checked chunk by chunk);
  * no-job rows: graph-connected zero (cost 0, zero grad), like 方案A;
  * compute_loss before any capture falls back to THIS chunk's own
    baseline (defensive path, == 方案A semantics for that row);
  * reductions: sum == mean * B; bogus reduction raises;
  * make_planner: ek parsing (rand_anchor_refresh / shell_kappa /
    shell_seed), invalid refresh raises;
  * anchors are detached cpu tensors (no graph retention across chunks).

Run:  python -m scout.eval._smoke_pshell
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner
from scout.guidance.rand_costs.pshell import PShellCostPlanner, make_planner

Ds, Da, Dz = 6, 5, 4          # s_bar dim / action dim / style dim
torch.manual_seed(0)
W = torch.randn(Dz, Da)
Wv = 0.3 * torch.randn(Dz, Da)


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                                # (B, Dz)
        logvar = -1.0 + 0.2 * (a @ Wv.T)            # a-dependent sigma head
        return mu, logvar


class MockVib:
    style_dim = Dz

    def __init__(self):
        self.vib_enc = MockEncNet()

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def encode(self, obs):
        if isinstance(obs, dict):                    # E_s-format -> s_bar
            return torch.as_tensor(obs["proprio"]).reshape(-1, Ds)
        return obs                                   # s_bar passthrough


def _x(a):
    return a.unsqueeze(1).clone().requires_grad_(True)


def _chunk(pl, jobs, x0, s_bar):
    """One replan chunk: set_row_jobs -> set_current_obs -> select_z."""
    if jobs is not None:
        pl.set_row_jobs(jobs)
    pl.set_current_obs(s_bar)
    pl.select_z(x0.unsqueeze(1), s_bar)


def _closed_form(pl, key, x0, xc, s_bar, cap_none=True):
    """Independent KL(q_a||q*) for one row from the planner's OWN anchor
    tensors (target built by hand, not through compute_loss)."""
    with torch.no_grad():
        mu0 = pl._anchor_mu[key].float()
        lv0 = pl._anchor_lv[key].float()
        mu, logvar = MockVib().vib_enc(s_bar, xc)
        # reproduce 方案A's deterministic u
        rng = np.random.default_rng([pl.shell_seed, int(key[0]), int(key[1])])
        v = rng.standard_normal(Dz)
        u = torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float32)
        tgt = mu0 + (2.0 * pl.shell_kappa) ** 0.5 * torch.exp(0.5 * lv0) * u
        kl = 0.5 * (((mu[0] - tgt) ** 2 / torch.exp(lv0))
                    + torch.exp(logvar[0]) / torch.exp(lv0) - 1.0
                    - (logvar[0] - lv0)).sum()
    return float(kl)


def main():
    s_bar = torch.randn(2, Ds)
    x0_c1 = torch.randn(2, Da)                       # chunk-1 intents
    x0_c2 = x0_c1 + 0.4 * torch.randn(2, Da)         # chunk-2 intents (moved)
    xc = x0_c1 + 0.2 * torch.randn(2, Da)            # candidates
    jobs2 = [(None, 7, 1), (None, 8, 2)]              # raw replan jobs
    keys2 = [(7, 1), (8, 2)]                          # anchor-dict keys
    kappa = 2.5

    # ---- 1) refresh=chunk == ShellTargetCostPlanner, bitwise -------------- #
    pl_c = PShellCostPlanner(MockVib(), anchor_refresh="chunk",
                             shell_kappa=kappa, shell_seed=42)
    ref = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa, shell_seed=42)
    for red in ("sum", "mean"):
        pl2 = PShellCostPlanner(MockVib(), anchor_refresh="chunk",
                                shell_kappa=kappa, shell_seed=42)
        r2 = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa,
                                    shell_seed=42)
        for chunk, (x0, sb) in enumerate([(x0_c1, s_bar),
                                          (x0_c2, s_bar + 0.1)]):
            for plx, rfx in ((pl2, r2), ):
                plx.set_row_jobs(jobs2)
                rfx.set_row_jobs(jobs2)
                plx.set_current_obs(sb)
                rfx.set_current_obs(sb)
                plx.select_z(x0.unsqueeze(1), sb)
                rfx.select_z(x0.unsqueeze(1), sb)
            c_p = pl2.compute_loss(_x(xc), sb, reduction=red)
            c_r = r2.compute_loss(_x(xc), sb, reduction=red)
            assert torch.equal(c_p, c_r), (red, chunk, float(c_p), float(c_r))
            xg_p, xg_r = _x(xc), _x(xc)
            pl2.compute_loss(xg_p, sb, reduction="sum").backward()
            r2.compute_loss(xg_r, sb, reduction="sum").backward()
            assert torch.equal(xg_p.grad, xg_r.grad), (red, chunk, "grads")
        assert pl2._anchor_mu == {} and pl2._anchor_lv == {}, \
            "chunk mode must never persist anchors"
    _ = pl_c  # (kept for symmetry; pl2/r2 above carry the real assertions)

    # ---- 2) refresh=retry: anchor frozen at the FIRST chunk ---------------- #
    pl_r = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                             shell_kappa=kappa, shell_seed=42)
    _chunk(pl_r, jobs2, x0_c1, s_bar)                # chunk 1
    a1 = {k: (pl_r._anchor_mu[k].clone(), pl_r._anchor_lv[k].clone())
          for k in keys2}
    with torch.no_grad():
        # SAME batch shape as the planner's forward (batch-2), so the check
        # is bitwise (batch-1 vs batch-2 matmuls may differ by 1 ulp).
        mu0_c1_all = MockVib().vib_enc(s_bar, x0_c1)[0]
        mu0_c1 = {j: mu0_c1_all[i].clone() for i, j in enumerate(keys2)}
    for k in keys2:
        assert torch.equal(a1[k][0], mu0_c1[k].cpu()), \
            "anchor must be chunk-1's own mu^0"
    _chunk(pl_r, jobs2, x0_c2, s_bar + 0.1)          # chunk 2 (intent moved)
    for k in keys2:
        assert torch.equal(pl_r._anchor_mu[k], a1[k][0]), \
            "retry anchor must be bit-stable across chunks"
        assert torch.equal(pl_r._anchor_lv[k], a1[k][1])
    # _base_mu still refreshes per chunk (parent semantics intact)
    with torch.no_grad():
        mu0_c2 = MockVib().vib_enc(s_bar + 0.1, x0_c2)[0]
    assert torch.allclose(pl_r._base_mu[0], mu0_c2[0]), \
        "per-chunk baseline must still refresh (parent hook intact)"

    # second-chunk cost == closed form from the FROZEN anchor
    got = float(pl_r.compute_loss(_x(xc), s_bar + 0.1,
                                  reduction="sum").detach())
    want = sum(_closed_form(pl_r, k, x0_c2, xc[i:i + 1], (s_bar + 0.1)[i:i + 1])
               for i, k in enumerate(keys2))
    assert abs(got - want) < 1e-5, (got, want)

    # chunk-mode cost on the SAME moved chunk differs (anchor chase vs frozen)
    pl_chk = PShellCostPlanner(MockVib(), anchor_refresh="chunk",
                               shell_kappa=kappa, shell_seed=42)
    _chunk(pl_chk, jobs2, x0_c1, s_bar)
    _chunk(pl_chk, jobs2, x0_c2, s_bar + 0.1)
    c_chk = float(pl_chk.compute_loss(_x(xc), s_bar + 0.1,
                                      reduction="sum").detach())
    assert abs(c_chk - got) > 1e-4, \
        "chunk vs retry must differ once the intent moves"

    # mu* constant across chunks for retry mode (per key)
    def _mustar(pl, key):
        m0 = pl._anchor_mu[key].float()
        lv0 = pl._anchor_lv[key].float()
        rng = np.random.default_rng([pl.shell_seed,
                                     int(key[0]), int(key[1])])
        v = rng.standard_normal(Dz)
        u = torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float32)
        return m0 + (2.0 * pl.shell_kappa) ** 0.5 * torch.exp(0.5 * lv0) * u
    t1 = {k: _mustar(pl_r, k) for k in keys2}
    _chunk(pl_r, jobs2, x0_c1 + 0.7, s_bar + 0.2)    # chunk 3, moved again
    for k in keys2:
        assert torch.equal(_mustar(pl_r, k), t1[k]), "mu* must be constant"

    # ---- 3) new retry -> fresh anchor -------------------------------------- #
    jobs_b = [(None, 7, 2)]                          # same scene, new try
    pl_r.set_row_jobs(jobs_b)
    pl_r.set_current_obs(s_bar[:1])
    pl_r.select_z((x0_c2[:1]).unsqueeze(1), s_bar[:1])
    assert (7, 2) in pl_r._anchor_mu
    with torch.no_grad():
        mu_new = MockVib().vib_enc(s_bar[:1], x0_c2[:1])[0][0]
    assert torch.equal(pl_r._anchor_mu[(7, 2)], mu_new.cpu()), \
        "new retry must capture ITS OWN first-chunk anchor"
    assert torch.equal(pl_r._anchor_mu[(7, 1)], a1[(7, 1)][0]), \
        "old retry's anchor must be untouched"

    # ---- 4) refresh=w5: capture at 1 and 6, not 2-5 ------------------------- #
    pl_w = PShellCostPlanner(MockVib(), anchor_refresh="w5",
                             shell_kappa=kappa, shell_seed=42)
    keyw_job = (None, 9, 3)
    keyw = (9, 3)
    for c in range(1, 7):
        xw = x0_c1[:1] + 0.1 * c
        _chunk(pl_w, [keyw_job], xw, s_bar[:1])
        if c == 1:
            snap1 = pl_w._anchor_mu[keyw].clone()
        elif c in (2, 3, 4, 5):
            assert torch.equal(pl_w._anchor_mu[keyw], snap1), \
                f"w5 must NOT refresh at chunk {c}"
        else:                                        # c == 6 -> refreshed
            assert not torch.equal(pl_w._anchor_mu[keyw], snap1), \
                "w5 must refresh at chunk 6"
    assert pl_w._chunk_ctr[keyw] == 6, "chunk counter must track select_z"

    # ---- 5) no-job rows: graph-connected zero ------------------------------- #
    pl_n = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                             shell_kappa=kappa, shell_seed=42)
    _chunk(pl_n, None, x0_c1, s_bar)                 # no set_row_jobs
    xg = _x(xc)
    c_n = pl_n.compute_loss(xg, s_bar, reduction="sum")
    assert float(c_n.detach()) == 0.0, "no-job rows must cost 0"
    c_n.backward()
    assert float(xg.grad.abs().sum()) == 0.0, "no-job rows must not push"

    # ---- 6) compute_loss before any capture: falls back to this chunk ------- #
    pl_d = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                             shell_kappa=kappa, shell_seed=42)
    _chunk(pl_d, jobs2, x0_c1, s_bar)
    pl_d._anchor_mu.clear()                          # simulate missing capture
    pl_d._anchor_lv.clear()
    refd = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa,
                                  shell_seed=42)
    refd.set_row_jobs(jobs2)
    refd.set_current_obs(s_bar)
    refd.select_z(x0_c1.unsqueeze(1), s_bar)
    assert torch.equal(
        pl_d.compute_loss(_x(xc), s_bar, reduction="sum"),
        refd.compute_loss(_x(xc), s_bar, reduction="sum")), \
        "pre-capture fallback must equal 方案A"

    # ---- 7) reductions + anchors detached ----------------------------------- #
    pl_red = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                               shell_kappa=kappa, shell_seed=42)
    _chunk(pl_red, jobs2, x0_c1, s_bar)
    s_ = pl_red.compute_loss(_x(xc), s_bar, reduction="sum")
    m_ = pl_red.compute_loss(_x(xc), s_bar, reduction="mean")
    assert torch.allclose(s_, m_ * x0_c1.shape[0], atol=1e-6)
    try:
        pl_red.compute_loss(_x(xc), s_bar, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass
    for k in keys2:
        assert not pl_red._anchor_mu[k].requires_grad, "anchor must be detached"

    # ---- 8) make_planner ek parsing ------------------------------------------ #
    p = make_planner(MockVib(), ek={})
    assert p.anchor_refresh == "retry" and p.shell_kappa == 2.5 \
        and p.shell_seed == 42
    p = make_planner(MockVib(), ek={"rand_anchor_refresh": "w5",
                                    "shell_kappa": 5.0})
    assert p.anchor_refresh == "w5" and p.shell_kappa == 5.0
    try:
        make_planner(MockVib(), ek={"rand_anchor_refresh": "bogus"})
        raise AssertionError("bogus refresh must raise")
    except ValueError:
        pass

    print("[smoke_pshell 持久锚定壳] OK: refresh=chunk bitwise==方案A "
          "(cost+grad, 2 chunks x 2 rows), retry anchor bit-stable across "
          "chunks (mu* constant, closed-form KL from frozen anchor), new "
          "retry -> fresh anchor, w5 captures at 1/6 only, no-job zero rows, "
          "pre-capture fallback == 方案A, reductions sum/mean, detached "
          "anchors, make_planner validation")


if __name__ == "__main__":
    main()
