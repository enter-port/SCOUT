"""Hermetic smoke for rand_ushuffle (per-chunk direction redraw, 2026-08-28).

Checks (campaign contract + width-grid cell):
  * u_resample=chunk: u CHANGES every chunk (rng([shell_seed, init, try,
    chunk_idx]), 1-based counter) and is reproducible from the raw seed;
    the compute_loss target is verified against an independent closed-form
    KL built with the hand-drawn per-chunk u (chunk-2 anchor mode);
  * u_resample=retry (default): BITWISE-equal to 方案A
    ShellTargetCostPlanner with anchor_refresh=chunk (cost + grad, both
    reductions, 2 chunks x 2 rows) and to PShellCostPlanner with
    anchor_refresh=retry (the two parent clocks unchanged);
  * anchor behavior == pshell's corresponding mode: anchor_refresh=retry
    freezes the first chunk's capture (bit-stable, _base_mu still refreshes
    per chunk); anchor_refresh=chunk never persists anchors;
  * a new retry restarts ITS OWN chunk clock (u of try2/chunk1 != try1's);
  * no-job rows: graph-connected zero (cost 0, zero grad);
  * reductions: sum == mean * B; bogus reduction raises;
  * make_planner: ek parsing (rand_u_resample / rand_anchor_refresh /
    shell_kappa / shell_seed), invalid modes raise.

Run:  python -m scout.eval._smoke_ushuffle
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner
from scout.guidance.rand_costs.pshell import PShellCostPlanner
from scout.guidance.rand_costs.ushuffle import UShuffleCostPlanner, make_planner

Ds, Da, Dz = 6, 5, 4          # s_bar dim / action dim / style dim
torch.manual_seed(0)
W = torch.randn(Dz, Da)
Wv = 0.3 * torch.randn(Dz, Da)


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                              # (B, Dz)
        logvar = -1.0 + 0.2 * (a @ Wv.T)          # a-dependent sigma head
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


def _u_raw(seed, init, try_, cidx):
    rng = np.random.default_rng([seed, init, try_, cidx])
    v = rng.standard_normal(Dz)
    return torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float32)


def main():
    s_bar = torch.randn(2, Ds)
    x0_c1 = torch.randn(2, Da)                       # chunk-1 intents
    x0_c2 = x0_c1 + 0.4 * torch.randn(2, Da)         # chunk-2 intents (moved)
    xc = x0_c1 + 0.2 * torch.randn(2, Da)            # candidates
    jobs2 = [(None, 7, 1), (None, 8, 2)]             # raw replan jobs
    keys2 = [(7, 1), (8, 2)]
    kappa = 2.5
    seed = 42

    # ---- 1) u_resample=retry == 方案A (anchor chunk) bitwise ---------------- #
    for red in ("sum", "mean"):
        pl = UShuffleCostPlanner(MockVib(), anchor_refresh="chunk",
                                 u_resample="retry", shell_kappa=kappa,
                                 shell_seed=seed)
        ref = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa,
                                     shell_seed=seed)
        for x0, sb in [(x0_c1, s_bar), (x0_c2, s_bar + 0.1)]:
            for p in (pl, ref):
                p.set_row_jobs(jobs2)
                p.set_current_obs(sb)
                p.select_z(x0.unsqueeze(1), sb)
            c_p = pl.compute_loss(_x(xc), sb, reduction=red)
            c_r = ref.compute_loss(_x(xc), sb, reduction=red)
            assert torch.equal(c_p, c_r), (red, float(c_p), float(c_r))
            xg_p, xg_r = _x(xc), _x(xc)
            pl.compute_loss(xg_p, sb, reduction="sum").backward()
            ref.compute_loss(xg_r, sb, reduction="sum").backward()
            assert torch.equal(xg_p.grad, xg_r.grad), (red, "grads")
        assert pl._anchor_mu == {} and pl._ush_u_cache == {}
        assert pl._u_ctr == {}, "retry mode must not tick the u clock"

    # ---- 2) u_resample=retry == pshell (anchor retry) bitwise --------------- #
    pl = UShuffleCostPlanner(MockVib(), anchor_refresh="retry",
                             u_resample="retry", shell_kappa=kappa,
                             shell_seed=seed)
    ref = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                            shell_kappa=kappa, shell_seed=seed)
    for x0, sb in [(x0_c1, s_bar), (x0_c2, s_bar + 0.1)]:
        for p in (pl, ref):
            p.set_row_jobs(jobs2)
            p.set_current_obs(sb)
            p.select_z(x0.unsqueeze(1), sb)
        assert torch.equal(pl.compute_loss(_x(xc), sb, reduction="sum"),
                           ref.compute_loss(_x(xc), sb, reduction="sum"))

    # ---- 3) u_resample=chunk: u clock + reproducibility -------------------- #
    pl = UShuffleCostPlanner(MockVib(), anchor_refresh="chunk",
                             u_resample="chunk", shell_kappa=kappa,
                             shell_seed=seed)
    _chunk(pl, jobs2, x0_c1, s_bar)                  # chunk 1
    assert pl._u_ctr == {keys2[0]: 1, keys2[1]: 1}
    u1 = {k: pl._u_for(k, torch.device("cpu"), torch.float32).clone()
          for k in keys2}
    u1_raw = {k: _u_raw(seed, k[0], k[1], 1) for k in keys2}
    for k in keys2:
        assert torch.equal(u1[k], u1_raw[k]), \
            "chunk-1 u must be rng([seed, init, try, 1])"
    _chunk(pl, jobs2, x0_c2, s_bar + 0.1)            # chunk 2
    assert pl._u_ctr == {keys2[0]: 2, keys2[1]: 2}
    u2 = {k: pl._u_for(k, torch.device("cpu"), torch.float32).clone()
          for k in keys2}
    for k in keys2:
        assert torch.equal(u2[k], _u_raw(seed, k[0], k[1], 2)), \
            "chunk-2 u must be rng([seed, init, try, 2])"
        assert not torch.allclose(u1[k], u2[k]), "u must change per chunk"
    assert len(pl._ush_u_cache) == 4                 # 2 keys x 2 chunks
    # same u object on repeat calls within the chunk (cached, deterministic)
    assert torch.equal(
        u2[keys2[0]],
        pl._u_for(keys2[0], torch.device("cpu"), torch.float32))

    # ---- 4) chunk-2 cost == closed form with the per-chunk u ---------------- #
    # anchor_refresh=chunk -> m0/lv0 are THIS chunk's capture; u is chunk 2's.
    got = float(pl.compute_loss(_x(xc), s_bar + 0.1,
                                reduction="sum").detach())
    want = 0.0
    for i, k in enumerate(keys2):
        with torch.no_grad():
            m0, lv0 = pl._base_mu[i].float(), pl._base_lv[i].float()
            mu, logvar = MockVib().vib_enc((s_bar + 0.1)[i:i + 1],
                                           xc[i:i + 1])
            tgt = m0 + (2.0 * kappa) ** 0.5 * torch.exp(0.5 * lv0) * u2[k]
            want += float(0.5 * (((mu[0] - tgt) ** 2 / torch.exp(lv0))
                                 + torch.exp(logvar[0]) / torch.exp(lv0)
                                 - 1.0 - (logvar[0] - lv0)).sum())
    assert abs(got - want) < 1e-5, (got, want)
    # ...and it must differ from the retry-u cost (same anchors/sigmas)
    pl_ru = UShuffleCostPlanner(MockVib(), anchor_refresh="chunk",
                                u_resample="retry", shell_kappa=kappa,
                                shell_seed=seed)
    _chunk(pl_ru, jobs2, x0_c1, s_bar)
    _chunk(pl_ru, jobs2, x0_c2, s_bar + 0.1)
    got_ru = float(pl_ru.compute_loss(_x(xc), s_bar + 0.1,
                                      reduction="sum").detach())
    assert abs(got - got_ru) > 1e-4, "per-chunk u must change the cost"

    # ---- 5) anchor behavior == pshell corresponding mode -------------------- #
    # (a) anchor_refresh=retry + u=chunk: anchor frozen at first chunk
    pl2 = UShuffleCostPlanner(MockVib(), anchor_refresh="retry",
                              u_resample="chunk", shell_kappa=kappa,
                              shell_seed=seed)
    _chunk(pl2, jobs2, x0_c1, s_bar)                 # chunk 1
    a1 = {k: (pl2._anchor_mu[k].clone(), pl2._anchor_lv[k].clone())
          for k in keys2}
    _chunk(pl2, jobs2, x0_c2, s_bar + 0.1)           # chunk 2
    for k in keys2:
        assert torch.equal(pl2._anchor_mu[k], a1[k][0]), \
            "retry anchor must be bit-stable across chunks (pshell parity)"
    with torch.no_grad():                            # _base_mu still refreshes
        mu0_c2 = MockVib().vib_enc(s_bar + 0.1, x0_c2)[0]
    assert torch.allclose(pl2._base_mu[0], mu0_c2[0])
    # pshell parity on the anchor dicts themselves (same chunks -> same caps)
    ref_p = PShellCostPlanner(MockVib(), anchor_refresh="retry",
                              shell_kappa=kappa, shell_seed=seed)
    _chunk(ref_p, jobs2, x0_c1, s_bar)
    _chunk(ref_p, jobs2, x0_c2, s_bar + 0.1)
    for k in keys2:
        assert torch.equal(pl2._anchor_mu[k], ref_p._anchor_mu[k])
    # (b) anchor_refresh=chunk + u=chunk: no anchors ever persisted
    assert pl._anchor_mu == {} and pl._anchor_lv == {}
    # independent clocks: u clock ticks in chunk-anchor mode (pshell's
    # own _chunk_ctr does not) -- both at 2 here
    assert pl._u_ctr[keys2[0]] == 2 and pl._chunk_ctr.get(keys2[0], 0) == 0

    # ---- 6) new retry -> fresh u clock --------------------------------------- #
    jobs_b = [(None, 7, 2)]                          # same scene, new try
    pl_b = UShuffleCostPlanner(MockVib(), anchor_refresh="chunk",
                               u_resample="chunk", shell_kappa=kappa,
                               shell_seed=seed)
    _chunk(pl_b, jobs_b, x0_c1[:1], s_bar[:1])
    u_new = pl_b._u_for((7, 2), torch.device("cpu"), torch.float32)
    assert torch.equal(u_new, _u_raw(seed, 7, 2, 1)), \
        "new retry chunk-1 u = rng([seed, 7, 2, 1])"
    assert not torch.equal(u_new, u1[(7, 1)]), \
        "different try must get a different direction"

    # ---- 7) no-job rows: graph-connected zero -------------------------------- #
    pl_n = UShuffleCostPlanner(MockVib(), anchor_refresh="chunk",
                               u_resample="chunk", shell_kappa=kappa,
                               shell_seed=seed)
    _chunk(pl_n, None, x0_c1, s_bar)                 # no set_row_jobs
    xg = _x(xc)
    c_n = pl_n.compute_loss(xg, s_bar, reduction="sum")
    assert float(c_n.detach()) == 0.0, "no-job rows must cost 0"
    c_n.backward()
    assert float(xg.grad.abs().sum()) == 0.0, "no-job rows must not push"

    # ---- 8) reductions -------------------------------------------------------- #
    pl_r = UShuffleCostPlanner(MockVib(), anchor_refresh="retry",
                               u_resample="chunk", shell_kappa=kappa,
                               shell_seed=seed)
    _chunk(pl_r, jobs2, x0_c1, s_bar)
    s_ = pl_r.compute_loss(_x(xc), s_bar, reduction="sum")
    m_ = pl_r.compute_loss(_x(xc), s_bar, reduction="mean")
    assert torch.allclose(s_, m_ * x0_c1.shape[0], atol=1e-6)
    try:
        pl_r.compute_loss(_x(xc), s_bar, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass

    # ---- 9) make_planner ek parsing ------------------------------------------- #
    p = make_planner(MockVib(), ek={})
    assert p.u_resample == "retry" and p.anchor_refresh == "retry" \
        and p.shell_kappa == 2.5 and p.shell_seed == 42
    p = make_planner(MockVib(), ek={"rand_u_resample": "chunk",
                                    "rand_anchor_refresh": "chunk",
                                    "shell_kappa": 5.0})
    assert p.u_resample == "chunk" and p.anchor_refresh == "chunk" \
        and p.shell_kappa == 5.0
    for bad in ({"rand_u_resample": "bogus"},
                {"rand_anchor_refresh": "bogus"}):
        try:
            make_planner(MockVib(), ek=bad)
            raise AssertionError(f"{bad} must raise")
        except ValueError:
            pass

    print("[smoke_ushuffle 逐chunk方向重抽] OK: u redrawn per chunk "
          "(rng[seed,init,try,chunk] reproducible, differs across chunks/"
          "tries), closed-form KL with per-chunk u, retry mode bitwise=="
          "方案A & ==pshell, anchor behavior == pshell parity (retry frozen "
          "/ chunk never persists), independent u clock ticks in chunk-anchor "
          "mode, no-job zero rows, reductions sum/mean, make_planner "
          "validation")


if __name__ == "__main__":
    main()
