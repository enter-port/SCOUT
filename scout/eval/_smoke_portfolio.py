"""Hermetic smoke for rand_portfolio (retry-index split, 2026-08-28).

Checks (per the campaign contract):
  * rand_split=10 (K=10): BITWISE-equal to AtypicalCostPlanner(cap=2.5) --
    cost AND gradient, both reductions, over 2 chunks with moving intents;
  * rand_split=0 (K=0): bitwise-equal to ShellTargetCostPlanner(shell_seed
    =42= rand_seed) -- cost AND gradient, both reductions (chunk-refresh
    anchor = shell original behavior);
  * 0<K<10 mixed batch: per-row DISPATCH -- row i's cost is bitwise-equal
    to the formula its try_idx selects (atypical's row / shell's row on the
    SAME batched forward, extracted via the None-out-baseline trick: a ref
    planner with every other baseline Noneed sums to exactly one row's
    cost); rows alternate ent/shell (odd/even behavioral split); ent costs
    <= 0, shell costs >= 0; gradients flow on every row;
  * reductions: sum == mean * B; bogus reduction raises;
  * no-job rows fall back to the ENTROPY-COST formula (bitwise vs
    AtypicalCostPlanner without jobs), NOT to a zero row;
  * no-baseline rows stay graph-connected zeros;
  * u seeding: from rand_seed (NOT the shell_seed kwarg) -- make_planner
    (rand_seed=7) draws u from default_rng([7, init, try]);
  * telemetry: fires only on reduction="sum" with grad; per-formula row
    counts and gradient-norm accumulators split correctly; print tick
    honors _tl_every; the extra backward retains the graph (outer
    .backward() after telemetry still works);
  * make_planner: ek parsing (rand_split / shell_kappa / rand_cap /
    rand_seed defaults + overrides).

Run:  python -m scout.eval._smoke_portfolio
"""
import contextlib
import io

import numpy as np
import torch

from scout.guidance.entropy_costs import (
    AtypicalCostPlanner,
    ShellTargetCostPlanner,
    _enc_forward,
)
from scout.guidance.rand_costs.portfolio import (
    PortfolioCostPlanner,
    make_planner,
)

Ds, Da, Dz = 6, 5, 4          # s_bar dim / action dim / style dim
B = 4
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
        return obs                                   # s_bar passthrough


def _x(a):
    return a.unsqueeze(1).clone().requires_grad_(True)


def _prime(pl, x0, s_bar, jobs=None):
    """set_row_jobs (opt) -> set_current_obs -> select_z."""
    if jobs is not None and hasattr(pl, "set_row_jobs"):
        pl.set_row_jobs(jobs)
    pl.set_current_obs(s_bar)
    pl.select_z(x0.unsqueeze(1), s_bar)


def _prime_ref(pl, x0, s_bar, jobs=None):
    """_prime for the shared-file planners (no set_row_jobs on atypical)."""
    if jobs is not None and hasattr(pl, "set_row_jobs"):
        pl.set_row_jobs(jobs)
    pl.set_current_obs(s_bar)
    pl.select_z(x0.unsqueeze(1), s_bar)


def _row_of_ref(ref, i):
    """Sum of a ref planner's cost with every baseline except row i Noneed
    == exactly row i's cost (same batched forward, bitwise)."""
    mus, lvs = list(ref._base_mu), list(ref._base_lv)
    ref._base_mu = [None if j != i else mus[j] for j in range(len(mus))]
    ref._base_lv = [None if j != i else lvs[j] for j in range(len(lvs))]
    with torch.no_grad():
        val = ref.compute_loss(xc_g.unsqueeze(1), s_bar_g, reduction="sum")
    ref._base_mu, ref._base_lv = mus, lvs
    return val


# batched tensors shared by the row-extraction helper (set in main)
xc_g = None
s_bar_g = None


def main():
    global xc_g, s_bar_g
    s_bar = torch.randn(B, Ds)
    x0 = torch.randn(B, Da)                    # chunk-1 intents
    xc = x0 + 0.25 * torch.randn(B, Da)        # candidates
    x0_c2 = x0 + 0.4 * torch.randn(B, Da)      # chunk-2 (moved) intents
    xc_g, s_bar_g = xc, s_bar
    kappa = 2.5
    jobs_ent = [(None, 3, 1), (None, 4, 9), (None, 5, 0), (None, 6, 3)]
    jobs_shell = [(None, 3, 0), (None, 4, 5), (None, 5, 9), (None, 6, 2)]
    jobs_mix = [(None, 3, 4), (None, 4, 5), (None, 5, 9), (None, 6, 0)]
    #                 ent(4<5)  shell(5)  shell(9)   ent(0)

    # ---- 1) K=10 == AtypicalCostPlanner (bitwise, both reductions) -------- #
    for red in ("sum", "mean"):
        pl = PortfolioCostPlanner(MockVib(), rand_split=10, shell_kappa=kappa,
                                  rand_cap=kappa, rand_seed=42)
        ref = AtypicalCostPlanner(MockVib(), cap=kappa)
        for x0_i, sb in ((x0, s_bar), (x0_c2, s_bar + 0.1)):  # 2 chunks
            _prime(pl, x0_i, sb, jobs=jobs_ent)
            _prime_ref(ref, x0_i, sb)
            c_p = pl.compute_loss(_x(xc), sb, reduction=red)
            c_r = ref.compute_loss(_x(xc), sb, reduction=red)
            assert torch.equal(c_p, c_r), (red, float(c_p), float(c_r))
            xg_p, xg_r = _x(xc), _x(xc)
            pl.compute_loss(xg_p, sb, reduction="sum").backward()
            ref.compute_loss(xg_r, sb, reduction="sum").backward()
            assert torch.equal(xg_p.grad, xg_r.grad), (red, "grads")

    # ---- 2) K=0 == ShellTargetCostPlanner (bitwise, both reductions) ------ #
    for red in ("sum", "mean"):
        pl = PortfolioCostPlanner(MockVib(), rand_split=0, shell_kappa=kappa,
                                  rand_cap=kappa, rand_seed=42)
        ref = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa,
                                     shell_seed=42)
        for x0_i, sb in ((x0, s_bar), (x0_c2, s_bar + 0.1)):
            _prime(pl, x0_i, sb, jobs=jobs_shell)
            _prime_ref(ref, x0_i, sb, jobs=jobs_shell)
            c_p = pl.compute_loss(_x(xc), sb, reduction=red)
            c_r = ref.compute_loss(_x(xc), sb, reduction=red)
            assert torch.equal(c_p, c_r), (red, float(c_p), float(c_r))
            xg_p, xg_r = _x(xc), _x(xc)
            pl.compute_loss(xg_p, sb, reduction="sum").backward()
            ref.compute_loss(xg_r, sb, reduction="sum").backward()
            assert torch.equal(xg_p.grad, xg_r.grad), (red, "grads")

    # ---- 3) mixed batch (K=5): per-row dispatch, bitwise ------------------ #
    K = 5
    pl = PortfolioCostPlanner(MockVib(), rand_split=K, shell_kappa=kappa,
                              rand_cap=kappa, rand_seed=42)
    att = AtypicalCostPlanner(MockVib(), cap=kappa)
    sh = ShellTargetCostPlanner(MockVib(), shell_kappa=kappa, shell_seed=42)
    _prime(pl, x0, s_bar, jobs=jobs_mix)
    _prime_ref(att, x0, s_bar)
    _prime_ref(sh, x0, s_bar, jobs=jobs_mix)
    # same batched forward the planners use -> identical mu/logvar tensors
    with torch.no_grad():
        mu, logvar = MockVib().vib_enc(s_bar,
                                       _enc_forward(pl, xc.unsqueeze(1)))
        rows_p, branches_p, _ = pl._row_costs(mu, logvar, xc.unsqueeze(1))
    for i, want in enumerate(("ent", "shell", "shell", "ent")):
        assert branches_p[i] == want, (i, branches_p[i])
        r_ref = (_row_of_ref(att, i) if want == "ent"
                 else _row_of_ref(sh, i))
        assert torch.equal(rows_p[i], r_ref), \
            (i, want, float(rows_p[i]), float(r_ref))
        if want == "ent":
            assert float(rows_p[i]) <= 0.0, (i, "ent cost must be <= 0")
        else:
            assert float(rows_p[i]) >= 0.0, (i, "shell cost must be >= 0")
    assert not torch.allclose(rows_p[0], rows_p[1]), \
        "ent and shell rows must diverge (odd/even split)"
    # mixed gradients flow on every row
    xg = _x(xc)
    pl.compute_loss(xg, s_bar, reduction="sum").backward()
    for i in range(B):
        assert float(xg.grad[i].abs().sum()) > 0.0, (i, "grad must flow")

    # ---- 4) reductions ------------------------------------------------------ #
    _prime(pl, x0_c2, s_bar + 0.1, jobs=jobs_mix)
    s_ = pl.compute_loss(_x(xc), s_bar + 0.1, reduction="sum")
    m_ = pl.compute_loss(_x(xc), s_bar + 0.1, reduction="mean")
    assert torch.allclose(s_, m_ * B, atol=1e-6)
    try:
        pl.compute_loss(_x(xc), s_bar + 0.1, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass

    # ---- 5) no-job rows: entropy-cost fallback (bitwise vs atypical) ------- #
    pl_n = PortfolioCostPlanner(MockVib(), rand_split=K, shell_kappa=kappa,
                                rand_cap=kappa, rand_seed=42)
    ref_n = AtypicalCostPlanner(MockVib(), cap=kappa)
    _prime(pl_n, x0, s_bar, jobs=None)          # set_row_jobs never called
    _prime_ref(ref_n, x0, s_bar)
    assert torch.equal(
        pl_n.compute_loss(_x(xc), s_bar, reduction="sum"),
        ref_n.compute_loss(_x(xc), s_bar, reduction="sum")), \
        "no-job rows must fall back to the entropy-cost formula"
    # explicit job with init=None also falls back to entropy (not zero)
    pl_n2 = PortfolioCostPlanner(MockVib(), rand_split=K, shell_kappa=kappa,
                                 rand_cap=kappa, rand_seed=42)
    _prime(pl_n2, x0[:1], s_bar[:1], jobs=[(None, None, 7)])
    val = pl_n2.compute_loss(_x(xc[:1]), s_bar[:1], reduction="sum")
    assert float(val.detach()) != 0.0, "init=None job must not zero the row"

    # ---- 6) no-baseline rows: graph-connected zero -------------------------- #
    pl_z = PortfolioCostPlanner(MockVib(), rand_split=K, shell_kappa=kappa,
                                rand_cap=kappa, rand_seed=42)
    pl_z.set_current_obs(s_bar)
    pl_z.set_row_jobs(jobs_mix)
    # select_z never fired -> no baseline captured
    xg = _x(xc)
    c_z = pl_z.compute_loss(xg, s_bar, reduction="sum")
    assert float(c_z.detach()) == 0.0
    c_z.backward()
    assert float(xg.grad.abs().sum()) == 0.0

    # ---- 7) u seeded from rand_seed, not shell_seed -------------------------- #
    p7 = make_planner(MockVib(), ek={"rand_seed": 7})
    rng = np.random.default_rng([7, 3, 2])
    v = rng.standard_normal(Dz)
    want = torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float32)
    assert torch.equal(
        p7._u_for((3, 2), torch.device("cpu"), torch.float32), want), \
        "u must be seeded by rand_seed"

    # ---- 8) telemetry -------------------------------------------------------- #
    pl_t = PortfolioCostPlanner(MockVib(), rand_split=K, shell_kappa=kappa,
                                rand_cap=kappa, rand_seed=42)
    pl_t._tl_every = 8                    # tick on the SECOND batched call
    _prime(pl_t, x0, s_bar, jobs=jobs_mix)
    xg1 = _x(xc)
    loss1 = pl_t.compute_loss(xg1, s_bar, reduction="sum")  # +2 ent, +2 shell
    assert pl_t._tl_n == [2, 2], pl_t._tl_n
    assert float(pl_t._tl_acc[0]) > 0 and float(pl_t._tl_acc[1]) > 0
    assert float(pl_t._tl_acc[4]) >= 0.0            # saturation counter live
    loss1.backward()                                 # graph was retained
    assert float(xg1.grad.abs().sum()) > 0.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _prime(pl_t, x0_c2, s_bar + 0.1, jobs=jobs_mix)
        xg2 = _x(xc)
        loss2 = pl_t.compute_loss(xg2, s_bar + 0.1, reduction="sum")  # tick
        loss2.backward()
    out = buf.getvalue()
    assert "[portfolio-telemetry]" in out and "rows ent=4 shell=4" in out, out
    assert "g-ratio" in out
    # mean reduction / no-grad: telemetry untouched
    n_before = list(pl_t._tl_n)
    _prime(pl_t, x0, s_bar, jobs=jobs_mix)
    pl_t.compute_loss(_x(xc), s_bar, reduction="mean")
    with torch.no_grad():
        pl_t.compute_loss(xc.unsqueeze(1), s_bar, reduction="sum")
    assert pl_t._tl_n == n_before, "telemetry must fire on sum+grad only"

    # ---- 9) make_planner ek parsing ------------------------------------------ #
    p = make_planner(MockVib(), ek={})
    assert p.rand_split == 5 and p.shell_kappa == 2.5 \
        and p.rand_cap == 2.5 and p.rand_seed == 42
    p = make_planner(MockVib(), ek={"rand_split": 7, "shell_kappa": 5.0,
                                    "rand_cap": 1.25, "rand_seed": 9})
    assert p.rand_split == 7 and p.shell_kappa == 5.0 \
        and p.rand_cap == 1.25 and p.rand_seed == 9

    print("[smoke_portfolio 重试组合] OK: K=10 bitwise==atypical / K=0 "
          "bitwise==shell (cost+grad, 2 reductions), mixed batch per-row "
          "dispatch (ent<=0 / shell>=0, odd-even diverge), reductions "
          "sum/mean + bogus raises, no-job -> entropy fallback, no-baseline "
          "zero rows, u seeded by rand_seed, per-formula telemetry "
          "(counts/grad-norms/tick/gate/retain_graph), make_planner ek")


if __name__ == "__main__":
    main()
