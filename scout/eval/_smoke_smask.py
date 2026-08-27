"""Hermetic smoke for rand_smask (方案F random temporal gating, 2026-08-27).

Checks (per the campaign contract):
  * window family: c0 per (init, try) is a permutation of the grid {0,L,..,9L}
    -> retry windows NEVER overlap; deterministic per key; varies with init;
    omega = 1 iff c0 <= c < c0+L;
  * front family: omega = 1 iff c < c0 (truncation); c0=0 -> all-off;
  * bern family: iid per chunk, empirical fraction ~ rho, deterministic per
    (seed, init, try, chunk), varies with chunk and with rand_seed;
  * w=1 everywhere (front, huge c0) is BYTE-EQUAL to AtypicalCostPlanner with
    the same cap: costs AND gradients, both reductions; no-job fallback too;
  * w=0 rows: graph-connected exact 0 (backward works, grad all-zero, value
    == 0.0), mixed batch cost == sum of the w=1 rows' atypical costs;
  * chunk counting: repeated set_row_jobs calls increment c (0,1,2,...) per
    (init, try) key; the cost switches on/off exactly at the schedule edges;
  * reductions: sum == mean * B; bogus reduction raises.

Run:  python -m scout.eval._smoke_smask
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner
from scout.guidance.rand_costs.smask import SMaskCostPlanner

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
        return obs                                  # s_bar passthrough


def _x(a):
    return a.unsqueeze(1).clone().requires_grad_(True)


def _atypical_row_cost(a_intent, a_cand, s_bar, cap=2.5):
    """Single-row plain 方案三 cost: baseline captured at a_intent, cost
    evaluated at a_cand."""
    ref = AtypicalCostPlanner(MockVib(), cap=cap)
    ref.set_current_obs(s_bar)
    ref.select_z(a_intent.unsqueeze(1), s_bar)
    return float(ref.compute_loss(_x(a_cand), s_bar,
                                  reduction="sum").detach())


def _prepare(pl, jobs, x0, s_bar):
    if jobs is not None:
        pl.set_row_jobs(jobs)
    pl.set_current_obs(s_bar)
    pl.select_z(x0.unsqueeze(1), s_bar)


def main():
    s_bar = torch.randn(2, Ds)
    cap = 2.5
    x0 = torch.randn(2, Da)
    # off-intent candidate (nonzero plain cost, strictly INSIDE the cap so
    # the gradients are live): shrink the shift until uncapped.  x0 is the
    # unguided intent captured by select_z; xc is what compute_loss sees.
    delta = torch.randn(2, Da)
    _r0 = AtypicalCostPlanner(MockVib(), cap=cap)
    _r0.set_current_obs(s_bar)
    _r0.select_z(x0.unsqueeze(1), s_bar)
    for _sc in (0.35, 0.15, 0.08, 0.04):
        xc = x0 + _sc * delta
        if float(_r0.compute_loss(_x(xc), s_bar,
                                  reduction="sum").detach()) > -2 * cap + 0.5:
            break

    # ---- window family: permutation grid, no overlap ---------------------- #
    L = 3
    plw = SMaskCostPlanner(MockVib(), family="window", win_len=L,
                           cap=cap, rand_seed=42)
    for init in (0, 7, 123):
        c0s = [plw._window_c0((init, k)) for k in range(10)]
        assert sorted(c0s) == [g * L for g in range(10)], \
            "c0 must be a permutation of the grid {0,L,...,9L}"
        iv = sorted((c, c + L) for c in c0s)
        for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
            assert b0 >= a1, "retry windows must never overlap"
    assert plw._window_c0((7, 3)) == plw._window_c0((7, 3)), "deterministic"
    assert len({plw._window_c0((i, 3)) for i in range(20)}) > 1, \
        "c0 must vary with init"
    c0a = plw._window_c0((7, 3))
    assert plw._omega_for((7, 3), c0a - 1) == 0.0
    for c in range(c0a, c0a + L):
        assert plw._omega_for((7, 3), c) == 1.0
    assert plw._omega_for((7, 3), c0a + L) == 0.0

    # ---- front family: truncation ----------------------------------------- #
    plf = SMaskCostPlanner(MockVib(), family="front", win_c0=4, cap=cap)
    assert [plf._omega_for((5, 2), c) for c in range(6)] == [1.0, 1.0, 1.0,
                                                             1.0, 0.0, 0.0]
    plfh = SMaskCostPlanner(MockVib(), family="front", win_c0=10 ** 6,
                            cap=cap)
    assert all(plfh._omega_for((5, 2), c) == 1.0 for c in range(1000))

    # ---- bern family: fraction / determinism / variety --------------------- #
    plb = SMaskCostPlanner(MockVib(), family="bern", bern_rho=0.7, cap=cap)
    draws = [plb._omega_for((i, k), c) for i in range(20)
             for k in range(10) for c in range(10)]
    frac = float(np.mean(draws))
    assert abs(frac - 0.7) < 0.04, f"bern fraction {frac} != 0.7"
    assert all(plb._omega_for((4, 4), 7) == plb._omega_for((4, 4), 7)
               for _ in range(3)), "bern must be deterministic per (i,k,c)"
    assert len({plb._omega_for((4, 4), c) for c in range(40)}) > 1, \
        "bern must vary with chunk"
    plb43 = SMaskCostPlanner(MockVib(), family="bern", bern_rho=0.7,
                             cap=cap, rand_seed=43)
    assert any(plb._omega_for((i, k), c) != plb43._omega_for((i, k), c)
               for i in range(5) for k in range(5) for c in range(5)), \
        "bern must vary with rand_seed"

    # ---- w=1 everywhere: BYTE-EQUAL to AtypicalCostPlanner ---------------- #
    pl1 = SMaskCostPlanner(MockVib(), family="front", win_c0=10 ** 6, cap=cap)
    ref = AtypicalCostPlanner(MockVib(), cap=cap)
    _prepare(pl1, [(None, 7, 3), (None, 8, 0)], x0, s_bar)
    ref.set_current_obs(s_bar)
    ref.select_z(x0.unsqueeze(1), s_bar)
    for red in ("sum", "mean"):
        c1 = pl1.compute_loss(_x(xc), s_bar, reduction=red)
        cr = ref.compute_loss(_x(xc), s_bar, reduction=red)
        assert torch.equal(c1, cr), (red, float(c1), float(cr))
    assert abs(float(c1.detach())) > 1e-3, "use a candidate with nonzero cost"
    xg1, xgr = _x(xc), _x(xc)
    pl1.compute_loss(xg1, s_bar, reduction="sum").backward()
    ref.compute_loss(xgr, s_bar, reduction="sum").backward()
    assert torch.equal(xg1.grad, xgr.grad), "w=1 gradients must be bit-equal"

    # ---- no-job fallback: also byte-equal ---------------------------------- #
    plfb = SMaskCostPlanner(MockVib(), family="window", win_len=L,
                            cap=cap, rand_seed=42)
    _prepare(plfb, None, x0, s_bar)                  # set_row_jobs never called
    cf = plfb.compute_loss(_x(xc), s_bar, reduction="sum")
    crf = ref.compute_loss(_x(xc), s_bar, reduction="sum")
    assert torch.equal(cf, crf), (float(cf), float(crf))

    # ---- w=0 rows: graph-connected exact zero ------------------------------ #
    pl0 = SMaskCostPlanner(MockVib(), family="front", win_c0=0, cap=cap)
    _prepare(pl0, [(None, 7, 3), (None, 8, 0)], x0, s_bar)
    xg0 = _x(xc)
    z = pl0.compute_loss(xg0, s_bar, reduction="sum")
    assert float(z.detach()) == 0.0, float(z.detach())  # masked to exact 0
    assert z.grad_fn is not None, "w=0 cost must stay graph-connected"
    z.backward()
    assert xg0.grad is not None and float(xg0.grad.abs().sum()) == 0.0

    # ---- chunk counting: c increments per repeated set_row_jobs ------------ #
    plcc = SMaskCostPlanner(MockVib(), family="front", win_c0=3, cap=cap)
    jobs1 = [(None, 7, 3)]
    x1 = x0[:1]
    x1c = xc[:1]
    for rep in range(5):
        plcc.set_row_jobs(jobs1)
        plcc.set_current_obs(s_bar)
        plcc.select_z(x1.unsqueeze(1), s_bar)
        assert plcc._row_c == [rep], (rep, plcc._row_c)
        c = float(plcc.compute_loss(_x(x1c), s_bar, reduction="sum").detach())
        r = _atypical_row_cost(x1, x1c, s_bar, cap)
        if rep < 3:
            assert abs(c - r) < 1e-6, (rep, c, r)    # inside the front window
        else:
            assert c == 0.0, (rep, c)                # released -> pure DP

    # ---- mixed batch: cost == w0*r0 + w1*r1 -------------------------------- #
    # pick keys deterministically: try k0 has c0=0 (permutation always
    # contains the grid origin), try k1 has c0 > 0 -> one ON row, one OFF row
    plsel = SMaskCostPlanner(MockVib(), family="window", win_len=L,
                             cap=cap, rand_seed=42)
    k0 = next(k for k in range(10) if plsel._window_c0((7, k)) == 0)
    k1 = next(k for k in range(10) if plsel._window_c0((7, k)) > 0)
    keys = [(7, k0), (7, k1)]
    plm = SMaskCostPlanner(MockVib(), family="window", win_len=L,
                           cap=cap, rand_seed=42)
    _prepare(plm, [(None, 7, k0), (None, 7, k1)], x0, s_bar)  # chunk c=0
    w = [plm._omega_for(k, 0) for k in keys]
    assert w == [1.0, 0.0], w
    r = [_atypical_row_cost(x0[i:i + 1], xc[i:i + 1], s_bar, cap)
         for i in range(2)]
    assert abs(r[1]) > 1e-3, "off row must have a NONZERO plain cost to mask"
    cm = float(plm.compute_loss(_x(xc), s_bar, reduction="sum").detach())
    assert abs(cm - (w[0] * r[0] + w[1] * r[1])) < 1e-6, (cm, w, r)
    assert abs(cm - r[0]) < 1e-6, "the OFF row must contribute exactly nothing"
    xg = _x(xc)
    plm.compute_loss(xg, s_bar, reduction="sum").backward()
    assert float(xg.grad[1].abs().sum()) == 0.0, \
        "masked row's gradient must be exactly zero"
    assert float(xg.grad[0].abs().sum()) > 0.0

    # ---- reductions: sum == mean * B; bogus raises ------------------------- #
    ssum = plm.compute_loss(_x(xc), s_bar, reduction="sum")
    smean = plm.compute_loss(_x(xc), s_bar, reduction="mean")
    assert torch.allclose(ssum, smean * x0.shape[0], atol=1e-6)
    try:
        plm.compute_loss(_x(xc), s_bar, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass

    print(f"[smoke_smask 方案F] OK: window grid permutation/no-overlap, "
          f"front truncation, bern fraction {frac:.3f}~0.7, "
          f"w=1 byte-equal to atypical (cost+grad, both reductions), "
          f"no-job fallback, w=0 graph-connected zero, chunk counting "
          f"(c=0..4, on/off at the window edge), mixed batch "
          f"w={w}, reductions")


if __name__ == "__main__":
    main()
