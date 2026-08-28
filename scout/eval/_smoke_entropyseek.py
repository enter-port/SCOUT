"""Hermetic smoke for rand_entropyseek (proposal 3 sigma-axis, 2026-08-28).

Checks (per the campaign contract):
  * mask: deterministic per (init, try) and rand_seed, varies across keys,
    entries in {0,1}, rho=0 -> all zeros, rho=1 -> all ones, all-zero first
    draw triggers exactly ONE redraw (anti-extinction) and the redraw value
    is what the same rng stream yields;
  * rho=0 degenerates BYTE-FOR-BYTE to AtypicalCostPlanner (w_A=1): outputs
    AND gradients, both reductions;
  * rows without job context fall back to plain 方案三 even for rho>0;
  * w_A scales only the anchor (rho=0, w_A=2 -> exactly 2x 方案三);
  * sigma-term direction: ln sigma UP on a MASKED dim is rewarded (cost
    strictly below 方案三); DOWN on a masked dim, or UP on an UNMASKED dim,
    is relu/mask-cut to exactly 0 (byte-equal to 方案三);
  * kappa_sigma truncation: big masked up-perturbation -> cost == 方案三
    cost - kappa_sigma exactly, gradient byte-equal (clamped flat);
  * differentiable; reduction sum == mean*B; bogus reduction raises.

The mock encoder uses a DIAGONAL logvar head (logvar_i = -1 + 0.2*wv_i*a_i)
so perturbing action dim j moves exactly ln sigma_j -- clean orthogonal
direction tests.

Run:  python -m scout.eval._smoke_entropyseek
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner
from scout.guidance.rand_costs.entropyseek import EntropySeekCostPlanner

Ds, Da, Dz = 6, 6, 6          # s_bar dim / action dim / style dim
torch.manual_seed(0)
W = torch.randn(Dz, Da)
wv = 0.5 + torch.rand(Dz)     # positive diagonal logvar gains


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                                # (B, Dz)
        logvar = -1.0 + 0.2 * wv * a                # a-dependent diag sigma head
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


CPU, F32 = torch.device("cpu"), torch.float32


def _prepare(pl, jobs, x0, s_bar):
    if jobs is None:
        pl._row_jobs = []
    else:
        pl.set_row_jobs(jobs)
    pl.select_z(x0.unsqueeze(1), s_bar)


def _ref_with(x0, s_bar):
    ref = AtypicalCostPlanner(MockVib(), cap=2.5)
    ref.set_row_context([7, 8])                     # no-op for atypical
    ref.select_z(x0.unsqueeze(1), s_bar)
    return ref


def main():
    s_bar = torch.randn(2, Ds)
    cap = 2.5
    jobs = [(None, 7, 3), (None, 8, 0)]             # (state, init_idx, try)
    x0 = torch.randn(2, Da)

    # ---- mask bookkeeping ----------------------------------------------- #
    pl = EntropySeekCostPlanner(MockVib(), rho=0.5, rand_seed=42)
    m1 = pl._mask_for((7, 3), CPU, F32)
    m1b = pl._mask_for((7, 3), CPU, F32)
    assert torch.equal(m1, m1b), "mask must be deterministic per (init,try)"
    assert set(m1.unique().tolist()) <= {0.0, 1.0}, "mask entries in {0,1}"
    keys = [(7, k) for k in range(6)] + [(8, 3)]
    masks = {tuple(pl._mask_for(k, CPU, F32).tolist()) for k in keys}
    assert len(masks) >= 2, "mask must vary across (init,try) keys"
    pl43 = EntropySeekCostPlanner(MockVib(), rho=0.5, rand_seed=43)
    assert not torch.equal(m1, pl43._mask_for((7, 3), CPU, F32)), \
        "mask must vary with rand_seed"
    plz = EntropySeekCostPlanner(MockVib(), rho=0.0, rand_seed=42)
    assert torch.equal(plz._mask_for((7, 3), CPU, F32), torch.zeros(Dz)), \
        "rho=0 -> all-zero mask"
    plo = EntropySeekCostPlanner(MockVib(), rho=1.0, rand_seed=42)
    assert torch.equal(plo._mask_for((7, 3), CPU, F32), torch.ones(Dz)), \
        "rho=1 -> all-one mask"

    # ---- anti-extinction redraw (exactly one, from the same stream) ------ #
    rho_t = 0.05
    plt = EntropySeekCostPlanner(MockVib(), rho=rho_t, rand_seed=42)
    found = None
    for init in range(64):
        rng = np.random.default_rng([42, init, 0])
        first = rng.random(Dz) < rho_t
        second = rng.random(Dz) < rho_t
        if not first.any() and second.any():
            found = (init, first, second)
            break
    assert found is not None, "no all-zero-first-draw key found for redraw test"
    init, first, second = found
    m = plt._mask_for((init, 0), CPU, F32)
    assert not torch.equal(m, torch.as_tensor(first.astype(np.float32))), \
        "all-zero first draw must be redrawn"
    assert torch.equal(m, torch.as_tensor(second.astype(np.float32))), \
        "the redraw value must be the same rng stream's next draw"

    # ---- rho=0 byte-for-byte equivalence with AtypicalCostPlanner -------- #
    pl0 = EntropySeekCostPlanner(MockVib(), rho=0.0, anchor_w=1.0, cap=cap,
                                 rand_seed=42)
    ref = _ref_with(x0, s_bar)
    _prepare(pl0, jobs, x0, s_bar)
    for red in ("sum", "mean"):
        c0 = pl0.compute_loss(_x(x0), s_bar, reduction=red)
        cr = ref.compute_loss(_x(x0), s_bar, reduction=red)
        assert torch.equal(c0, cr), (red, float(c0), float(cr))
    xg0, xgr = _x(x0), _x(x0)
    pl0.compute_loss(xg0, s_bar, reduction="sum").backward()
    ref.compute_loss(xgr, s_bar, reduction="sum").backward()
    assert torch.equal(xg0.grad, xgr.grad), "rho=0 gradients must be bit-equal"

    # ---- fallback: no job rows, rho>0 still equals 方案三 ----------------- #
    plf = EntropySeekCostPlanner(MockVib(), rho=0.5, anchor_w=1.0, cap=cap)
    _prepare(plf, None, x0, s_bar)
    cf = plf.compute_loss(_x(x0), s_bar, reduction="sum")
    crf = ref.compute_loss(_x(x0), s_bar, reduction="sum")
    assert torch.equal(cf, crf), (float(cf), float(crf))

    # ---- w_A scales only the anchor (rho=0) ------------------------------- #
    plw = EntropySeekCostPlanner(MockVib(), rho=0.0, anchor_w=2.0, cap=cap)
    _prepare(plw, jobs, x0, s_bar)
    cw = plw.compute_loss(_x(x0), s_bar, reduction="sum")
    cr2 = ref.compute_loss(_x(x0), s_bar, reduction="sum")
    assert torch.equal(cw, 2.0 * cr2), (float(cw), float(cr2))

    # ---- direction: masked/unmasked, up/down ------------------------------ #
    plb = EntropySeekCostPlanner(MockVib(), rho=0.5, kappa_sigma=1.25,
                                 anchor_w=1.0, cap=cap, rand_seed=42)
    _prepare(plb, jobs, x0, s_bar)
    mask = plb._mask_for((7, 3), CPU, F32).tolist()
    assert 1.0 in mask and 0.0 in mask, "need a mixed mask for this seed"
    j1, j0 = mask.index(1.0), mask.index(0.0)       # masked / unmasked dims
    ref1 = _ref_with(x0, s_bar)

    def _one_row(delta):
        a = (x0[0] + delta).unsqueeze(0)            # (1, Da)
        ce = float(plb.compute_loss(_x(a), s_bar, reduction="sum").detach())
        ca = float(ref1.compute_loss(_x(a), s_bar, reduction="sum").detach())
        return ce, ca

    alpha = 1.0
    d = torch.zeros(Da); d[j1] = alpha              # ln sigma_j1 UP (masked)
    ce, ca = _one_row(d)
    assert ce < ca - 1e-6, ("masked ln-sigma UP must be rewarded", ce, ca)
    assert ce - ca > -1.25 - 1e-6, "sigma reward capped by kappa_sigma"
    d = torch.zeros(Da); d[j1] = -alpha             # ln sigma_j1 DOWN (masked)
    ce, ca = _one_row(d)
    assert ce == ca, ("masked ln-sigma DOWN must be relu-cut to exactly 0",
                      ce, ca)
    d = torch.zeros(Da); d[j0] = alpha              # ln sigma_j0 UP (unmasked)
    ce, ca = _one_row(d)
    assert ce == ca, ("unmasked ln-sigma UP must be mask-cut to exactly 0",
                      ce, ca)
    d = torch.zeros(Da); d[j0] = -alpha             # unmasked DOWN
    ce, ca = _one_row(d)
    assert ce == ca, "unmasked ln-sigma DOWN must be exactly 0"

    # gradient of the sigma part alone: nonzero toward further increase on
    # the masked dim, exactly zero on the unmasked dim
    d = torch.zeros(Da); d[j1] = alpha
    a = (x0[0] + d).unsqueeze(0)
    xe, xa = _x(a), _x(a)
    plb.compute_loss(xe, s_bar, reduction="sum").backward()
    ref1.compute_loss(xa, s_bar, reduction="sum").backward()
    g_sig = (xe.grad - xa.grad).flatten()
    assert float(g_sig[j1]) < 0.0, "reward must push a_j1 further up"
    assert float(g_sig[j0]) == 0.0, "no sigma force on unmasked dims"
    assert torch.isfinite(g_sig).all()

    # ---- kappa_sigma truncation: flat beyond the cap ----------------------- #
    d = torch.zeros(Da); d[j1] = 50.0               # ln sigma_j1 way up
    a = (x0[0] + d).unsqueeze(0)
    ce = float(plb.compute_loss(_x(a), s_bar, reduction="sum").detach())
    ca = float(ref1.compute_loss(_x(a), s_bar, reduction="sum").detach())
    assert abs((ce - ca) + 1.25) < 1e-5, ("sigma term must sit at -kappa_sigma",
                                          ce, ca)
    xe, xa = _x(a), _x(a)
    plb.compute_loss(xe, s_bar, reduction="sum").backward()
    ref1.compute_loss(xa, s_bar, reduction="sum").backward()
    assert torch.equal(xe.grad, xa.grad), \
        "clamped sigma term must contribute exactly zero gradient"

    # ---- partial job context: row 1 without a job = plain 方案三 ----------- #
    plp = EntropySeekCostPlanner(MockVib(), rho=0.5, kappa_sigma=1.25, cap=cap,
                                 rand_seed=42)
    _prepare(plp, [(None, 7, 3)], x0, s_bar)        # only row 0 has a job
    refp = _ref_with(x0, s_bar)
    a2 = x0.clone()
    d1 = torch.zeros(Da); d1[j1] = alpha
    a2[0] = a2[0] + d1                              # row 0: masked-up (j1)
    m8 = plb._mask_for((8, 0), CPU, F32).tolist()   # row 1's mask under plb
    assert 1.0 in m8, "need a masked dim for key (8,0) too"
    j1b = m8.index(1.0)
    a2[1] = a2[1] + torch.eye(Da)[j1b]              # row 1: masked-up under (8,0)
    cp = float(plp.compute_loss(_x(a2), s_bar, reduction="sum").detach())
    cpr = float(refp.compute_loss(_x(a2), s_bar, reduction="sum").detach())
    cpb = float(plb.compute_loss(_x(a2), s_bar, reduction="sum").detach())
    # plb (both keyed) minus plp (row 1 unkeyed) == row 1's sigma reward
    lv_r1 = -1.0 + 0.2 * wv * a2[1]
    lv0_r1 = -1.0 + 0.2 * wv * x0[1]
    sig_row1 = -min(max(float((plb._mask_for((8, 0), CPU, F32)
                               * (lv_r1 - lv0_r1)).sum()), 0.0), 1.25)
    assert abs((cpb - cp) - sig_row1) < 1e-5, \
        "rows without job context must contribute no sigma term"
    assert cp < cpr, "row 0 (keyed, masked-up) still gets its sigma reward"

    # ---- reductions: sum == mean * B, bogus raises ------------------------- #
    plb.set_row_jobs(jobs)
    plb.select_z(x0.unsqueeze(1), s_bar)
    ssum = plb.compute_loss(_x(x0), s_bar, reduction="sum")
    smean = plb.compute_loss(_x(x0), s_bar, reduction="mean")
    assert torch.allclose(ssum, smean * x0.shape[0], atol=1e-6)
    try:
        plb.compute_loss(_x(x0), s_bar, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass

    print("[smoke_entropyseek 方案3-sigma] OK: mask determinism/variability/"
          "rho0/rho1/one-redraw, rho=0 byte-equal to atypical (cost+grad, "
          "both reductions), no-job fallback, w_A=2 anchor-only scaling, "
          "sigma direction (masked-up rewarded j1=%d, masked-down / "
          "unmasked-up/down exactly 0 j0=%d), kappa_sigma clamp (-1.25 flat, "
          "zero grad), partial jobs, reductions" % (j1, j0))


if __name__ == "__main__":
    main()
