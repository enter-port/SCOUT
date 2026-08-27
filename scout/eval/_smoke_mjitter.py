"""Hermetic smoke for rand_mjitter (方案D random metric rotation, 2026-08-27).

Checks (per the campaign contract):
  * b=0 degenerates BYTE-FOR-BYTE to AtypicalCostPlanner (same cap): outputs
    AND gradients, both reductions;
  * s = Dz*softmax(b*xi): sum(s) = Dz, ln_s == log(s), deterministic per
    (init, try), varies with (init, try) and with rand_seed, b=0 -> exact
    ones/zeros;
  * rows without job context fall back to s=1 (== 方案三) even for b>0;
  * closed form: at the intent (a == a^0) KL = 0.5*sum(s - 1 - ln s);
  * cap: far action -> cost == -cap exactly;
  * b>0 cost/gradient differ from 方案三 (the rotation actually bites);
  * differentiable; reduction sum == mean*B.

Run:  python -m scout.eval._smoke_mjitter
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner
from scout.guidance.rand_costs.mjitter import MJitterCostPlanner

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


def _both(b, cap=2.5, seed=42):
    pl = MJitterCostPlanner(MockVib(), rand_b=b, cap=cap, rand_seed=seed)
    ref = AtypicalCostPlanner(MockVib(), cap=cap)
    return pl, ref


def _prepare(pl, jobs, x0, s_bar):
    if jobs is None:
        pl._row_jobs = []
    else:
        pl.set_row_jobs(jobs)
    pl.select_z(x0.unsqueeze(1), s_bar)


def main():
    s_bar = torch.randn(2, Ds)
    cap = 2.5
    jobs = [(None, 7, 3), (None, 8, 0)]             # (state, init_idx, try)
    x0 = torch.randn(2, Da)

    # ---- s bookkeeping ---------------------------------------------------- #
    pl = MJitterCostPlanner(MockVib(), rand_b=1.0, cap=cap, rand_seed=42)
    s0, l0 = pl._s_for((7, 3), torch.device("cpu"), torch.float32)
    s0b, l0b = pl._s_for((7, 3), torch.device("cpu"), torch.float32)
    assert torch.equal(s0, s0b) and torch.equal(l0, l0b), "s must be deterministic"
    assert abs(float(s0.sum()) - Dz) < 1e-5, "sum(s) must equal Dz (16-dim analog)"
    assert torch.allclose(l0, torch.log(s0), atol=1e-6), "ln_s == log(s)"
    assert float(s0.min()) > 0.0
    s_ik = [pl._s_for(k, torch.device("cpu"), torch.float32)[0]
            for k in ((7, 4), (8, 3))]
    assert not torch.allclose(s0, s_ik[0]) and not torch.allclose(s0, s_ik[1]), \
        "s must vary with (init, try)"
    pl2 = MJitterCostPlanner(MockVib(), rand_b=1.0, cap=cap, rand_seed=43)
    s_alt, _ = pl2._s_for((7, 3), torch.device("cpu"), torch.float32)
    assert not torch.allclose(s0, s_alt), "s must vary with rand_seed"
    plz = MJitterCostPlanner(MockVib(), rand_b=0.0, cap=cap, rand_seed=42)
    sz, lz = plz._s_for((7, 3), torch.device("cpu"), torch.float32)
    assert torch.equal(sz, torch.ones(Dz)) and torch.equal(lz, torch.zeros(Dz)), \
        "b=0 -> exact uniform s (byte-exact 方案三)"
    fb_s, fb_l = pl._s_for((None, None), torch.device("cpu"), torch.float32)
    assert torch.equal(fb_s, torch.ones(Dz)) and torch.equal(fb_l, torch.zeros(Dz)), \
        "no job context -> identity rotation"

    # ---- b=0 byte-for-byte equivalence with AtypicalCostPlanner ----------- #
    pl0, ref = _both(b=0.0, cap=cap)
    _prepare(pl0, jobs, x0, s_bar)
    ref.set_row_context([7, 8])                     # no-op for atypical
    ref.select_z(x0.unsqueeze(1), s_bar)
    for red in ("sum", "mean"):
        c0 = pl0.compute_loss(_x(x0), s_bar, reduction=red)
        cr = ref.compute_loss(_x(x0), s_bar, reduction=red)
        assert torch.equal(c0, cr), (red, float(c0), float(cr))
    xg0, xgr = _x(x0), _x(x0)
    pl0.compute_loss(xg0, s_bar, reduction="sum").backward()
    ref.compute_loss(xgr, s_bar, reduction="sum").backward()
    assert torch.equal(xg0.grad, xgr.grad), "b=0 gradients must be bit-equal"

    # ---- fallback: no job rows, b>0 still equals 方案三 -------------------- #
    plf, reff = _both(b=1.0, cap=cap)
    _prepare(plf, None, x0, s_bar)
    reff.select_z(x0.unsqueeze(1), s_bar)
    cf = plf.compute_loss(_x(x0), s_bar, reduction="sum")
    crf = reff.compute_loss(_x(x0), s_bar, reduction="sum")
    assert torch.equal(cf, crf), (float(cf), float(crf))

    # ---- closed form at the intent: KL = 0.5*sum(s - 1 - ln s) ------------ #
    plb, _ = _both(b=1.0, cap=cap)
    _prepare(plb, jobs, x0, s_bar)
    s, ln_s = plb._s_for((7, 3), torch.device("cpu"), torch.float32)
    kl_expect_row0 = 0.5 * float((s - 1.0 - ln_s).sum())
    plb.set_row_jobs([(None, 7, 3)])
    plb.select_z(x0[:1].unsqueeze(1), s_bar)        # row 0's baseline = mu(x0[0])
    c_int = plb.compute_loss(_x(x0[:1]), s_bar, reduction="sum")
    assert abs(float(c_int.detach()) + min(kl_expect_row0, cap)) < 1e-5, \
        (float(c_int.detach()), kl_expect_row0)

    # ---- cap: far action -> cost == -cap exactly -------------------------- #
    far = 50.0 * x0
    for b in (0.0, 1.0):
        plk, _ = _both(b=b, cap=cap)
        _prepare(plk, jobs, x0, s_bar)
        ck = plk.compute_loss(_x(far), s_bar, reduction="sum")
        assert abs(float(ck.detach()) + 2 * cap) < 1e-4, (b, float(ck.detach()))

    # ---- b>0 rotates the force: cost & gradient differ from 方案三 --------- #
    plb, refb = _both(b=1.0, cap=cap)
    _prepare(plb, jobs, x0, s_bar)
    refb.select_z(x0.unsqueeze(1), s_bar)
    # pick an off-intent candidate strictly INSIDE the cap for both planners
    # (once clamped both costs sit at -cap with zero gradient)
    dirn = torch.randn_like(x0)
    a_off, rb = None, None
    for scale in (0.3, 0.15, 0.08, 0.04, 0.02):
        cand = x0 + scale * dirn
        rc = float(refb.compute_loss(_x(cand), s_bar, reduction="sum").detach())
        if rc > -2 * cap + 0.1:
            a_off, rb = cand, rc
            break
    assert a_off is not None, "could not find an unclamped candidate"
    cb = plb.compute_loss(_x(a_off), s_bar, reduction="sum")
    cbf = float(cb.detach())
    assert abs(cbf - rb) > 1e-6, (cbf, rb)
    xgb, xgrb = _x(a_off), _x(a_off)
    plb.compute_loss(xgb, s_bar, reduction="sum").backward()
    refb.compute_loss(xgrb, s_bar, reduction="sum").backward()
    assert torch.isfinite(xgb.grad).all() and float(xgb.grad.norm()) > 0.0
    cos = float(torch.nn.functional.cosine_similarity(
        xgb.grad.flatten(), xgrb.flatten(), dim=0))
    assert cos < 0.999999, f"rotation must change the force direction (cos={cos})"

    # ---- reductions: sum == mean * B --------------------------------------- #
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

    print("[smoke_mjitter 方案D] OK: s determinism/variety/sum=Dz/ln_s, "
          "b=0 byte-equal to atypical (cost+grad, both reductions), "
          "no-job fallback, closed form at intent, cap clamp, "
          "b>0 rotates force (cos=%.4f), differentiable, reductions" % cos)


if __name__ == "__main__":
    main()
