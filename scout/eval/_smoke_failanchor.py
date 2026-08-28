"""Hermetic smoke for rand_failanchor (反锚定, rand LESSONS 提案 1, 2026-08-28).

Checks (per the campaign contract):
  * retry0 mode, BEFORE the anchor freezes: retry-0 rows are BYTE-EQUAL to
    AtypicalCostPlanner (方案三) -- cost AND gradient, both reductions;
    no-job rows likewise;
  * retry0 accumulation: every try-0 chunk's μ⁰ lands in the accumulator;
    the anchor frozen at the scene's first on_try_done == MEAN of those μ⁰
    (executed codes must NOT contaminate it);
  * retry0 after freeze: try>=1 rows use the anchored KL
    C = -min(KL(q_a ‖ N(anchor, σ⁰²)), κ)  (verified against BOTH a hacked
    方案三 reference and an independent closed-form computation); try-0 rows
    STILL byte-equal 方案三;
  * trailing mode: encode_executed returns μ(s̄, executed chunk) (checked
    against a direct MockVib call); on_try_done commits codes and the anchor
    == per-dim trimmed mean (q=0.2, hand-checked incl. N=1/N=2 edges); the
    anchor DRIFTS as more tries commit; try 0 (no anchor yet) == 方案三;
    a successful retry's codes are included (no success flag);
  * serialization contract: planner exposes on_try_done AND encode_executed
    (this is what activates rollout_vec's per-scene job gate);
  * reductions: sum == mean * B; bogus reduction raises;
  * cap: a far-anchor row saturates at exactly -κ with zero gradient.

Run:  python -m scout.eval._smoke_failanchor
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner
from scout.guidance.rand_costs.failanchor import (
    FailAnchorCostPlanner, _trimmed_mean)

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
        return obs                                  # s_bar passthrough


def _x(a):
    return a.unsqueeze(1).clone().requires_grad_(True)


def _ref_cost(a_intent, a_cand, s_bar, cap=2.5, m0_override=None):
    """Plain 方案三 single-row cost (optionally with a swapped reference
    mean -- the anchored formula via the SAME class code path)."""
    ref = AtypicalCostPlanner(MockVib(), cap=cap)
    ref.set_current_obs(s_bar)
    ref.select_z(a_intent.unsqueeze(1), s_bar)
    if m0_override is not None:
        ref._base_mu = [m0_override.clone()]
    return ref


def _prepare(pl, jobs, x0, s_bar):
    if jobs is not None:
        pl.set_row_jobs(jobs)
    pl.set_current_obs(s_bar)
    pl.select_z(x0.unsqueeze(1), s_bar)


def main():
    s_bar = torch.randn(2, Ds)
    cap = 2.5
    x0 = torch.randn(2, Da)
    # candidate strictly INSIDE the cap so gradients stay live
    delta = torch.randn(2, Da)
    _r0 = AtypicalCostPlanner(MockVib(), cap=cap)
    _r0.set_current_obs(s_bar)
    _r0.select_z(x0.unsqueeze(1), s_bar)
    for _sc in (0.35, 0.15, 0.08, 0.04):
        xc = x0 + _sc * delta
        if float(_r0.compute_loss(_x(xc), s_bar,
                                  reduction="sum").detach()) > -2 * cap + 0.5:
            break

    # ---- retry0: pre-freeze rows are byte-equal 方案三 --------------------- #
    pl = FailAnchorCostPlanner(MockVib(), anchor_mode="retry0", cap=cap)
    ref = _ref_cost(x0, xc, s_bar, cap)             # (retry-0 row semantics)
    _prepare(pl, [(None, 7, 0), (None, 8, 3)], x0, s_bar)
    # try 3 of scene 8 also unanchored (nothing frozen yet) -> both rows
    # must match 方案三 exactly
    for red in ("sum", "mean"):
        c1 = pl.compute_loss(_x(xc), s_bar, reduction=red)
        cr = ref.compute_loss(_x(xc), s_bar, reduction=red)
        assert torch.equal(c1, cr), (red, float(c1), float(cr))
    assert abs(float(c1.detach())) > 1e-3, "need a nonzero-cost candidate"
    xg1, xgr = _x(xc), _x(xc)
    pl.compute_loss(xg1, s_bar, reduction="sum").backward()
    ref.compute_loss(xgr, s_bar, reduction="sum").backward()
    assert torch.equal(xg1.grad, xgr.grad), "pre-freeze grads must be equal"

    # no-job fallback: byte-equal too
    plnj = FailAnchorCostPlanner(MockVib(), anchor_mode="retry0", cap=cap)
    _prepare(plnj, None, x0, s_bar)
    assert torch.equal(plnj.compute_loss(_x(xc), s_bar, reduction="sum"),
                       ref.compute_loss(_x(xc), s_bar, reduction="sum"))

    # ---- retry0: accumulation + freeze semantics -------------------------- #
    pl2 = FailAnchorCostPlanner(MockVib(), anchor_mode="retry0", cap=cap)
    mus = []
    for c in range(3):                              # 3 try-0 chunks
        x0c = x0[:1] + 0.05 * c
        _prepare(pl2, [(None, 7, 0)], x0c, s_bar[:1])
        mus.append((x0c @ W.T).detach().clone())    # μ⁰ of this chunk
    assert pl2._acc[7] is not None and len(pl2._acc[7]) == 3
    for got, want in zip(pl2._acc[7], mus):
        assert torch.allclose(got, want[0], atol=1e-6), "μ⁰ must accumulate"
    # first on_try_done = retry 0 finalize (job-gate order); executed codes
    # must NOT contaminate the retry0 anchor
    fake_codes = [torch.randn(Dz) for _ in range(5)]
    pl2.on_try_done(7, fake_codes)
    want_anchor = torch.stack(mus).mean(dim=0)[0]
    assert torch.allclose(pl2._anchor[7], want_anchor, atol=1e-6)
    assert 7 not in pl2._acc, "accumulator must be popped at freeze"
    # later on_try_done calls must NOT move the frozen anchor
    pl2.on_try_done(7, [torch.randn(Dz)])
    assert torch.allclose(pl2._anchor[7], want_anchor, atol=1e-6)
    # no further accumulation after retry 0 finalized
    _prepare(pl2, [(None, 7, 0)], x0[:1], s_bar[:1])
    assert len(pl2._acc) == 0

    # ---- retry0: post-freeze anchored KL (try>=1) ------------------------- #
    xg = _x(xc)
    pl2.set_row_jobs([(None, 7, 1), (None, 9, 0)])
    pl2.set_current_obs(s_bar)
    pl2.select_z(x0.unsqueeze(1), s_bar)            # fresh chunk baseline
    got = float(pl2.compute_loss(xg, s_bar, reduction="sum").detach())
    # row 0 (scene 7 try 1): 方案三 with base_mu swapped to the anchor;
    # row 1 (scene 9 try 0): plain 方案三 on ITS OWN row tensors.
    ra = _ref_cost(x0[0:1], xc[0:1], s_bar[0:1], cap,
                   m0_override=pl2._anchor[7])
    r1 = _ref_cost(x0[1:2], xc[1:2], s_bar[1:2], cap)
    want = (float(ra.compute_loss(_x(xc[0:1]), s_bar[0:1],
                                  reduction="sum").detach())
            + float(r1.compute_loss(_x(xc[1:2]), s_bar[1:2],
                                    reduction="sum").detach()))
    assert abs(got - want) < 1e-5, (got, want)
    # independent closed form for the anchored row
    with torch.no_grad():
        mu_a, lv_a = MockVib().vib_enc(s_bar[:1], xc[:1])
        mu0, lv0 = MockVib().vib_enc(s_bar[:1], x0[:1])
        kl = 0.5 * (((mu_a[0] - want_anchor) ** 2 / torch.exp(lv0[0]))
                    + torch.exp(lv_a[0]) / torch.exp(lv0[0]) - 1.0
                    - (lv_a[0] - lv0[0])).sum()
    assert abs((float(pl2.compute_loss(_x(xc[:1]), s_bar[:1],
                                       reduction="sum").detach()))
               - (-min(float(kl), cap))) < 1e-5, "closed-form KL mismatch"
    # gradient flows on the anchored row (anchor NEAR the intent so the
    # KL sits below the cap -- a far anchor saturates, see next block)
    pl2b = FailAnchorCostPlanner(MockVib(), anchor_mode="retry0", cap=cap)
    near = (x0[:1] @ W.T).detach()[0] * 0.9        # μ⁰ of the chunk, shrunk
    pl2b.on_try_done(7, [near])
    pl2b.set_row_jobs([(None, 7, 4)])
    pl2b.set_current_obs(s_bar[:1])
    pl2b.select_z(x0[:1].unsqueeze(1), s_bar[:1])
    xga = _x(xc[:1])
    pl2b.compute_loss(xga, s_bar[:1], reduction="sum").backward()
    assert float(xga.grad.abs().sum()) > 0.0, "anchored row grad must flow"

    # cap saturation: anchor far away -> cost == -cap exactly, zero grad
    plfar = FailAnchorCostPlanner(MockVib(), anchor_mode="trailing", cap=cap)
    far = torch.full((Dz,), 50.0)
    plfar._anchor[7] = far.clone()
    plfar.set_row_jobs([(None, 7, 2)])
    plfar.set_current_obs(s_bar[:1])
    plfar.select_z(x0[:1].unsqueeze(1), s_bar[:1])
    xgf = _x(xc[:1])
    cf = plfar.compute_loss(xgf, s_bar[:1], reduction="sum")
    assert float(cf.detach()) == -cap, (float(cf.detach()), -cap)
    cf.backward()
    assert float(xgf.grad.abs().sum()) == 0.0, "saturated row must not push"

    # ---- trailing: encode_executed + trimmed-mean drift -------------------- #
    plt = FailAnchorCostPlanner(MockVib(), anchor_mode="trailing",
                                cap=cap, trim_q=0.2)
    obs = {"proprio": s_bar[:1].numpy()}
    chunk = np.random.default_rng(1).normal(size=(1, Da)).astype(np.float32)
    code = plt.encode_executed(obs, chunk)
    with torch.no_grad():
        mu_want, _ = MockVib().vib_enc(s_bar[:1],
                                       torch.as_tensor(chunk).reshape(1, -1))
    assert torch.allclose(code, mu_want[0], atol=1e-5), \
        "encode_executed must return μ(s̄, executed chunk)"
    # try 0 done -> anchor = its codes (N=2, k=int(2*.2)=0 -> plain mean)
    c0 = [torch.randn(Dz), torch.randn(Dz)]
    plt.on_try_done(7, c0)
    assert torch.allclose(plt._anchor[7], torch.stack(c0).mean(dim=0))
    # try 1 done -> anchor = trimmed mean over ALL 5 codes (drifts)
    c1 = [torch.randn(Dz), torch.randn(Dz), torch.randn(Dz)]
    plt.on_try_done(7, c1)
    allc = torch.stack(c0 + c1)                     # (5, Dz)
    k = int(5 * 0.2)
    want_t = torch.sort(allc, dim=0).values[k:5 - k].mean(dim=0)
    assert torch.allclose(plt._anchor[7], want_t, atol=1e-6), \
        "trailing anchor must be the per-dim trimmed mean and drift"
    # successful retry's codes included (no success flag) -- same path, ok
    # try 0 of an UNanchored scene in trailing mode == 方案三
    plt.set_row_jobs([(None, 11, 0)])
    plt.set_current_obs(s_bar[:1])
    plt.select_z(x0[:1].unsqueeze(1), s_bar[:1])
    rt = _ref_cost(x0[:1], xc[:1], s_bar[:1], cap)
    assert torch.equal(
        plt.compute_loss(_x(xc[:1]), s_bar[:1], reduction="sum"),
        rt.compute_loss(_x(xc[:1]), s_bar[:1], reduction="sum"))
    # trailing try>=1 on anchored scene uses the drifting anchor
    plt.set_row_jobs([(None, 7, 2)])
    plt.set_current_obs(s_bar[:1])
    plt.select_z(x0[:1].unsqueeze(1), s_bar[:1])
    rt2 = _ref_cost(x0[:1], xc[:1], s_bar[:1], cap,
                    m0_override=plt._anchor[7])
    assert torch.equal(
        plt.compute_loss(_x(xc[:1]), s_bar[:1], reduction="sum"),
        rt2.compute_loss(_x(xc[:1]), s_bar[:1], reduction="sum"))

    # ---- _trimmed_mean edges ---------------------------------------------- #
    one = [torch.tensor([1.0, 2.0])]
    assert torch.allclose(_trimmed_mean(one, 0.2), one[0])
    two = [torch.tensor([0.0, 0.0]), torch.tensor([2.0, 4.0])]
    assert torch.allclose(_trimmed_mean(two, 0.2), torch.tensor([1.0, 2.0]))
    five = [torch.tensor([float(i)]) for i in range(5)]     # 0,1,2,3,4
    assert float(_trimmed_mean(five, 0.2)[0]) == 2.0        # trim 0 and 4

    # ---- serialization contract + reductions ------------------------------ #
    for mode in ("retry0", "trailing"):
        p = FailAnchorCostPlanner(MockVib(), anchor_mode=mode, cap=cap)
        assert hasattr(p, "on_try_done") and hasattr(p, "encode_executed"), \
            "job-gate hooks must exist in BOTH modes"
    pls = FailAnchorCostPlanner(MockVib(), anchor_mode="trailing", cap=cap)
    pls.on_try_done(7, [torch.randn(Dz)])
    _prepare(pls, [(None, 7, 1), (None, 7, 2)], x0, s_bar)
    ssum = pls.compute_loss(_x(xc), s_bar, reduction="sum")
    smean = pls.compute_loss(_x(xc), s_bar, reduction="mean")
    assert torch.allclose(ssum, smean * x0.shape[0], atol=1e-6)
    try:
        pls.compute_loss(_x(xc), s_bar, reduction="bogus")
        raise AssertionError("bad reduction must raise")
    except ValueError:
        pass
    # make_planner validation
    from scout.guidance.rand_costs.failanchor import make_planner
    p = make_planner(MockVib(), ek={"rand_anchor_mode": "trailing",
                                    "rand_cap": 2.5})
    assert p.anchor_mode == "trailing" and p.cap == 2.5
    try:
        make_planner(MockVib(), ek={"rand_anchor_mode": "bogus"})
        raise AssertionError("bogus anchor mode must raise")
    except ValueError:
        pass

    print(f"[smoke_failanchor 反锚定] OK: retry0 pre-freeze byte-equal 方案三 "
          f"(cost+grad), μ⁰ accumulation + frozen-mean anchor (codes never "
          f"contaminate), post-freeze anchored KL == closed form, try-0 row "
          f"stays 方案三, cap saturation (-κ, zero grad), trailing "
          f"encode_executed μ + drifting trimmed mean (edges N=1/2/5), "
          f"unanchored-trailing == 方案三, job-gate hooks both modes, "
          f"reductions, make_planner validation")


if __name__ == "__main__":
    main()
