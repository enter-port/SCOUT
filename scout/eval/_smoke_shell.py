"""Hermetic smoke for ShellTargetCostPlanner (方案A, user 2026-08-27):
per-retry random target posterior on the kappa-shell of the intent.

Run:  python -m scout.eval._smoke_shell
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner

Ds, Da, Dz = 6, 5, 4          # s_bar dim / action dim / style dim
torch.manual_seed(0)
W = torch.randn(Dz, Da)


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                                # (B, Dz)
        logvar = torch.full_like(mu, -1.0)
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


def _action_for_mu(target_mu):
    """Mock encoder is linear (mu = a @ W.T); solve for an action hitting
    target_mu exactly (W is Dz x Da with Da > Dz -> consistent lstsq)."""
    sol = torch.linalg.lstsq(W, target_mu.unsqueeze(1))     # mu = a@W.T -> W a = t
    return sol.solution.squeeze(-1)


def main():
    vib = MockVib()
    s_bar = torch.randn(2, Ds)
    kappa = 2.5
    pl = ShellTargetCostPlanner(vib, shell_kappa=kappa, shell_seed=42)
    pl.set_current_obs(s_bar)

    # ---- u bookkeeping: deterministic per (init, try), unit norm, varies --- #
    u0 = pl._u_for((7, 3), torch.device("cpu"), torch.float32)
    u0b = pl._u_for((7, 3), torch.device("cpu"), torch.float32)
    u1 = pl._u_for((7, 4), torch.device("cpu"), torch.float32)
    u2 = pl._u_for((8, 3), torch.device("cpu"), torch.float32)
    assert torch.equal(u0, u0b), "u must be deterministic"
    assert abs(float(u0.norm()) - 1.0) < 1e-6
    assert not torch.allclose(u0, u1) and not torch.allclose(u0, u2)

    # ---- baseline capture (same hook semantics as atypical) --------------- #
    x0 = torch.randn(2, Da)
    pl.set_row_jobs([(None, 0, 3), (None, 8, 0)])   # (state, init_idx, try)
    pl.select_z(x0.unsqueeze(1), s_bar)
    assert len(pl._base_mu) == 2 and len(pl._base_lv) == 2

    # ---- cost at the intent == kappa per row (KL(q0||q*) = kappa*||u||^2) - #
    c_intent = pl.compute_loss(_x(x0), s_bar, reduction="sum")
    assert abs(float(c_intent.detach()) - 2 * kappa) < 1e-4, float(c_intent.detach())

    # ---- cost at the target == 0 (quadratic well centered at q*) ---------- #
    # single-row batch so only row 0's target matters
    m0 = pl._base_mu[0]
    lv0 = pl._base_lv[0]
    u_a = pl._u_for((0, 3), m0.device, m0.dtype)
    target_mu = m0 + (2.0 * kappa) ** 0.5 * torch.exp(0.5 * lv0) * u_a
    a_star = _action_for_mu(target_mu)
    pl.set_row_jobs([(None, 0, 3)])
    pl.select_z(x0[:1].unsqueeze(1), s_bar)
    c_intent1 = pl.compute_loss(_x(x0[:1]), s_bar, reduction="sum")
    assert abs(float(c_intent1.detach()) - kappa) < 1e-4
    a_half = _action_for_mu(m0 + 0.5 * (target_mu - m0))
    c_half = pl.compute_loss(_x(a_half[None, :]), s_bar, reduction="sum")
    c_star = pl.compute_loss(_x(a_star[None, :]), s_bar, reduction="sum")
    assert float(c_star.detach()) < 1e-3, float(c_star.detach())
    assert c_star < c_half < c_intent1, (float(c_star.detach()),
                                         float(c_half.detach()),
                                         float(c_intent1.detach()))

    # ---- a descent step on the cost moves row 0's code toward its target --- #
    # (at the intent the cost-gradient is anti-parallel to u: gradient DESCENT
    #  along the injection path tilts the code along +u)
    pl.set_row_jobs([(None, 0, 3), (None, 8, 0)])
    pl.select_z(x0.unsqueeze(1), s_bar)
    xg = _x(x0)
    pl.compute_loss(xg, s_bar, reduction="sum").backward()
    with torch.no_grad():
        a_step = x0[0] - 0.05 * xg.grad[0]
        d_old = (x0[0] @ W.T - target_mu).norm()
        d_new = (a_step @ W.T - target_mu).norm()
    assert d_new < d_old, (float(d_old), float(d_new))

    # ---- per-row targets differ (rows pulled toward their own u) ---------- #
    ca = pl.compute_loss(_x(torch.stack([a_star, a_star])), s_bar,
                         reduction="sum")
    assert float(ca.detach()) > 1e-3, "row 1 (different u) must not be at its target"

    # ---- fallback: no job context at all (used outside the vec runner) ---- #
    pl._row_jobs = []
    pl.select_z(torch.randn(1, Da).unsqueeze(1), s_bar)
    c_fb = pl.compute_loss(_x(torch.randn(1, Da)), s_bar, reduction="sum")
    assert float(c_fb.detach()) == 0.0

    # ---- reductions -------------------------------------------------------- #
    pl.set_row_jobs([(None, 0, 3), (None, 8, 0)])
    pl.select_z(x0.unsqueeze(1), s_bar)
    s = pl.compute_loss(_x(x0), s_bar, reduction="sum")
    m = pl.compute_loss(_x(x0), s_bar, reduction="mean")
    assert torch.allclose(m, s / 2, atol=1e-6)

    print("[smoke_shell 方案A] OK: u determinism/unit/variety, baseline "
          "capture, cost==kappa at intent, cost==0 at target, monotone well, "
          "gradient along +u, per-row targets, fallback zero, reductions")


if __name__ == "__main__":
    main()
