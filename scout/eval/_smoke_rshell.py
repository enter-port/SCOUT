"""Hermetic smoke for rand_rshell (geometry.md 方案B): reachable-subspace
whitening shell -- probe recovers the top singular subspace of the whitened
encoder Jacobian; u_hat lives in col(V_r), deterministic per (init, try);
well math inherited from 方案A (cost=kappa at intent, 0 at target).

Run:  python -m scout.eval._smoke_rshell
"""
import numpy as np
import torch

from scout.guidance.rand_costs.rshell import RShellCostPlanner

Ds, Da, Dz = 6, 5, 4          # s_bar dim / per-step action dim / style dim
NSTEP = 4                     # chunk = NSTEP x Da = 20 dims (probe tau needs >=2 steps)
DCH = NSTEP * Da
torch.manual_seed(0)

# encoder Jacobian with a controlled spectrum: singulars [3, 2, 0.01, 0.01]
# (top-2 subspace well separated -> eps=1e-2 cut must keep exactly r=2)
A_orth = torch.linalg.qr(torch.randn(Dz, Dz)).Q          # (4,4) orthogonal
B_ = torch.linalg.qr(torch.randn(DCH, Dz)).Q             # (20,4) orthonormal
S_TRUE = torch.tensor([3.0, 2.0, 0.01, 0.01])
W = A_orth @ torch.diag(S_TRUE) @ B_.T                   # (4, 20)
V2 = torch.randn(Dz, DCH) / DCH ** 0.5    # nonlinearity mixing (spectral ~1.4)
NL = 0.002                                # ||nl*V2|| << 0.01 small dims
TRUE_TOP2 = A_orth[:, :2]


class MockEncNet:
    action_dim = DCH

    def __init__(self, nl: float = NL):
        self.nl = nl

    def __call__(self, s_bar, a):
        mu = a @ W.T + self.nl * torch.sin(a @ V2.T)      # (B, 4)
        logvar = torch.full_like(mu, -1.0)                # sigma^0 const
        return mu, logvar


class MockVib:
    style_dim = Dz

    def __init__(self, nl: float = NL):
        self.vib_enc = MockEncNet(nl)

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def encode(self, obs):
        return obs                                        # s_bar passthrough


def _x(a_flat):
    return a_flat.reshape(a_flat.shape[0], NSTEP, Da).clone().requires_grad_(True)


def _subspace_angle(V: torch.Tensor, U_true: torch.Tensor) -> float:
    """Largest principal angle (deg) between col(V) and col(U_true)."""
    M = U_true.T @ V
    sv = torch.linalg.svdvals(M).clamp(max=1.0)
    return float(torch.rad2deg(torch.arccos(sv)).max())


def main():
    kappa = 2.5
    pl = RShellCostPlanner(MockVib(nl=NL), alpha=0.0, probe_p=32,
                           probe_eps=1e-2, probe_tau=0.1,
                           kappa=kappa, rand_seed=42)
    s_bar = torch.randn(2, Ds)
    x0 = torch.randn(2, NSTEP, Da)

    # ---- probe: r cut + top-2 subspace recovered (<10 deg) + RNG clean ---- #
    st = torch.get_rng_state()
    pl.set_row_jobs([(None, 3, 1), (None, 4, 0)])
    pl.select_z(x0, s_bar)
    assert torch.equal(st, torch.get_rng_state()), \
        "probe must not touch the global torch RNG stream"
    for i in range(2):
        V, lam = pl._row_V[i], pl._row_lam[i]
        assert V is not None and V.shape == (Dz, 2), (i, V)
        ang = _subspace_angle(V, TRUE_TOP2)
        assert ang < 10.0, (i, ang)
        assert abs(float(lam[0]) - 1.0) < 1e-6
        ratio = float(lam[1])            # true sigma2/sigma1 = 2/3
        assert 0.35 < ratio < 1.05, ratio
        print(f"[row {i}] r=2, angle={ang:.3f} deg, lam2/lam1={ratio:.3f}")

    # ---- u: unit, in col(V_r), deterministic per (i,k), varies, alpha mix -- #
    u0 = pl._u_for(0, (3, 1), torch.device("cpu"), torch.float32)
    u0b = pl._u_for(0, (3, 1), torch.device("cpu"), torch.float32)
    assert torch.equal(u0, u0b), "u must be deterministic per (i,k)"
    u_k2 = pl._u_for(0, (3, 2), torch.device("cpu"), torch.float32)
    u_i2 = pl._u_for(0, (4, 1), torch.device("cpu"), torch.float32)
    assert not torch.allclose(u0, u_k2) and not torch.allclose(u0, u_i2)
    for u in (u0, u_k2):
        assert abs(float(u.norm()) - 1.0) < 1e-6
        resid = (u - pl._row_V[0] @ (pl._row_V[0].T @ u)).norm()
        assert float(resid) < 1e-5, float(resid)
    # alpha=0.25: still in col(V_r), different direction mix
    pl_a = RShellCostPlanner(MockVib(nl=NL), alpha=0.25, probe_p=32,
                             probe_eps=1e-2, probe_tau=0.1,
                             kappa=kappa, rand_seed=42)
    pl_a.set_row_jobs([(None, 3, 1), (None, 4, 0)])
    pl_a.select_z(x0, s_bar)
    u_a = pl_a._u_for(0, (3, 1), torch.device("cpu"), torch.float32)
    assert float((u_a - pl_a._row_V[0] @ (pl_a._row_V[0].T @ u_a)).norm()) < 1e-5
    assert not torch.allclose(u_a, u0), "alpha must change the direction mix"
    # 10 retries of one scene -> 10 distinct directions (retry diversity)
    us = [pl._u_for(0, (3, k), torch.device("cpu"), torch.float32)
          for k in range(10)]
    uniq = {tuple(np.round(u.numpy(), 6)) for u in us}
    assert len(uniq) == 10, len(uniq)
    dots = [abs(float(us[a] @ us[b])) for a in range(10) for b in range(a + 1, 10)]
    print(f"[u] 10 retries: min |cos| pair = {min(dots):.3f} (r=2 subsphere)")

    # ---- well math on the EXACTLY linear mock (cost checks need inversion) - #
    pl2 = RShellCostPlanner(MockVib(nl=0.0), kappa=kappa, rand_seed=42)
    x1 = torch.randn(1, NSTEP, Da)
    pl2.set_row_jobs([(None, 3, 1)])
    pl2.select_z(x1, s_bar[:1])
    c_int = pl2.compute_loss(_x(x1.reshape(1, -1).detach()), s_bar[:1],
                             reduction="sum")
    assert abs(float(c_int.detach()) - kappa) < 1e-4, float(c_int.detach())

    m0, lv0 = pl2._base_mu[0], pl2._base_lv[0]
    u = pl2._u_for(0, (3, 1), m0.device, m0.dtype)
    t_star = m0 + (2.0 * kappa) ** 0.5 * torch.exp(0.5 * lv0) * u
    # linear encoder: a* @ W.T = t_star exactly (W full row rank)
    sol = torch.linalg.lstsq(W, t_star.unsqueeze(1))
    a_star = sol.solution.squeeze(-1)                    # (DCH,)
    assert float((a_star @ W.T - t_star).norm()) < 1e-4
    a0_flat = x1.reshape(1, -1)[0]
    a_half = 0.5 * (a0_flat + a_star)
    c_star = pl2.compute_loss(_x(a_star[None, :]), s_bar[:1], reduction="sum")
    c_half = pl2.compute_loss(_x(a_half[None, :]), s_bar[:1], reduction="sum")
    c_int1 = pl2.compute_loss(_x(a0_flat[None, :]), s_bar[:1], reduction="sum")
    assert float(c_star.detach()) < 1e-3, float(c_star.detach())
    assert float(c_star.detach()) < float(c_half.detach()) < float(c_int1.detach())

    # ---- gradient descent on the cost moves the code toward the target ----- #
    xg = _x(a0_flat[None, :])
    pl2.compute_loss(xg, s_bar[:1], reduction="sum").backward()
    with torch.no_grad():
        a_step = a0_flat - 0.05 * xg.grad.reshape(-1)
    d_old = float((a0_flat @ W.T - t_star).norm())
    d_new = float((a_step @ W.T - t_star).norm())
    assert d_new < d_old, (d_old, d_new)

    # ---- fallbacks + reductions ------------------------------------------- #
    pl2._row_V[0] = None                     # dead probe row -> no pull
    c_dead = pl2.compute_loss(_x(a0_flat[None, :]), s_bar[:1], reduction="sum")
    assert float(c_dead.detach()) == 0.0
    pl2b = RShellCostPlanner(MockVib(nl=0.0), kappa=kappa, rand_seed=42)
    pl2b._row_jobs = []                      # no job context at all
    pl2b.select_z(x1, s_bar[:1])
    c_fb = pl2b.compute_loss(_x(a0_flat[None, :]), s_bar[:1], reduction="sum")
    assert float(c_fb.detach()) == 0.0
    pl3 = RShellCostPlanner(MockVib(nl=0.0), kappa=kappa, rand_seed=42)
    pl3.set_row_jobs([(None, 3, 1), (None, 4, 0)])
    pl3.select_z(x0, s_bar)
    s = pl3.compute_loss(_x(x0.reshape(2, -1).detach()), s_bar, reduction="sum")
    m = pl3.compute_loss(_x(x0.reshape(2, -1).detach()), s_bar, reduction="mean")
    assert torch.allclose(m, s / 2, atol=1e-6)

    print("[smoke_rshell 方案B] OK: probe r=2 + subspace angle<10deg + RNG "
          "clean, u unit/in-col(V_r)/deterministic/10-retry-distinct, "
          "cost==kappa at intent, ==0 at target, monotone well, descent toward "
          "target, dead-probe & no-job fallbacks zero, sum/mean reductions")


if __name__ == "__main__":
    main()
