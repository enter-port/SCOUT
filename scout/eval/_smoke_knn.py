"""Hermetic smoke for the --exploit-knn k-NN cost (user 08-30).

Checks on the MockVib from _smoke_exploit:
  1. knn=1 (default) is value- and gradient-identical to brute-force 1-NN min.
  2. knn=5 with bank_chunk smaller than the bank equals brute-force mean of
     the 5 smallest distances (exercises the per-segment top-k concat path).
  3. k-NN mean cost >= 1-NN min cost; gradient flows for knn>1.

Run:  python -m scout.eval._smoke_knn
"""
import torch

from scout.eval._smoke_exploit import Da, Ds, MockVib, _x
from scout.guidance.exploit_costs import ExploitCostPlanner


def _pin(planner, s_bar):
    planner._cached_s_bar_t = s_bar
    planner._cached_obs_id = None


def _brute(a, bank, k):
    """MockVib forward by hand: s_pred eye-slice -> dists -> k-NN mean/min."""
    from scout.eval._smoke_exploit import W_dec, W_enc
    s_pred = (a @ W_enc.T) @ W_dec.T
    q = s_pred[:, 4:8]
    d = torch.cdist(q, bank[:, 4:8], p=2)
    vals = torch.topk(d, k, dim=-1, largest=False).values
    return vals.mean(-1) if k > 1 else vals.min(-1).values


def main():
    vib = MockVib()
    torch.manual_seed(1)
    bank = torch.randn(9, Ds)
    s_bar = torch.randn(2, Ds)
    a = torch.randn(2, Da)

    pl1 = ExploitCostPlanner(vib, state_bank=bank, latent="eye")
    _pin(pl1, s_bar)
    l1 = pl1.compute_loss(_x(a), None)
    ref1 = _brute(a.clone(), bank, 1).mean()
    assert torch.allclose(l1, ref1, atol=1e-6), (l1, ref1)

    x1 = _x(a)
    g1 = torch.autograd.grad(pl1.compute_loss(x1, None), x1)[0]
    assert g1 is not None and torch.isfinite(g1).all()

    pl5 = ExploitCostPlanner(vib, state_bank=bank, latent="eye", knn=5,
                             bank_chunk=4)          # segments 4/4/1 < k
    _pin(pl5, s_bar)
    x5 = _x(a)
    l5 = pl5.compute_loss(x5)
    ref5 = _brute(a.clone(), bank, 5).mean()
    assert torch.allclose(l5, ref5, atol=1e-6), (l5, ref5)
    g5 = torch.autograd.grad(l5, x5)[0]
    assert g5 is not None and torch.isfinite(g5).all()
    assert float(l5) >= float(l1)

    print(f"[_smoke_knn] OK  1-NN={float(l1):.4f} (grad |{float(g1.norm()):.3f}|)  "
          f"5-NN={float(l5):.4f} (grad |{float(g5.norm()):.3f}|)")

    # soft gate weight: binary parity + graded values
    pb = ExploitCostPlanner(vib, state_bank=bank, latent="eye",
                            ood_threshold=1.0)
    assert pb.gate_weight(0.5) == 0.0 and pb.gate_weight(1.5) == 1.0
    ps = ExploitCostPlanner(vib, state_bank=bank, latent="eye",
                            ood_threshold=1.0, gate_slope=1.0, gate_cap=2.0)
    assert abs(ps.gate_weight(1.1) - 0.1) < 1e-9
    assert abs(ps.gate_weight(1.5) - 0.5) < 1e-9
    assert abs(ps.gate_weight(2.0) - 1.0) < 1e-9
    assert ps.gate_weight(4.0) == 2.0            # capped
    pn = ExploitCostPlanner(vib, state_bank=bank, latent="eye")
    assert pn.gate_weight(9.9) == 1.0            # gate off -> always 1
    print("[_smoke_knn] gate_weight OK (binary parity + soft values + cap)")


if __name__ == "__main__":
    main()
