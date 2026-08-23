"""Hermetic smoke for the entropy-cost planners, v2 API (executed-code
buffers + on_try_done + graph-connected empty rows).

Run:  python -m scout.eval._smoke_entropy
"""
import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, NoveltyCostPlanner

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

    def encode(self, obs):
        return obs                                  # s_bar passthrough


def _x(a):
    return a.unsqueeze(1).clone().requires_grad_(True)


def main():
    vib = MockVib()
    s_bar = torch.randn(2, Ds)

    # ---------------- novelty v2 ------------------------------------------- #
    pl = NoveltyCostPlanner(vib, h_scale=5.0, sample_z=False,
                            obs_adapter=lambda d: (torch.as_tensor(d["v"])[None]
                                                   if isinstance(d, dict) else d))
    pl.set_current_obs(s_bar)
    pl.set_row_context([7, 8])

    # all-empty buffers: cost is a graph-connected zero -> backward works
    x = _x(torch.randn(2, Da))
    loss = pl.compute_loss(x, s_bar, reduction="sum")
    loss.backward()
    assert x.grad is not None and float(x.grad.abs().sum()) == 0.0

    # try 0 of scene 7 executed two chunks -> codes committed at try end
    acts0 = [np.random.randn(1, Da).astype(np.float32) for _ in range(2)]  # 1-step chunks (mock encoder is per-step)
    codes0 = [pl.encode_executed({"v": s_bar[0].numpy()}, c) for c in acts0]
    pl.on_try_done(7, codes0)
    assert len(pl._buffers[7]) == 2

    # candidate near the executed region -> HIGHER cost than one far away
    a_near = torch.stack([torch.from_numpy(acts0[0][0]).float(),
                          torch.randn(Da)])
    a_far = a_near + 2.0
    c_near = pl.compute_loss(_x(a_near), s_bar, reduction="sum")
    c_far = pl.compute_loss(_x(a_far), s_bar, reduction="sum")
    assert c_far < c_near, (c_near.item(), c_far.item())

    # gradient direction pushes row 0 AWAY from the executed codes
    xg = _x(a_near)
    pl.compute_loss(xg, s_bar, reduction="sum").backward()
    d0 = (a_near[0] @ W.T - codes0[0]).norm()
    d1 = ((a_near[0] - 0.1 * xg.grad[0]) @ W.T - codes0[0]).norm()
    assert d1 > d0, "gradient must push away from executed codes"

    # a fresh-scene row (empty buffer) contributes exactly 0
    pl.set_row_context([8])
    c2 = pl.compute_loss(_x(a_near[1:]), s_bar, reduction="sum")
    assert float(c2.detach()) == 0.0
    pl.set_row_context([7, 8])

    # batch invariance: row 0's cost alone == its cost in the batch
    c_alone = pl.compute_loss(_x(a_near[:1]), s_bar, reduction="sum")
    c_batch = pl.compute_loss(_x(a_near), s_bar, reduction="sum")
    assert torch.allclose(c_alone, c_batch, atol=1e-5)

    # ---------------- atypical --------------------------------------------- #
    pa = AtypicalCostPlanner(vib, cap=2.0)
    pa.set_current_obs(s_bar)
    x0 = torch.randn(2, Da)
    pa.select_z(x0.unsqueeze(1), s_bar)
    same = pa.compute_loss(_x(x0), s_bar, reduction="sum")
    moved = pa.compute_loss(_x(x0 + 1.0), s_bar, reduction="sum")
    assert moved < same
    huge = pa.compute_loss(_x(x0 + 50.0), s_bar, reduction="sum")
    assert abs(huge.item() + 2 * 2.0) < 1e-4, huge.item()

    print("[smoke_entropy v2] OK: empty-row graph safety, executed-code "
          "buffers, direction, batch invariance, atypical cap")


if __name__ == "__main__":
    main()
