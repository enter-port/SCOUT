"""Hermetic smoke for the entropy-cost planners (novelty / atypical).

Pure mocks -- no robomimic. The mock encoder is mu = W @ a (+ constant logvar),
so gradients w.r.t. the candidate action flow through a linear map and the
"away from the visited code" direction is checkable in closed form.

Run:  python -m scout.eval._smoke_entropy
"""
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, NoveltyCostPlanner


Ds, Da, Dz = 6, 5, 4          # s_bar dim / action dim / style dim
torch.manual_seed(0)
W = torch.randn(Dz, Da)


class MockEncNet:
    action_dim = Da

    def __call__(self, s_bar, a):
        mu = a @ W.T                                # (B, Dz)
        logvar = torch.full_like(mu, -1.0)          # sigma ~ 0.61, constant
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
    """wrap a flat action (B, Da) as an x0_hat of shape (B, T=1, Da)."""
    return a.unsqueeze(1).clone().requires_grad_(True)


def main():
    vib = MockVib()
    s_bar = torch.randn(2, Ds)

    # ---------------- novelty (方案二) -------------------------------------- #
    pl = NoveltyCostPlanner(vib, h_scale=1.0, sample_z=False)
    pl.set_current_obs(s_bar)
    pl.set_row_context([7, 8])                       # two scenes

    # empty buffers -> zero cost, zero grad
    x = _x(torch.randn(2, Da))
    assert pl.compute_loss(x, s_bar).item() == 0.0

    # chunk 1: select_z parks anchors; chunk 2: previous committed to buffer
    pl.select_z(x.detach(), s_bar)
    pl.select_z(x.detach(), s_bar)                   # second call commits
    assert len(pl._buffers[7]) == 1 and len(pl._buffers[8]) == 1

    # candidate AT the buffered code of scene 7 -> cost ~ log(1+eps) ~ 0
    # candidate FAR  from it -> cost ~ log(eps) < 0 (reward for novelty)
    a0 = torch.linalg.pinv(W) @ pl._buffers[7][0]    # preimage: c = W a0
    at = torch.stack([a0, torch.randn(Da)])
    cost_near = pl.compute_loss(_x(at), s_bar, reduction="sum")
    af = at + 0.5
    cost_far = pl.compute_loss(_x(af), s_bar, reduction="sum")
    assert cost_far < cost_near - 2.0, (cost_near.item(), cost_far.item())

    # gradient direction: -grad must push the code AWAY from the buffer
    xg = _x(at)
    pl.compute_loss(xg, s_bar, reduction="sum").backward()
    d_far = (af[0] @ W.T - pl._buffers[7][0]).norm()
    d_step = ((at[0] - 0.1 * xg.grad[0]) @ W.T - pl._buffers[7][0]).norm()
    assert d_step > (at[0] @ W.T - pl._buffers[7][0]).norm(), "grad sign wrong"

    # batch invariance: row 0's cost identical alone vs in a batch of 2
    c1 = pl.compute_loss(_x(at[:1]), s_bar, reduction="sum")
    c2 = pl.compute_loss(_x(at), s_bar, reduction="sum") \
        - pl.compute_loss(_x(at[1:]), s_bar, reduction="sum")
    assert torch.allclose(c1, c2, atol=1e-5), (c1.item(), c2.item())

    # per-scene isolation: scene 8's buffer is independent of scene 7's
    pl._buffers[8].append(pl._buffers[7][0] + 10.0)
    c8 = pl.compute_loss(_x(at[1:]), s_bar, reduction="sum")
    assert c8.item() < -2.0, "scene 8 should now be pulled away too"

    # ---------------- atypical (方案三) ------------------------------------- #
    pa = AtypicalCostPlanner(vib, cap=2.0)
    pa.set_current_obs(s_bar)
    x0 = torch.randn(2, Da)
    pa.select_z(x0.unsqueeze(1), s_bar)              # baseline from x0
    same = pa.compute_loss(_x(x0), s_bar, reduction="sum")
    moved = pa.compute_loss(_x(x0 + 1.0), s_bar, reduction="sum")
    assert moved < same, "moving away from own baseline must lower the cost"
    # cap: a huge deviation must saturate at -cap per row (2 rows -> -2*cap)
    huge = pa.compute_loss(_x(x0 + 50.0), s_bar, reduction="sum")
    assert abs(huge.item() + 2 * 2.0) < 1e-4, huge.item()

    print("[smoke_entropy] OK: novelty(buffer lifecycle, direction, batch "
          "invariance, per-scene isolation) + atypical(direction, cap)")


if __name__ == "__main__":
    main()
