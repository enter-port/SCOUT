# SCOUT

Implementation of **classifier-guided exploration via a VIB latent dynamics model**:
a frozen base Diffusion Policy provides the action score at test time, while a
separately-trained VIB model supplies the skill encoder `q_φ(z | s̄, a)` whose
posterior movement is used as the **entropy cost**
`-min(KL(q_φ(z|s̄,a) ‖ q_φ(z|s̄,a⁰)), κ)` (candidate action vs the DP's own
unguided intent `a⁰`, capped at κ nats) injected into the DP denoising loop.
Derivation: [`idea/entropy_cost.md`](../idea/entropy_cost.md). The earlier
Gaussian-NLL cost with a prior-sampled target z (v0) is kept in
`scout/guidance/cost.py` (`--guide dyn` / `expert`).

Current experiments: robomimic `can` (LPB-aligned image pipeline).

- Authoritative design: [`idea/scout_design.md`](../idea/scout_design.md)
- Entropy cost derivation: [`idea/entropy_cost.md`](../idea/entropy_cost.md)
- Implementation plan: [`idea/scout_impl_plan.md`](../idea/scout_impl_plan.md)
- Experiment spec: [`idea/stage1_plan.md`](../idea/stage1_plan.md)

## Layout

```
scout/
  data/    # TransitionSource interface + ReplayBuffer + robomimic low_dim backend
  model/   # EncoderMLP (SOE port); StateEncoder (E_s, LPB-style) / VIB / ScoutVIB
configs/   # YAML configs (Phase 2+)
```

## Status

Stage 1 implemented (see root `README.md`); the formal entropy-cost experiment
(can, 3 seeds × DP/SCOUT arms, `--guide atypical`) has been running since
2026-08-24.
