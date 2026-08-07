# SCOUT

Implementation of **classifier-guided exploration via a VIB latent dynamics model**:
a frozen base Diffusion Policy provides the action score at test time, while a
separately-trained VIB world model supplies skill latents `z ~ N(0,I)` and a
guidance cost `||z - mu(s_t, a)||` injected into the DP denoising loop.

Stage 1 targets the robomimic `lift` low_dim task.

- Authoritative design: [`idea/scout_design.md`](../idea/scout_design.md)
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

Phase 1 (scaffold + data module) in progress.
