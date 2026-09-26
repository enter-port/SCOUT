# Shared-base campaigns

Register a full experiment with `scripts/experiment_registry.py` first. Then:

```bash
python -m scripts.launch_campaign --experiment-id TOOL_HANG-2026-09-25-01 \
  --seeds 233 2333 23333 --methods DP SCOUT-aty --core-demos 40 \
  --gpus 0 1 2 3 4 5 --beta 1e-6 --rounds 6 \
  --wandb-prefix TOOL_HANG-2026-09-25
```

`--methods DP` or `--methods SCOUT-aty` selects one method. Supply one GPU per
seed/method pair. GPU7 is reserved. `--prepare-only` writes the core provenance,
configs and manifest without starting training. The seed controls a uniform
sample without replacement from the official dataset. Each seed trains one
shared base DP/dyn pair before its method chains start. Jobs use separate OS
sessions, so stopping an ATY does not close DP's controlling terminal.

The defaults preserve the confirmed protocol: DP600/dyn300, both batch64,
eight exploration workers with25 environments each,100 fixed scenes,
pass@5 stopping at first success, five retraining rounds plus a final full
evaluation/exploration round. Round1 uses kappa2.5/R=.01; rounds2–5 use R/C
with bracket C calibration. R's iteration budget is8 and still requires the
original .009–.011 band. Online W&B uses one native ID/name per round, with
evaluation, merged exploration, DP and dyn resuming it sequentially.

For the authorized beta gate, run:

```bash
python -m scripts.atom.beta_gate check --root data/TOOL_HANG/full/ID
```

The command checks process identities and complete round1 receipts. After all
three pairs finish, a per-seed rescue deficit >=4 pauses the three ATYs on that
server. Their W&B round runs and ATY output directories are removed; DP and
shared bases are preserved. Calibration is separately registered, uses the
largest-deficit seed and DP's frozen failures, and tries
`1e-7,3e-7,1e-6,3e-6,1e-5,3e-5` on the three released GPUs in parallel batches.
Every candidate still uses eight workers x25 envs, no W&B or trajectory HDF5.
All complete candidates are compared by rescued count; ties prefer the initial
beta, then smaller beta. Nonconvergent R candidates are explicitly unusable.
Failed infrastructure stages require recovery, not a fabricated score.

The selected beta restarts all three ATYs in the same experiment/projects.
Only the selected seed can reuse its search dyn; others train a new base dyn
online. Checks request60-minute intervals after replacement, and completion
once all three ATYs enter round2 DP retraining. Without replacement checks
remain30-minute until that point. `check_beta_campaigns.ps1` checks both servers.
The desktop automation owns timing and notifications; no extra W&B publisher
or shadow run is started.

Recovery: inspect the saved error and exact job identity first. A failed formal
job can resume with `python -m scripts.atom.parallel_campaign job --manifest
<root>/manifest.json --name s233-aty` (or `s233-dp`). Completed stage receipts
are reused. Do not restart the full supervisor during beta replacement. Search
recovery uses `python -m scripts.atom.beta_gate candidate --config <exact JSON>`;
after failed candidates recover, resume `search --root <aux root>`.

The memory optimization changes only buffered storage: retain the last frame
in the exact HDF5 uint8 layout instead of two float32 frames per observation.
Policy inputs, sampling, actions, retries, metrics, HDF5 values and ordering are
unchanged. Completed slot references are released after the spool consumes them.
Metric-only calibration opts out of observation recording entirely.
