"""Read the can entropy campaign s233 wandb history as the calibration
baseline reference (user order 2026-09-02: no fresh atypical/placebo arms)."""
import wandb

api = wandb.Api()
runs = sorted(
    api.runs("jiachunbao-sjtu/CAN-8-24-entropy-s233"),
    key=lambda r: r.name,
)
for r in runs:
    s = r.summary
    keys = [
        "eval/success_rate",
        "explore/pass_at_5",
        "explore/pass@10",
        "explore/rescued",
        "final/eval_success_rate",
    ]
    vals = {k: s.get(k) for k in keys if s.get(k) is not None}
    print(f"{r.name:24s} state={r.state:9s} {vals}")
