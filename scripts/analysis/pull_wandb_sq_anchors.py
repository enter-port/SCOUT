# Pull round-wise SR / pass@10 / dose for square s233 anchor chains from wandb.
import os
with open("/root/workspace/baojiachun/.secrets/wandb.env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k.removeprefix("export ")] = v
import wandb
api = wandb.Api()

def dump(project, want):
    runs = api.runs(f"jiachunbao-sjtu/{project}")
    rows = []
    for r in runs:
        if not any(w in r.name for w in want):
            continue
        h = r.history(samples=4000, pandas=False)
        def last(key):
            vals = [x[key] for x in h if x.get(key) is not None]
            return vals[-1] if vals else None
        def mean_tail(key, n=5):
            vals = [x[key] for x in h if x.get(key) is not None]
            return sum(vals[-n:]) / len(vals[-n:]) if vals else None
        keys = set()
        for x in h:
            keys.update(k for k in x if not k.startswith("_"))
        rows.append((r.name, {k: v for k, v in {
            "SR": last("eval/success_rate"),
            "p@10": last("explore/pass_at_10") or last("explore/pass@10"),
            "rescued": last("explore/rescued") or last("explore/exploration_rescued"),
            "mean_inject_tail": mean_tail("explore/mean_inject") or mean_tail("explore/guidance_mean_inject"),
        }.items() if v is not None}))
    for name, m in sorted(rows):
        print(f"  {project} | {name}: {m}")

print("== SQUARE-8-26-entropy-s233 (aty SCOUT + DP arms)")
dump("SQUARE-8-26-entropy-s233", ["SCOUT-s233-round", "DP-s233-round"])
print("== SQUARE-9-1-orbit-s233 (v3 orbit chain, runs restarted 09-02)")
dump("SQUARE-9-1-orbit-s233", ["SCOUT-s233-round"])
