"""V3: per-scene turnover matrix from scene_detail.jsonl.
For each seed/arm:
- failed-set membership per round; rescue outcome per (round, scene)
- composition of late failed sets (r5/r6): scenes previously RESCUED by the
  same arm's guidance (machinery-lost) vs scenes NEVER rescued in any round
  (hard-core candidates) vs scenes new to the failed pool
- r1 cross-arm table (all three arms share the identical r1 failed set)."""
import json
from collections import defaultdict

recs = [json.loads(l) for l in open(
    "experiments/2026_9_21_ATY_decline_analysis/scene_detail.jsonl", encoding="utf-8")]
E = {}  # (seed,arm,rnd) -> {scene: solved}
for r in recs:
    E[(r["seed"], r["arm"], r["rnd"])] = {
        d["init"]: bool(d["solved"]) for d in (r["detail"] or [])}

for seed in ("233", "2333", "23333"):
    print(f"===== seed {seed} =====")
    # r1 cross-arm shared set
    f1 = {a: set(E[(seed, a, 1)]) for a in ("DP", "ATY", "ORBIT")}
    print("r1 failed sets equal across arms:",
          f1["DP"] == f1["ATY"] == f1["ORBIT"], "| size", len(f1["ATY"]))
    for a in ("ATY", "ORBIT", "DP"):
        ever_rescued = {s for r in range(1, 7) for s, ok in E[(seed, a, r)].items() if ok}
        print(f"-- {a}: ever-rescued scenes ({len(ever_rescued)}): {sorted(ever_rescued)}")
        for rnd in (5, 6):
            fr = set(E[(seed, a, rnd)])
            if not fr:
                continue
            prev_resc = fr & ever_rescued
            # scenes that THIS arm rescued in an earlier round AND appear again now
            again = fr & {s for r in range(1, rnd) for s, ok in E[(seed, a, r)].items() if ok}
            # scenes seen in earlier failed sets and never rescued then or now
            hard = {s for s in fr
                    if not any(E[(seed, a, r)].get(s, False) for r in range(1, rnd + 1))}
            print(f"   r{rnd}: |F|={len(fr):2d}  previously-rescued-by-self={len(again):2d}"
                  f"  ever-rescued-any-round={len(prev_resc & again) if False else len(again)}"
                  f"  never-rescued-any-round={len(hard)}")
        # rescue rate on scenes by 'rescue history'
        for rnd in range(2, 7):
            fr = E[(seed, a, rnd)]
            old_ok = {s for r in range(1, rnd) for s, ok in E[(seed, a, r)].items() if ok}
            grp = {"was_rescued_before": [ok for s, ok in fr.items() if s in old_ok],
                   "never_rescued_before": [ok for s, ok in fr.items() if s not in old_ok]}
            line = f"   r{rnd}: "
            for g, v in grp.items():
                if v:
                    line += f"{g} {sum(v)}/{len(v)}  "
            print(line)
